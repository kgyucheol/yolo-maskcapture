"""Validated, reproducible and non-destructive YOLO segmentation snapshots."""
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from dataclasses import asdict

import cv2
import yaml

from .augmentation import AugmentConfig, augmentation_plan, augment_sample, sample_seed
from .project import atomic_json, image_bytes, read_image


def mask_polygons(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons, holes = [], 0
    if hierarchy is None:
        return polygons, holes
    for contour, relation in zip(contours, hierarchy[0]):
        if relation[3] >= 0:
            holes += 1
            continue
        epsilon = 0.001 * cv2.arcLength(contour, True)
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) >= 3 and cv2.contourArea(points) >= 1:
            polygons.append(points)
    return polygons, holes


def objects_to_labels(objects, image, warnings):
    rows = []
    for obj in objects:
        polys, holes = mask_polygons(obj.mask)
        if not polys:
            raise ValueError(f'{image}: 내보낼 수 없는 작은 마스크입니다.')
        if holes or len(polys) > 1:
            warnings.append(dict(image=image, object=obj.id, filled_holes=holes, components=len(polys)))
        height, width = obj.mask.shape
        for points in polys:
            coords = [f'{v:.8f}' for x, y in points for v in (float(x) / width, float(y) / height)]
            rows.append(f'{obj.class_id} ' + ' '.join(coords))
    return '\n'.join(rows) + ('\n' if rows else '')


def split_sources(items, val_ratio, seed):
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('검증 비율은 0보다 크고 1보다 작아야 합니다.')
    if len(items) < 2:
        raise ValueError(f'작업 완료 이미지가 최소 2장 필요합니다. 현재 {len(items)}장입니다.')
    items = list(items)
    random.Random(seed).shuffle(items)
    nval = min(len(items) - 1, max(1, int(len(items) * val_ratio)))
    return dict(train=items[nval:], val=items[:nval])


def build_dataset(project, val_ratio=0.2, seed=0, augmentation=None):
    augmentation = augmentation if augmentation is not None else AugmentConfig(copies=0)
    if not isinstance(augmentation, AugmentConfig):
        raise ValueError('올바른 증강 설정이 필요합니다.')
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('검증 비율은 0보다 크고 1보다 작아야 합니다.')
    if not project.classes:
        raise ValueError('클래스 태그를 먼저 등록하세요.')
    included, excluded, warnings = [], [], []
    annotation_versions = {}
    backgrounds = 0
    for entry in project.images:
        image = entry['file']
        annotation = project.annotation_path(image)
        version = annotation.read_bytes() if annotation.exists() else None
        objects, reviewed = project.load_annotation(image)
        if not reviewed:
            excluded.append(dict(image=image, reason='미작업 또는 태그 미지정'))
            continue
        if annotation.read_bytes() != version:
            raise ValueError(f'{image}: 생성 중 주석이 변경되었습니다. 다시 생성하세요.')
        annotation_versions[image] = version
        backgrounds += not objects
        included.append((image, objects_to_labels(objects, image, warnings)))
    if len(included) < 2:
        raise ValueError(f'작업 완료 이미지가 최소 2장 필요합니다. 현재 {len(included)}장입니다.')
    splits = split_sources(included, val_ratio, seed)
    source_splits = {key: [image for image, _ in pairs] for key, pairs in splits.items()}
    plan = augmentation_plan(augmentation, len(splits['train']) * augmentation.copies, seed)
    augmented = []
    reserved = {Path(image).stem for image, _ in included}
    parent = project.root / 'datasets'
    target = parent / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    staging = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    try:
        for split, pairs in splits.items():
            (staging / 'images' / split).mkdir(parents=True)
            (staging / 'labels' / split).mkdir(parents=True)
            for image, label in pairs:
                shutil.copy2(project.image_path(image), staging / 'images' / split / image)
                (staging / 'labels' / split / (Path(image).stem + '.txt')).write_text(label, encoding='utf-8')
                if split != 'train' or not augmentation.copies:
                    continue
                frame = read_image(project.image_path(image))
                if frame is None:
                    raise ValueError(f'{image}: 증강할 원본 이미지를 읽을 수 없습니다.')
                if project.annotation_path(image).read_bytes() != annotation_versions[image]:
                    raise ValueError(f'{image}: 생성 중 주석이 변경되었습니다. 다시 생성하세요.')
                objects, reviewed = project.load_annotation(image)
                if not reviewed or project.annotation_path(image).read_bytes() != annotation_versions[image]:
                    raise ValueError(f'{image}: 생성 중 주석이 변경되었습니다. 다시 생성하세요.')
                for index in range(augmentation.copies):
                    variant_seed = sample_seed(seed, image, index)
                    recipe = plan[len(augmented)]
                    variant, masks, info = augment_sample(frame, objects, augmentation, variant_seed, methods=recipe)
                    stem = f'_aug_{sample_seed(0, image, 0):016x}_{index + 1:02d}'
                    while stem in reserved:
                        stem += '_'
                    reserved.add(stem)
                    name = stem + '.png'
                    text = objects_to_labels(masks, name, warnings)
                    (staging / 'images/train' / name).write_bytes(image_bytes(variant))
                    (staging / 'labels/train' / (stem + '.txt')).write_text(text, encoding='utf-8')
                    augmented.append(dict(image=name, source=image, index=index, seed=variant_seed,
                                          background=not masks, **info))
        data = dict(path=str(target), train='images/train', val='images/val',
                    names={c['id']: c['name'] for c in project.classes})
        (staging / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
        atomic_json(staging / 'report.json', dict(seed=seed, val_ratio=val_ratio, backgrounds=int(backgrounds),
            splits=dict(train=source_splits['train'] + [item['image'] for item in augmented], val=source_splits['val']),
            source_splits=source_splits, augmentation=dict(config=asdict(augmentation), generated=len(augmented),
                multiplier=augmentation.copies + 1, single=sum(len(s['methods']) == 1 for s in augmented),
                composite=sum(len(s['methods']) > 1 for s in augmented), samples=augmented),
            excluded=excluded, warnings=warnings))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging)
        raise
    return target / 'data.yaml'
