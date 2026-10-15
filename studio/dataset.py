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

import cv2
import yaml

from .project import atomic_json


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


def build_dataset(project, val_ratio=0.2, seed=0):
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('검증 비율은 0보다 크고 1보다 작아야 합니다.')
    if not project.classes:
        raise ValueError('클래스 태그를 먼저 등록하세요.')
    included, excluded, warnings = [], [], []
    backgrounds = 0
    for entry in project.images:
        image = entry['file']
        objects, reviewed = project.load_annotation(image)
        if not reviewed:
            excluded.append(dict(image=image, reason='미작업 또는 태그 미지정'))
            continue
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
        backgrounds += not objects
        included.append((image, '\n'.join(rows) + ('\n' if rows else '')))
    if len(included) < 2:
        raise ValueError(f'작업 완료 이미지가 최소 2장 필요합니다. 현재 {len(included)}장입니다.')
    random.Random(seed).shuffle(included)
    nval = min(len(included) - 1, max(1, int(len(included) * val_ratio)))
    splits = dict(train=included[nval:], val=included[:nval])
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
        data = dict(path=str(target), train='images/train', val='images/val',
                    names={c['id']: c['name'] for c in project.classes})
        (staging / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
        atomic_json(staging / 'report.json', dict(seed=seed, val_ratio=val_ratio, backgrounds=int(backgrounds),
            splits={k: [i for i, _ in v] for k, v in splits.items()}, excluded=excluded, warnings=warnings))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging)
        raise
    return target / 'data.yaml'
