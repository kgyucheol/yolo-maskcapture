import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from studio.document import MaskObject
from studio.project import Project, read_image
from studio.dataset import build_dataset


def sample():
    frame = np.zeros((60, 100, 3), np.uint8)
    frame[10:30, 10:30] = (60, 120, 240)
    mask = np.zeros((60, 100), np.uint8)
    mask[10:30, 10:30] = 255
    return frame, [MaskObject(mask, class_id=2)]


def test_horizontal_flip_transforms_pixels_and_mask_together_without_mutating_source():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    original_frame, original_mask = frame.copy(), objects[0].mask.copy()
    image, transformed, info = augment_sample(frame, objects, AugmentConfig(copies=1), seed=4)
    assert np.array_equal(image, original_frame[:, ::-1])
    assert np.array_equal(transformed[0].mask, original_mask[:, ::-1])
    assert transformed[0].class_id == 2
    assert np.array_equal(frame, original_frame) and np.array_equal(objects[0].mask, original_mask)


def test_geometry_preserves_alignment_and_binary_masks_and_is_reproducible():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    frame[:] = np.repeat((objects[0].mask > 0)[..., None], 3, axis=2) * 255
    config = AugmentConfig(copies=2, horizontal_flip=False, rotation=25, scale=0.15, translate=0.08)
    a, masks_a, info_a = augment_sample(frame, objects, config, seed=12)
    b, masks_b, info_b = augment_sample(frame, objects, config, seed=12)
    assert np.array_equal(a, b) and np.array_equal(masks_a[0].mask, masks_b[0].mask) and info_a == info_b
    assert a.shape == frame.shape and masks_a[0].class_id == 2
    assert set(np.unique(masks_a[0].mask)) <= {0, 255}
    foreground = a[:, :, 0] > 200
    target = masks_a[0].mask > 0
    assert np.count_nonzero(foreground & target) / np.count_nonzero(foreground | target) > 0.9


def test_photometric_changes_preserve_mask_geometry_and_classes():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    config = AugmentConfig(copies=1, horizontal_flip=False, brightness=0.2,
                          contrast=0.2, saturation=0.2, hue=8, noise=4, blur=True)
    image, transformed, info = augment_sample(frame, objects, config, seed=4)
    assert not np.array_equal(image, frame)
    assert np.array_equal(transformed[0].mask, objects[0].mask)
    assert transformed[0].class_id == 2
    assert image.dtype == np.uint8 and info['operations']


@pytest.mark.parametrize('options', [dict(copies=-1), dict(copies=21), dict(rotation=float('nan')),
                                   dict(translate=2), dict(noise=-1), dict(copies=1, horizontal_flip=False)])
def test_invalid_augmentation_settings_are_rejected(options):
    from studio.augmentation import AugmentConfig
    with pytest.raises(ValueError):
        AugmentConfig(**options)


def project_with_labels(tmp_path):
    project = Project.create(tmp_path / 'p')
    for name in ('컵', '병', '상자'):
        project.add_class(name)
    frame, objects = sample()
    for i in range(4):
        name = project.capture(frame + i)
        project.save_annotation(name, objects, True)
    return project


def test_dataset_augments_only_training_split_and_records_reproducible_provenance(tmp_path):
    from studio.augmentation import AugmentConfig
    project = project_with_labels(tmp_path)
    original = {p: p.read_bytes() for sub in ('images', 'annotations', 'masks')
                for p in (project.root / sub).rglob('*') if p.is_file()}
    config = AugmentConfig(copies=2, brightness=0.15, rotation=10)
    a = build_dataset(project, 0.25, 42, augmentation=config).parent
    b = build_dataset(project, 0.25, 42, augmentation=config).parent
    ra, rb = [json.loads((p / 'report.json').read_text()) for p in (a, b)]
    assert ra == rb
    assert len(ra['splits']['train']) == 9 and len(ra['splits']['val']) == 1
    assert ra['augmentation']['generated'] == 6
    assert len(ra['source_splits']['train']) == 3
    for entry in ra['augmentation']['samples']:
        assert entry['source'] in ra['source_splits']['train']
        assert entry['source'] not in ra['splits']['val']
        path = a / 'images/train' / entry['image']
        assert read_image(path).shape == (60, 100, 3)
        assert path.read_bytes() == (b / 'images/train' / entry['image']).read_bytes()
        labels = (a / 'labels/train' / (path.stem + '.txt')).read_text().splitlines()
        assert labels
        for row in labels:
            fields = row.split()
            assert fields[0] == '2' and len(fields) >= 7
            assert all(0 <= float(value) <= 1 for value in fields[1:])
    for name in ra['splits']['val']:
        assert (a / 'images/val' / name).read_bytes() == project.image_path(name).read_bytes()
    assert all(p.read_bytes() == data for p, data in original.items())


def test_background_augmentation_keeps_empty_labels_and_disabled_config_keeps_original_counts(tmp_path):
    from studio.augmentation import AugmentConfig
    project = project_with_labels(tmp_path)
    for entry in project.images:
        project.save_annotation(entry['file'], [], True)
    path = build_dataset(project, 0.25, 42, augmentation=AugmentConfig(copies=1)).parent
    assert len(list((path / 'images/train').glob('*.png'))) == 6
    assert all(not p.read_text() for p in (path / 'labels/train').glob('*.txt'))
    path = build_dataset(project, 0.25, 42, augmentation=AugmentConfig(copies=0)).parent
    assert len(list((path / 'images/train').glob('*.png'))) == 3


def test_reviewed_annotation_change_during_generation_aborts_inconsistent_snapshot(tmp_path, monkeypatch):
    from studio.augmentation import AugmentConfig
    project = project_with_labels(tmp_path)
    marker = project.root / 'datasets/keep.txt'
    marker.write_text('previous dataset')
    original_load = project.load_annotation
    reads = {}

    def edited_annotation(image):
        reads[image] = reads.get(image, 0) + 1
        if reads[image] == 2:
            # A valid reviewed edit can happen while the worker is generating.
            project.save_annotation(image, [], True)
        return original_load(image)

    monkeypatch.setattr(project, 'load_annotation', edited_annotation)
    with pytest.raises(ValueError, match='변경'):
        build_dataset(project, 0.25, 42, augmentation=AugmentConfig(copies=1))
    assert sorted(p.name for p in (project.root / 'datasets').iterdir()) == ['keep.txt']
    assert marker.read_text() == 'previous dataset'


def test_single_augmentation_chooses_one_enabled_method_and_preserves_matching_geometry():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    config = AugmentConfig(copies=1, horizontal_flip=True, brightness=0.2, mode='single')
    seen = set()
    for seed in range(20):
        image, masks, info = augment_sample(frame, objects, config, seed)
        assert len(info['methods']) == 1
        method = info['methods'][0]
        seen.add(method)
        if method == 'horizontal_flip':
            assert np.array_equal(image, frame[:, ::-1])
            assert np.array_equal(masks[0].mask, objects[0].mask[:, ::-1])
        else:
            assert method == 'brightness' and not np.array_equal(image, frame)
            assert np.array_equal(masks[0].mask, objects[0].mask)
    assert seen == {'horizontal_flip', 'brightness'}


def test_random_combination_varies_method_count_and_obeys_maximum():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    config = AugmentConfig(copies=2, horizontal_flip=True, brightness=0.2,
                          noise=3, mode='random', max_transforms=2)
    counts = set()
    for seed in range(20):
        image, masks, info = augment_sample(frame, objects, config, seed)
        methods = info['methods']
        assert 1 <= len(methods) <= 2 and len(methods) == len(set(methods))
        assert set(methods) <= {'horizontal_flip', 'brightness', 'noise'}
        assert np.array_equal(masks[0].mask, objects[0].mask[:, ::-1] if 'horizontal_flip' in methods else objects[0].mask)
        counts.add(len(methods))
    assert counts == {1, 2}


def test_all_mode_applies_selected_flips_and_does_not_choose_disabled_methods():
    from studio.augmentation import AugmentConfig, augment_sample
    frame, objects = sample()
    config = AugmentConfig(copies=1, horizontal_flip=True, vertical_flip=True, mode='all')
    image, masks, info = augment_sample(frame, objects, config, seed=4)
    assert np.array_equal(image, frame[::-1, ::-1])
    assert np.array_equal(masks[0].mask, objects[0].mask[::-1, ::-1])
    assert info['methods'] == ['horizontal_flip', 'vertical_flip']
