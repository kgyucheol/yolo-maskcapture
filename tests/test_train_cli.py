import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def train_module():
    spec = importlib.util.spec_from_file_location('train_cli', Path(__file__).resolve().parents[1] / 'scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_data(tmp_path, label):
    raw = tmp_path / 'raw'
    (raw / 'images').mkdir(parents=True)
    (raw / 'labels').mkdir()
    (raw / 'classes.txt').write_text('컵\n')
    for i in range(2):
        cv2.imwrite(str(raw / 'images' / f'{i}.png'), np.zeros((48, 84, 3), np.uint8))
        (raw / 'labels' / f'{i}.txt').write_text(label)
    return raw


def test_legacy_segmentation_dataset_preserves_existing_output(tmp_path):
    raw = raw_data(tmp_path, '0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n')
    out = tmp_path / 'datasets'
    out.mkdir()
    (out / 'keep.txt').write_text('keep')
    result = train_module().build_dataset(raw, out, 0.2, 0)
    assert result.is_file() and (out / 'keep.txt').read_text() == 'keep'


def test_detection_labels_rejected_in_segmentation_training(tmp_path):
    raw = raw_data(tmp_path, '0 0.5 0.5 0.3 0.3\n')
    with pytest.raises(ValueError, match='세그먼테이션'):
        train_module().build_dataset(raw, tmp_path / 'datasets', 0.2, 0)
