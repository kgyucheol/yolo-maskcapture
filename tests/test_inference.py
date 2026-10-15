from pathlib import Path
import types
import sys

import numpy as np
import pytest

from studio.inference import SegmentationPredictor, ResultGate
from studio.jobs import validate_train_config, run_job


def test_model_switch_and_stop_reject_old_prediction():
    gate = ResultGate()
    first = gate.start()
    assert gate.accepts(first)
    second = gate.start()
    assert not gate.accepts(first) and gate.accepts(second)
    gate.stop()
    assert not gate.accepts(second)


def test_detection_model_rejected_before_inference(tmp_path, monkeypatch):
    path = tmp_path / 'detect.pt'
    path.touch()
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=lambda path, **kw: types.SimpleNamespace(task='detect')))
    with pytest.raises(ValueError, match='세그먼테이션'):
        SegmentationPredictor(path)


def test_missing_model_and_wrong_format_are_clear(tmp_path):
    with pytest.raises(ValueError):
        SegmentationPredictor(tmp_path / 'missing.pt')
    path = tmp_path / 'model.bin'
    path.touch()
    with pytest.raises(ValueError):
        SegmentationPredictor(path)


@pytest.mark.parametrize('change', [dict(imgsz=0), dict(batch=0), dict(epochs=0), dict(imgsz=848), dict(mode='unknown')])
def test_invalid_training_config_rejected(tmp_path, change):
    config = dict(mode='train', data=str(tmp_path / 'data.yaml'), model='yolo11s-seg.pt',
                  imgsz=864, epochs=10, batch=4, device='cpu', rect=False, project=str(tmp_path / 'runs'))
    config.update(change)
    with pytest.raises(ValueError):
        validate_train_config(config)


def test_training_missing_data_does_not_create_success_manifest(tmp_path):
    config = dict(mode='train', data=str(tmp_path / 'missing.yaml'), model='yolo11s-seg.pt',
                  imgsz=864, epochs=1, batch=1, device='cpu', rect=False, project=str(tmp_path / 'runs'))
    with pytest.raises((ValueError, FileNotFoundError)):
        run_job(config)
    assert not list(tmp_path.rglob('result.json'))
