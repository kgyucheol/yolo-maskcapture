import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import types

import pytest

from studio.jobs import run_job, validate_train_config
from studio.qt import QtWidgets


def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_training_controls_collect_parameters_and_explain_them(monkeypatch):
    from studio.training import TrainingPage
    app = application()
    page = TrainingPage()
    assert isinstance(page.device, QtWidgets.QComboBox)
    assert not page.device.isEditable()
    assert {page.device.itemData(i) for i in range(page.device.count())} >= {'', '0', 'cpu'}
    options = page.options
    page.device.setCurrentIndex(page.device.findData('0'))
    page.epochs.setValue(80)
    page.batch.setValue(8)
    options.patience.setValue(25)
    options.optimizer.setCurrentText('AdamW')
    options.lr0.setValue(.002)
    options.lrf.setValue(.05)
    options.momentum.setValue(.9)
    options.weight_decay.setValue(.001)
    options.warmup_epochs.setValue(5)
    options.cos_lr.setChecked(True)
    options.mask_ratio.setCurrentIndex(options.mask_ratio.findData(2))
    options.overlap_mask.setChecked(False)
    options.amp.setChecked(False)
    options.workers.setValue(2)
    options.seed.setValue(42)
    captured = []
    monkeypatch.setattr(page, 'launch', captured.append)
    page.start_training()
    config = captured.pop()
    expected = dict(device='0', epochs=80, batch=8, patience=25, optimizer='AdamW',
                    lr0=.002, lrf=.05, momentum=.9, weight_decay=.001, warmup_epochs=5.,
                    cos_lr=True, mask_ratio=2, overlap_mask=False, amp=False, workers=2, seed=42)
    assert expected.items() <= config.items()
    icons = page.findChildren(QtWidgets.QToolButton, 'helpIcon')
    assert len(icons) >= 20
    assert all(icon.toolTip() and icon.accessibleName() for icon in icons)
    page.close()
    app.processEvents()


def test_automatic_optimizer_and_batch_have_clear_effects(monkeypatch):
    from studio.training import TrainingPage
    app = application()
    page = TrainingPage()
    options = page.options
    assert not options.lr0.isEnabled() and not options.momentum.isEnabled()
    options.optimizer.setCurrentText('SGD')
    assert options.lr0.isEnabled() and options.momentum.isEnabled()
    options.momentum.setValue(0)
    assert options.momentum.value() > 0
    options.auto_batch.setChecked(True)
    assert not page.batch.isEnabled()
    captured = []
    monkeypatch.setattr(page, 'launch', captured.append)
    page.start_training()
    assert captured[-1]['batch'] == -1
    options.auto_batch.setChecked(False)
    assert page.batch.isEnabled()
    page.start_training()
    assert captured[-1]['batch'] == 4
    page.close()
    app.processEvents()


def test_advanced_parameters_are_collapsed_and_remain_editable_when_opened():
    from studio.training import TrainingPage
    app = application()
    page = TrainingPage()
    page.show()
    options = page.options
    assert not options.advanced_toggle.isChecked()
    assert options.tabs.isHidden()
    assert page.epochs.isVisible()
    options.advanced_toggle.click()
    app.processEvents()
    assert options.tabs.isVisible()
    options.optimizer.setCurrentText('AdamW')
    options.lr0.setValue(.003)
    options.advanced_toggle.click()
    assert options.tabs.isHidden()
    assert options.config()['lr0'] == .003
    page.close()


def training_config(tmp_path, **changes):
    data = tmp_path / 'data.yaml'
    data.touch()
    config = dict(mode='train', data=str(data), model='yolo11s-seg.pt', imgsz=864,
                  epochs=100, batch=4, device='cpu', project=str(tmp_path))
    config.update(changes)
    return config


@pytest.mark.parametrize('change', [dict(patience=-1), dict(lr0=0), dict(lr0=float('nan')),
    dict(lrf=1.5), dict(momentum=-.1), dict(weight_decay=-1), dict(warmup_epochs=-1),
    dict(optimizer='typo'), dict(optimizer='SGD', momentum=0), dict(workers=-1),
    dict(mask_ratio=0), dict(seed=-1), dict(amp='false')])
def test_invalid_advanced_settings_fail_before_loading_model(tmp_path, change):
    with pytest.raises(ValueError):
        validate_train_config(training_config(tmp_path, **change))


def test_training_worker_passes_the_selected_options_to_ultralytics(tmp_path, monkeypatch):
    best = tmp_path / 'best.pt'
    best.touch()
    received = {}
    class Model:
        task = 'segment'
        trainer = types.SimpleNamespace(best=best)
        def add_callback(self, *args):
            pass
        def train(self, **kwargs):
            received.update(kwargs)
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=lambda _: Model()))
    changed = dict(patience=20, optimizer='AdamW', lr0=.002, lrf=.05, momentum=.9,
        weight_decay=.001, warmup_epochs=5., cos_lr=True, mask_ratio=2,
        overlap_mask=False, amp=False, seed=42, workers=2, rect=True, device='0')
    result = run_job(training_config(tmp_path, **changed))
    assert result['status'] == 'success'
    assert changed.items() <= received.items()


def test_switching_prediction_device_invalidates_old_results():
    import numpy as np
    from studio.inference import Prediction
    from studio.window import MainWindow
    app = application()
    window = MainWindow()
    page = window.prediction
    assert isinstance(page.device, QtWidgets.QComboBox)
    generation = page.gate.start()
    page.on_result(generation, (Prediction(np.zeros((4, 8, 3), np.uint8), [], 1.), 'zmq'))
    assert page.last_result is not None
    page.device.setCurrentIndex(page.device.findData('0'))
    assert page.last_result is None
    assert not page.gate.accepts(generation)
    window.close()
    app.processEvents()
