import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import subprocess
import sys
import pytest

import numpy as np

from studio.qt import QtWidgets
from studio.project import Project
from studio.window import MainWindow


def test_full_window_project_and_navigation(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = Project.create(tmp_path / 'project')
    p.add_class('컵')
    p.capture(np.zeros((480, 848, 3), np.uint8))
    window = MainWindow()
    window.set_project(p)
    window.show()
    for index in range(5):
        window.navigation.setCurrentRow(index)
        app.processEvents()
        assert window.stack.currentIndex() == index
    assert window.training.model.text() == 'yolo11s-seg.pt'
    assert window.training.imgsz.value() == 864
    window.close()
    app.processEvents()


def test_failed_job_writes_failure_manifest(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(mode='train', model='yolo11s-seg.pt', imgsz=864,
        epochs=1, batch=1, data=str(tmp_path / 'missing.yaml'))))
    result = subprocess.run([sys.executable, '-m', 'studio.jobs', '--config', str(config)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    manifest = json.loads((tmp_path / 'result.json').read_text())
    assert manifest['status'] == 'failed'


def test_prediction_settings_remain_visible_at_default_window_size():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.navigation.setCurrentRow(4)
    app.processEvents()
    assert window.prediction.conf.height() >= 25
    assert window.prediction.iou.height() >= 25
    window.close()


def test_connection_save_failure_is_recoverable(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = Project.create(tmp_path / 'project')
    window = MainWindow()
    window.set_project(p)
    previous = p.data['connection'].copy()
    window.capture.host.setText('192.168.1.1')
    def fail():
        raise OSError('disk full')
    monkeypatch.setattr(p, 'save', fail)
    window.capture.toggle_connection()
    assert p.data['connection'] == previous
    assert 'disk full' in window.capture.notice.text()
    assert window.capture.worker is None
    monkeypatch.undo()
    window.close()


def test_editing_model_invalidates_old_results_and_clear_saveable_result(tmp_path):
    from studio.inference import Prediction
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    p = Project.create(tmp_path / 'project')
    window.set_project(p)
    page = window.prediction
    page.model.setText('model_a.pt')
    generation = page.gate.start()
    page.on_result(generation, (Prediction(np.zeros((4, 8, 3), np.uint8), [], 1.0), 'zmq'))
    assert page.last_result is not None
    page.model.setText('model_b.pt')
    assert not page.gate.accepts(generation)
    assert page.last_result is None
    page.save_result()
    assert not list((p.root / 'predictions').iterdir())
    window.close()


def test_dataset_worker_connects_to_training_and_process_failure_is_reported(tmp_path):
    from studio.document import AnnotationDocument
    from studio.qt import QtTest
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = Project.create(tmp_path / 'project')
    p.add_class('컵')
    for _ in range(2):
        image = p.capture(np.zeros((48, 84, 3), np.uint8))
        doc = AnnotationDocument((48, 84))
        doc.add_mask(np.ones((48, 84), np.uint8))
        doc.assign_class(0)
        p.save_annotation(image, doc.objects, True)
    window = MainWindow()
    window.set_project(p)
    window.dataset.create_dataset()
    for _ in range(300):
        QtTest.QTest.qWait(10)
        if not window.dataset.busy():
            app.processEvents()
            break
    assert window.training.data.text().endswith('data.yaml')
    assert window.dataset.create_button.isEnabled()
    window.training.model.setText(str(tmp_path / 'missing.invalid'))
    window.training.start_training()
    for _ in range(500):
        QtTest.QTest.qWait(10)
        if not window.training.busy():
            app.processEvents()
            break
    assert not window.training.busy()
    assert '실패' in window.training.notice.text()
    assert window.training.panel.isEnabled()
    window.close()
