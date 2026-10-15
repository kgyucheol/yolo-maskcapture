import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from studio.qt import QtWidgets, QtTest


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(predicate, timeout=3000):
    for _ in range(timeout // 10):
        if predicate():
            return
        QtTest.QTest.qWait(10)
    assert predicate(), 'background operation timed out'


def test_backend_groups_positive_negative_points_and_caches_only_current_image(monkeypatch):
    import torch
    import ultralytics.models.sam
    from studio.sam import SamSegmenter

    class Predictor:
        def __init__(self, overrides):
            self.model = None
            self.device = 'cpu'
            self.encoded = 0

        def setup_model(self):
            self.model = SimpleNamespace(mask_threshold=0.0)

        def reset_image(self):
            self.image = None

        def set_image(self, image):
            self.image = image
            self.encoded += 1

        def __call__(self, *, points, labels):
            # Multiple clicks must describe ONE object, never separate instances.
            assert points == [[[20, 30], [90, 30]]]
            assert labels == [[1, 0]]
            result = torch.zeros((1, 60, 120), dtype=torch.bool)
            result[0, 10:40, 10:50] = True
            return [SimpleNamespace(masks=SimpleNamespace(data=result),
                                    boxes=SimpleNamespace(conf=torch.tensor([0.9])))]

    monkeypatch.setattr(ultralytics.models.sam, 'SAM2Predictor', Predictor)
    backend = SamSegmenter(device='cpu')
    image = np.zeros((60, 120, 3), np.uint8)
    for key in (1, 1, 2):
        result = backend.predict(key, image, [(20, 30), (90, 30)], [1, 0])
        assert result.shape == (60, 120) and result.dtype == np.uint8
        assert result[30, 20] == 255 and result[30, 90] == 0
    assert backend.predictor.encoded == 2


@pytest.mark.parametrize('points, labels', [([], []), ([(1, 1)], [0]), ([(999, 1)], [1]),
                                            ([(1, 1)], [1, 0]), ([(1, 1)], [2])])
def test_invalid_prompts_fail_before_running_inference(points, labels):
    from studio.sam import SamSegmenter
    backend = SamSegmenter(device='cpu')
    with pytest.raises(ValueError):
        backend.predict(1, np.zeros((60, 120, 3), np.uint8), points, labels)


def test_worker_keeps_latest_request_and_suppresses_cancelled_results(app, monkeypatch):
    import studio.sam as sam
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def close(self):
            pass

        def load(self):
            return 'test GPU'

        def predict(self, key, frame, points, labels, mask_threshold=0.0):
            if points[0][0] == 1:
                entered.set()
                assert release.wait(3)
            return np.full((10, 10), points[0][0], np.uint8)

    monkeypatch.setattr(sam, 'SamSegmenter', Backend)
    worker = sam.SamWorker()
    results = []
    worker.result.connect(lambda revision, mask: results.append((revision, mask)))
    frame = np.zeros((10, 10, 3), np.uint8)
    worker.submit(1, 1, frame, [(1, 1)], [1])
    worker.start()
    try:
        wait_until(entered.is_set)
        worker.submit(2, 1, frame, [(2, 1)], [1])
        worker.submit(3, 1, frame, [(3, 1)], [1])
        release.set()
        wait_until(lambda: len(results) == 1)
        assert results[0][0] == 3 and np.all(results[0][1] == 3)
        entered.clear()
        release.clear()
        worker.submit(4, 2, frame, [(1, 1)], [1])
        wait_until(entered.is_set)
        worker.clear_pending()
        release.set()
        QtTest.QTest.qWait(80)
        assert len(results) == 1
    finally:
        release.set()
        worker.stop()
        assert worker.wait(3000)


def test_worker_recovers_after_prediction_error(app, monkeypatch):
    import studio.sam as sam

    class Backend:
        def close(self):
            pass

        def load(self):
            return 'test GPU'

        def predict(self, key, frame, points, labels, mask_threshold=0.0):
            if key == 1:
                raise RuntimeError('GPU allocation failed')
            return np.ones((10, 10), np.uint8)

    monkeypatch.setattr(sam, 'SamSegmenter', Backend)
    worker = sam.SamWorker()
    failures, results = [], []
    worker.failed.connect(lambda revision, error: failures.append((revision, error)))
    worker.result.connect(lambda revision, mask: results.append((revision, mask)))
    worker.start()
    try:
        worker.submit(1, 1, np.zeros((10, 10, 3), np.uint8), [(1, 1)], [1])
        wait_until(lambda: len(failures) == 1)
        assert failures[0] == (1, 'GPU allocation failed')
        worker.submit(2, 2, np.zeros((10, 10, 3), np.uint8), [(1, 1)], [1])
        wait_until(lambda: len(results) == 1)
        assert results[0][0] == 2
    finally:
        worker.stop()
        assert worker.wait(3000)


def test_labeling_sam_result_saves_and_navigation_rejects_unfinished_draft(app, monkeypatch, tmp_path):
    import studio.sam as sam
    from studio.labeling import LabelingPage
    from studio.project import Project

    class Backend:
        def close(self):
            pass

        def load(self):
            return 'test GPU'

        def predict(self, key, frame, points, labels, mask_threshold=0.0):
            mask = np.zeros(frame.shape[:2], np.uint8)
            mask[10:40, 10:50] = 255
            return mask

    monkeypatch.setattr(sam, 'SamSegmenter', Backend)
    project = Project.create(tmp_path / 'p')
    project.add_class('컵')
    first = project.capture(np.zeros((60, 120, 3), np.uint8))
    project.capture(np.zeros((60, 120, 3), np.uint8))
    page = LabelingPage()
    page.set_project(project)
    canvas = page.canvas
    try:
        canvas.set_tool('sam')
        canvas.sam_points, canvas.sam_labels = [(20, 30)], [1]
        canvas._request_sam()
        page.navigate(1)
        assert page.index == 0
        wait_until(lambda: canvas.sam_preview is not None)
        canvas.finish()
        canvas.assign_tag(0)
        page.navigate(1)
        assert page.index == 1
        objects, reviewed = project.load_annotation(first)
        assert reviewed and len(objects) == 1 and objects[0].class_id == 0
        assert objects[0].mask[30, 20] == 255
    finally:
        page.stop_sam()
        if page.sam_worker:
            assert page.sam_worker.wait(3000)
        page.close()
        app.processEvents()


def test_model_load_failure_can_be_retried_without_discarding_draft(app, monkeypatch):
    import studio.sam as sam
    from studio.labeling import LabelingPage
    attempts = []

    class Backend:
        def close(self):
            pass

        def load(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError('model unavailable')
            return 'test GPU'

        def predict(self, key, frame, points, labels, mask_threshold=0.0):
            return np.ones(frame.shape[:2], np.uint8)

    monkeypatch.setattr(sam, 'SamSegmenter', Backend)
    page = LabelingPage()
    page.canvas.set_image(np.zeros((60, 120, 3), np.uint8))
    c = page.canvas
    try:
        c.set_tool('sam')
        c.sam_points, c.sam_labels = [(20, 30)], [1]
        c._request_sam()
        wait_until(lambda: page.sam_worker is None)
        assert 'model unavailable' in page.sam_status.text()
        assert c.pending and c.doc.objects == []
        page.prepare_sam()
        wait_until(lambda: 'test GPU' in page.sam_status.text())
        wait_until(lambda: c.sam_preview is not None)
        c.finish()
        assert len(c.doc.objects) == 1
    finally:
        c.cancel()
        page.stop_sam()
        if page.sam_worker:
            assert page.sam_worker.wait(3000)
        page.close()
        app.processEvents()
