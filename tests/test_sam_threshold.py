import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from studio.qt import QtWidgets, QtTest


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_backend_boundary_threshold_changes_mask_not_object_quality(monkeypatch):
    import torch
    import ultralytics.models.sam
    from studio.sam import SamSegmenter

    class Predictor:
        def __init__(self, overrides):
            self.model = None
            self.device = 'cpu'

        def setup_model(self):
            self.model = SimpleNamespace(mask_threshold=0.0)

        def reset_image(self):
            pass

        def set_image(self, image):
            pass

        def __call__(self, **kwargs):
            logits = torch.tensor([[[-1., 0.5, 2.0], [-1., 0.5, 2.0]]])
            return [SimpleNamespace(masks=SimpleNamespace(data=logits > self.model.mask_threshold),
                                    boxes=SimpleNamespace(conf=torch.tensor([0.9])))]

    monkeypatch.setattr(ultralytics.models.sam, 'SAM2Predictor', Predictor)
    backend = SamSegmenter(device='cpu')
    frame = np.zeros((2, 3, 3), np.uint8)
    wide = backend.predict(1, frame, [(1, 1)], [1], mask_threshold=-2)
    narrow = backend.predict(1, frame, [(1, 1)], [1], mask_threshold=1)
    assert wide.tolist() == [[255, 255, 255], [255, 255, 255]]
    assert narrow.tolist() == [[0, 0, 255], [0, 0, 255]]


def test_slider_changes_invalidate_old_preview_and_preserve_confirmed_masks(app, monkeypatch):
    import studio.sam as sam
    from studio.labeling import LabelingPage
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def load(self):
            return 'test GPU'

        def close(self):
            pass

        def predict(self, key, frame, points, labels, mask_threshold=0):
            entered.set()
            assert release.wait(3)
            mask = np.zeros(frame.shape[:2], np.uint8)
            mask[10:40, 10:50 if mask_threshold <= 1 else 30] = 255
            return mask

    monkeypatch.setattr(sam, 'SamSegmenter', Backend)
    page = LabelingPage()
    c = page.canvas
    c.set_image(np.zeros((60, 120, 3), np.uint8))
    original = np.zeros((60, 120), np.uint8)
    original[5:15, 5:15] = 255
    c.doc.add_mask(original)
    try:
        page.sam_threshold.setValue(10)
        assert page.sam_worker is None  # adjusting a slider alone does not load a model
        c.set_tool('sam')
        c.sam_points, c.sam_labels = [(20, 30)], [1]
        c._request_sam()
        for _ in range(300):
            if entered.is_set():
                break
            QtTest.QTest.qWait(10)
        assert entered.is_set()  # first inference is running while the slider changes
        old = c.sam_revision
        page.sam_threshold.setValue(15)
        assert c.sam_revision > old and c.sam_preview is None
        c.accept_sam(old, original)
        c.finish()
        assert len(c.doc.objects) == 1 and np.array_equal(c.doc.objects[0].mask, original)
        release.set()
        for _ in range(300):
            QtTest.QTest.qWait(10)
            if c.sam_preview is not None:
                break
        assert c.sam_preview is not None
        assert c.sam_preview[20, 20] == 255 and c.sam_preview[20, 40] == 0
        c.finish()
        assert len(c.doc.objects) == 2
        page.reset_sam_threshold()
        assert page.sam_threshold.value() == 0
        assert np.array_equal(c.doc.objects[0].mask, original)
    finally:
        release.set()
        c.cancel()
        page.stop_sam()
        if page.sam_worker:
            assert page.sam_worker.wait(3000)
        page.close()
        app.processEvents()
