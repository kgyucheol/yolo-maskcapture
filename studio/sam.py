"""On-demand SAM 2.1 Tiny, owned by one worker with a bounded prompt queue."""
import threading
import math
from pathlib import Path

import numpy as np

from .qt import QtCore, Signal


DEFAULT_SAM_MODEL = Path(__file__).resolve().parents[1] / 'models' / 'sam2.1_t.pt'


class SamSegmenter:
    def __init__(self, model=DEFAULT_SAM_MODEL, device=None):
        self.model_path = Path(model)
        self.device = device
        self.predictor = None
        self.image_key = None

    def load(self):
        import torch
        from ultralytics.models.sam import SAM2Predictor

        if self.predictor is None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            device = self.device if self.device is not None else ('0' if torch.cuda.is_available() else 'cpu')
            self.predictor = SAM2Predictor(overrides=dict(model=str(self.model_path), device=device,
                task='segment', mode='predict', imgsz=1024, conf=0.0, save=False, verbose=False))
        with torch.inference_mode():
            if self.predictor.model is None:
                self.predictor.setup_model()
        return str(self.predictor.device)

    def predict(self, image_key, frame, points, labels, mask_threshold=0.0):
        if not math.isfinite(mask_threshold) or not -5 <= mask_threshold <= 5:
            raise ValueError('SAM 경계 임계값은 -5~5 범위여야 합니다.')
        coords = np.asarray(points, dtype=np.float32)
        tags = np.asarray(labels)
        if (frame is None or frame.ndim != 3 or frame.shape[2] != 3 or coords.ndim != 2
                or coords.shape[1:] != (2,) or len(coords) == 0 or tags.shape != (len(coords),)
                or not np.isfinite(coords).all() or not np.isin(tags, [0, 1]).all() or not np.any(tags == 1)):
            raise ValueError('물체 내부 포함 점과 올바른 제외 점을 지정하세요.')
        height, width = frame.shape[:2]
        if np.any(coords < 0) or np.any(coords[:, 0] >= width) or np.any(coords[:, 1] >= height):
            raise ValueError('클릭 좌표가 이미지 범위를 벗어났습니다.')
        import torch

        self.load()
        with torch.inference_mode():
            self.predictor.model.mask_threshold = float(mask_threshold)
            if self.image_key != image_key:
                self.predictor.reset_image()
                self.predictor.set_image(frame)
                self.image_key = image_key
            # A leading object dimension is essential: all clicks refine ONE instance.
            result = self.predictor(points=[coords.tolist()], labels=[tags.tolist()])[0]
            if result.masks is None or len(result.masks.data) == 0:
                raise ValueError('마스크가 없습니다. 물체 안쪽에 포함 점을 추가하세요.')
            best = int(result.boxes.conf.argmax().item())
            mask = result.masks.data[best].detach().cpu().numpy()
        if mask.shape != (height, width) or not np.any(mask > 0):
            raise ValueError('원본 크기의 유효한 마스크를 만들지 못했습니다.')
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def close(self):
        if self.predictor is None:
            return
        self.predictor = None
        self.image_key = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SamWorker(QtCore.QThread):
    loaded = Signal(str)
    result = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = threading.Condition()
        self.pending = None
        self.latest_revision = None

    def submit(self, revision, image_key, frame, points, labels, mask_threshold=0.0):
        with self.condition:
            self.latest_revision = revision
            self.pending = (revision, image_key, frame, list(points), list(labels), float(mask_threshold))
            self.condition.notify()

    def clear_pending(self):
        with self.condition:
            self.pending = None
            self.latest_revision = None

    def stop(self):
        self.requestInterruption()
        with self.condition:
            self.pending = None
            self.latest_revision = None
            self.condition.notify()

    def run(self):
        backend = None
        try:
            backend = SamSegmenter()
            device = backend.load()
            if self.isInterruptionRequested():
                return
            self.loaded.emit(device)
            while not self.isInterruptionRequested():
                with self.condition:
                    if self.pending is None:
                        self.condition.wait(0.1)
                    item, self.pending = self.pending, None
                if item is None or self.isInterruptionRequested():
                    continue
                revision, image_key, frame, points, labels, threshold = item
                try:
                    mask = backend.predict(image_key, frame, points, labels, mask_threshold=threshold)
                    with self.condition:
                        current = revision == self.latest_revision
                    if current and not self.isInterruptionRequested():
                        self.result.emit(revision, mask)
                except Exception as exc:
                    with self.condition:
                        current = revision == self.latest_revision
                    if current and not self.isInterruptionRequested():
                        self.failed.emit(revision, str(exc))
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(-1, str(exc))
        finally:
            if backend is not None:
                backend.close()
