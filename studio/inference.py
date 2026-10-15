"""Shared segmentation inference for Ultralytics PT and its ONNX exports."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ResultGate:
    def __init__(self):
        self.generation = 0
        self.active = False

    def start(self):
        self.generation += 1
        self.active = True
        return self.generation

    def stop(self):
        self.active = False
        self.generation += 1

    def accepts(self, generation):
        return self.active and generation == self.generation


@dataclass
class Prediction:
    image: np.ndarray
    objects: list
    elapsed_ms: float


class SegmentationPredictor:
    def __init__(self, path, conf=0.25, iou=0.45, device='cpu', imgsz=864):
        path = Path(path)
        if not path.is_file() or path.suffix.lower() not in ('.pt', '.onnx'):
            raise ValueError('존재하는 세그먼테이션 .pt 또는 .onnx 파일을 선택하세요.')
        if not 0 < conf <= 1 or not 0 < iou <= 1:
            raise ValueError('신뢰도와 IoU는 0보다 크고 1 이하여야 합니다.')
        if path.suffix.lower() == '.onnx':
            try:
                import onnx
                import onnxruntime  # Prevent backend auto-install attempts during inference.
            except ImportError as exc:
                raise ImportError('ONNX 추론에는 onnx와 onnxruntime이 필요합니다.') from exc
            graph = onnx.load(str(path), load_external_data=False)
            metadata = {p.key: p.value for p in graph.metadata_props}
            if metadata.get('task') != 'segment':
                raise ValueError('Ultralytics 세그먼테이션 ONNX 모델(task=segment)을 선택하세요.')
            del graph
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('추론에는 ultralytics가 필요합니다: python -m pip install -r requirements.txt') from exc
        self.model = YOLO(str(path), task='segment') if path.suffix.lower() == '.onnx' else YOLO(str(path))
        if self.model.task != 'segment':
            raise ValueError('객체 검출 모델입니다. 세그먼테이션 모델을 선택하세요.')
        self.conf, self.iou, self.device, self.imgsz = conf, iou, device, imgsz

    def predict(self, frame):
        start = time.perf_counter()
        result = self.model.predict(source=frame, conf=self.conf, iou=self.iou, device=self.device,
                                    imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        objects = []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None:
                raise ValueError('모델 결과에 마스크가 없습니다. 세그먼테이션 모델을 확인하세요.')
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for cid, score, polygon in zip(classes, scores, result.masks.xy):
                objects.append(dict(class_id=int(cid), name=result.names[int(cid)], confidence=float(score),
                                    polygon=polygon.tolist()))
        visual = result.plot(boxes=True, masks=True, labels=True)
        return Prediction(visual, objects, (time.perf_counter() - start) * 1000)
