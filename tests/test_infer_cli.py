import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def test_cli_output_never_overwrites_inputs_or_same_stem_results(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/infer_segment.py'
    spec = importlib.util.spec_from_file_location('infer_segment_cli', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ext in ['png', 'jpg']:
        cv2.imwrite(str(tmp_path / f'input.{ext}'), np.zeros((20, 30, 3), np.uint8))
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    class Predictor:
        def __init__(self, *args):
            pass
        def predict(self, frame):
            return SimpleNamespace(image=np.full_like(frame, 255), objects=[], elapsed_ms=1)
    monkeypatch.setattr(module, 'SegmentationPredictor', Predictor)
    monkeypatch.setattr(sys, 'argv', ['infer_segment.py', '--model', 'unused.pt', '--source', str(tmp_path),
                                    '--save', str(tmp_path), '--no-display'])
    module.main()
    for path, content in originals.items():
        assert path.read_bytes() == content
    assert len(list(tmp_path.glob('prediction_*/*.png'))) == 2
