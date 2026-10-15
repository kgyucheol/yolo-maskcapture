"""3. ONNX 추론 확인: ZMQ 스트림 / 이미지 / 폴더 입력, onnxruntime + 직접 구현한 전후처리.

출력 형태 가정: Ultralytics export (1, 4+nc, N), box=xywh(입력 해상도 기준), NMS 미포함.
예) python scripts/infer_onnx.py --model runs/exp/weights/best.onnx --source zmq --host 192.168.0.10
    python scripts/infer_onnx.py --model best.onnx --source test.jpg
    python scripts/infer_onnx.py --model best.onnx --source data/raw/images --save out/
"""
import argparse
import ast
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size[0] - nh) // 2, (size[1] - nw) // 2
    out = np.full((size[0], size[1], 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


class OnnxDetector:
    def __init__(self, path, conf=0.25, iou=0.45, names=None):
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = (int(inp.shape[2]), int(inp.shape[3]))  # (H, W)
        self.conf, self.iou = conf, iou
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = names or (ast.literal_eval(meta["names"]) if "names" in meta else {})
        print(f"[onnx] providers={self.sess.get_providers()} input={inp.shape} names={self.names}")

    def __call__(self, bgr):
        img, r, left, top = letterbox(bgr, self.size)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        pred = out[0].T  # (N, 4+nc)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(pred)), cls]
        keep = scores > self.conf
        pred, cls, scores = pred[keep], cls[keep], scores[keep]
        if len(pred) == 0:
            return []
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # letterbox 좌표 -> 원본 좌표
        x1 = (cx - w / 2 - left) / r
        y1 = (cy - h / 2 - top) / r
        bw, bh = w / r, h / r
        boxes = np.stack([x1, y1, bw, bh], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), cls.tolist(), self.conf, self.iou)
        H, W = bgr.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            bx, by, bw_, bh_ = boxes[i]
            dets.append((max(0, bx), max(0, by), min(W, bx + bw_), min(H, by + bh_),
                         float(scores[i]), int(cls[i])))
        return dets

    def draw(self, img, dets):
        for x1, y1, x2, y2, s, c in dets:
            col = tuple(int(v) for v in np.random.RandomState(c).randint(60, 255, 3))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(img, f"{self.names.get(c, c)} {s:.2f}", (int(x1), max(14, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        return img


def frames(args):
    if args.source == "zmq":
        from zmq_source import ZmqImageSource
        src = ZmqImageSource(args.host, args.port, args.mode, args.bind)
        print(f"[infer] {src.addr} ({args.mode})")
        try:
            while True:
                f = src.read()
                if f is not None:
                    yield "zmq", f
        finally:
            src.close()
    else:
        p = Path(args.source)
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) \
            if p.is_dir() else [p]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield f.name, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="zmq", help="zmq | 이미지 파일 | 폴더")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=55555)
    ap.add_argument("--mode", default="sub", choices=["sub", "pull", "req"])
    ap.add_argument("--bind", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--names", default=None, help="ONNX 메타데이터에 없을 때 쉼표 구분 클래스명")
    ap.add_argument("--save", default=None, help="결과 이미지 저장 폴더")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    names = dict(enumerate(args.names.split(","))) if args.names else None
    det = OnnxDetector(args.model, args.conf, args.iou, names)
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    t_prev = time.time()
    try:
        for name, img in frames(args):
            t0 = time.time()
            dets = det(img)
            ms = (time.time() - t0) * 1000
            vis = det.draw(img.copy(), dets)
            fps = 1.0 / max(1e-6, time.time() - t_prev)
            t_prev = time.time()
            cv2.putText(vis, f"{ms:.1f}ms  {fps:.1f}fps  det={len(dets)}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"{name}: {len(dets)} det, {ms:.1f}ms")
            if args.save:
                cv2.imwrite(str(Path(args.save) / (Path(name).stem + "_det.jpg")), vis)
            if not args.no_display:
                cv2.imshow("onnx", vis)
                # 파일/폴더 입력은 키 입력 대기(q 종료), 스트림은 연속
                if cv2.waitKey(1 if args.source == "zmq" else 0) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
