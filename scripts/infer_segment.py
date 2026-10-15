"""PT/ONNX segmentation inference for ZMQ, an image, or a folder."""
import argparse
import sys
from pathlib import Path
from uuid import uuid4

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.inference import SegmentationPredictor
from studio.project import IMAGE_EXTENSIONS, read_image, atomic_bytes, image_bytes


def frames(args):
    if args.source == 'zmq':
        from zmq_source import ZmqImageSource
        source = ZmqImageSource(args.host, args.port, args.mode, timeout_ms=200)
        try:
            count = 0
            while True:
                image = source.read()
                if image is not None:
                    count += 1
                    yield f'frame_{count:08d}.png', image
        finally:
            source.close()
    else:
        path = Path(args.source)
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if path.is_dir() else [path]
        for item in files:
            image = read_image(item)
            if image is None:
                raise ValueError('이미지를 읽을 수 없습니다: ' + str(item))
            yield item.stem + '.png', image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', default='zmq')
    parser.add_argument('--host', default='192.168.123.164')
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--mode', choices=['sub', 'pull', 'req'], default='sub')
    parser.add_argument('--imgsz', type=int, default=864)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    predictor = SegmentationPredictor(args.model, args.conf, args.iou, args.device, args.imgsz)
    output = Path(args.save) / ('prediction_' + uuid4().hex[:12]) if args.save else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    try:
        for name, frame in frames(args):
            result = predictor.predict(frame)
            print(f'{name}: {len(result.objects)} objects, {result.elapsed_ms:.1f} ms')
            if output:
                target = output / (Path(name).stem + '_' + uuid4().hex[:8] + '.png')
                atomic_bytes(target, image_bytes(result.image))
            if not args.no_display:
                cv2.imshow('Segmentation', result.image)
                if cv2.waitKey(1 if args.source == 'zmq' else 0) & 0xff in (27, ord('q')):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
