"""YOLO11 segmentation training, for Studio projects or existing YOLO polygon labels.

python scripts/train.py --project projects/my-project --epochs 100 --device 0
python scripts/train.py --raw data/raw --model yolo11s-seg.pt --imgsz 864

Raw layout: images/, labels/<stem>.txt (class x1 y1 x2 y2 ...), classes.txt.
Output datasets and runs always receive a new directory; old outputs are preserved.
"""
import argparse
import math
import random
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.project import IMAGE_EXTENSIONS, Project, atomic_json
from studio.dataset import build_dataset as build_project_dataset
from studio.jobs import run_job


def build_dataset(raw: Path, out: Path, val_ratio: float, seed: int) -> Path:
    raw, out = Path(raw), Path(out)
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('검증 비율은 0과 1 사이여야 합니다.')
    names = [line.strip() for line in (raw / 'classes.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not names:
        raise ValueError('classes.txt에 클래스 이름이 필요합니다.')
    pairs = []
    stems = set()
    for image in sorted((raw / 'images').iterdir()):
        label = raw / 'labels' / (image.stem + '.txt')
        if image.suffix.lower() not in IMAGE_EXTENSIONS or not label.exists():
            continue
        if image.stem in stems:
            raise ValueError('같은 stem의 이미지가 여러 개여서 라벨이 모호합니다: ' + image.stem)
        stems.add(image.stem)
        for row in label.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            fields = row.split()
            try:
                cid = int(fields[0])
                points = [float(v) for v in fields[1:]]
                valid = len(points) >= 6 and len(points) % 2 == 0 and 0 <= cid < len(names)
                valid = valid and all(math.isfinite(v) and 0 <= v <= 1 for v in points)
            except (ValueError, IndexError):
                valid = False
            if not valid:
                raise ValueError(f'{label.name}: YOLO 세그먼테이션 폴리곤 라벨이 필요합니다.')
        pairs.append((image, label))
    if len(pairs) < 2:
        raise ValueError('라벨된 이미지가 최소 2장 필요합니다.')
    random.Random(seed).shuffle(pairs)
    nval = min(len(pairs) - 1, max(1, int(len(pairs) * val_ratio)))
    out.mkdir(parents=True, exist_ok=True)
    target = out / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    stage = Path(tempfile.mkdtemp(prefix='.building_', dir=out))
    try:
        for split, entries in {'train': pairs[nval:], 'val': pairs[:nval]}.items():
            (stage / 'images' / split).mkdir(parents=True)
            (stage / 'labels' / split).mkdir(parents=True)
            for image, label in entries:
                shutil.copy2(image, stage / 'images' / split / image.name)
                shutil.copy2(label, stage / 'labels' / split / label.name)
        (stage / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(target.resolve()),
            train='images/train', val='images/val', names=dict(enumerate(names))), allow_unicode=True), encoding='utf-8')
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage)
        raise
    return target / 'data.yaml'


def main():
    parser = argparse.ArgumentParser(description='YOLO segmentation training')
    parser.add_argument('--project', help='Studio 프로젝트 폴더')
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--dataset', default='data/dataset')
    parser.add_argument('--model', default='yolo11s-seg.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=864)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='segment')
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--opset', type=int, default=17, help='ONNX export opset')
    args = parser.parse_args()
    project = Project.open(args.project) if args.project else None
    data = build_project_dataset(project, args.val_ratio, args.seed) if project else build_dataset(
        Path(args.raw), Path(args.dataset), args.val_ratio, args.seed)
    runs = project.root / 'runs' if project else Path('runs')
    run_dir = runs / (args.name + '_' + uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    config = dict(mode='train', model=args.model, data=str(data.resolve()), epochs=args.epochs,
                  imgsz=args.imgsz, batch=args.batch, device=args.device, rect=args.rect,
                  project=str(run_dir.resolve()), name='segment')
    atomic_json(run_dir / 'config.json', config)
    result = run_job(config)
    atomic_json(run_dir / 'result.json', result)
    print(result)
    if not args.no_export:
        export_dir = run_dir / 'onnx'
        exported = run_job(dict(mode='export', model=result['best'], imgsz=args.imgsz,
                               project=str(export_dir.resolve()), opset=args.opset))
        atomic_json(export_dir / 'result.json', exported)
        print(exported)


if __name__ == '__main__':
    main()
