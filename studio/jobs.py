"""Worker process CLI. Heavy ML imports happen only inside a requested job."""
import argparse
import json
import math
import signal
import sys
import traceback
import time
from pathlib import Path
from uuid import uuid4

from .project import atomic_json


OPTIMIZERS = ('auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp')
TRAIN_DEFAULTS = dict(patience=100, optimizer='auto', lr0=.01, lrf=.01, momentum=.937,
                      weight_decay=.0005, warmup_epochs=3., cos_lr=False, amp=True,
                      seed=0, workers=0, mask_ratio=4, overlap_mask=True)


def validate_train_config(config):
    if config.get('mode') not in ('train', 'export'):
        raise ValueError('알 수 없는 작업입니다.')
    size = config.get('imgsz', 864)
    if not isinstance(size, int) or size < 32 or size % 32:
        raise ValueError('학습 크기는 32의 배수여야 합니다. 848×480 원본에는 imgsz=864로 시작하세요.')
    if config['mode'] == 'train':
        if config.get('epochs', 0) < 1 or config.get('batch', 0) == 0 or config.get('batch', -2) < -1:
            raise ValueError('epoch는 1 이상, batch는 1 이상 또는 자동(-1)이어야 합니다.')
        if not Path(config.get('data', '')).is_file():
            raise ValueError('생성된 데이터셋의 data.yaml을 선택하세요.')
        settings = {**TRAIN_DEFAULTS, **config}
        for name, minimum, maximum in (('patience', 0, 100000), ('workers', 0, 32),
                                      ('seed', 0, 2147483647), ('mask_ratio', 1, 16)):
            value = settings[name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{name}: {minimum}~{maximum} 범위의 정수를 입력하세요.')
        for name, minimum, maximum in (('lr0', .000001, 1), ('lrf', 0, 1), ('momentum', 0, .999),
                                      ('weight_decay', 0, 1), ('warmup_epochs', 0, 1000)):
            value = settings[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f'{name}: {minimum}~{maximum} 범위의 숫자를 입력하세요.')
        if settings['optimizer'] not in OPTIMIZERS:
            raise ValueError('목록에서 optimizer를 선택하세요.')
        if settings['optimizer'] == 'SGD' and settings['momentum'] <= 0:
            raise ValueError('SGD의 모멘텀은 0보다 커야 합니다.')
        for name in ('cos_lr', 'amp', 'overlap_mask'):
            if type(settings[name]) is not bool:
                raise ValueError(f'{name}: 켜짐/꺼짐 값이 필요합니다.')
    elif not Path(config.get('model', '')).is_file():
        raise ValueError('내보낼 PT 파일을 선택하세요.')


def run_job(config):
    validate_train_config(config)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError('학습 패키지가 없습니다. python -m pip install -r requirements.txt') from exc
    model = YOLO(config['model'])
    if model.task != 'segment':
        raise ValueError('세그먼테이션 모델을 사용하세요. 기본 모델: yolo11s-seg.pt')
    if config['mode'] == 'train':
        epoch_started = [None]
        def on_epoch_start(trainer):
            epoch_started[0] = time.monotonic()
        def on_epoch(trainer):
            print('STUDIO_PROGRESS ' + json.dumps(dict(epoch=int(trainer.epoch) + 1,
                epochs=int(trainer.epochs), epoch_seconds=(time.monotonic() - epoch_started[0]
                    if epoch_started[0] is not None else None))), flush=True)
        model.add_callback('on_train_epoch_start', on_epoch_start)
        model.add_callback('on_fit_epoch_end', on_epoch)
        model.train(data=config['data'], epochs=int(config['epochs']), imgsz=config['imgsz'],
                    batch=int(config['batch']), device=config.get('device') or None,
                    rect=bool(config.get('rect', False)), project=config['project'],
                    name=config.get('name', 'train_' + uuid4().hex[:8]), exist_ok=False,
                    **{key: config.get(key, value) for key, value in TRAIN_DEFAULTS.items()})
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError('학습은 종료되었지만 best.pt를 찾을 수 없습니다.')
        result = dict(status='success', mode='train', best=str(best.resolve()))
    else:
        import shutil
        # Export beside a job-local copy, never overwrite a user model's adjacent ONNX file.
        export_dir = Path(config['project'])
        export_dir.mkdir(parents=True, exist_ok=True)
        local_model = export_dir / ('export_' + uuid4().hex[:8] + '.pt')
        shutil.copy2(config['model'], local_model)
        model = YOLO(str(local_model))
        output = model.export(format='onnx', imgsz=config['imgsz'], dynamic=False,
                              simplify=False, opset=int(config.get('opset', 17)), nms=False, device='cpu')
        if not Path(output).is_file():
            raise RuntimeError('ONNX 출력 파일을 찾을 수 없습니다.')
        result = dict(status='success', mode='export', onnx=str(Path(output).resolve()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    config = json.loads(path.read_text(encoding='utf-8'))
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    result_path = path.parent / 'result.json'
    try:
        result = run_job(config)
        atomic_json(result_path, result)
        print('STUDIO_RESULT ' + json.dumps(result, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        atomic_json(result_path, dict(status='cancelled'))
        sys.exit(130)
    except Exception as exc:
        traceback.print_exc()
        atomic_json(result_path, dict(status='failed', error=str(exc)))
        sys.exit(1)


if __name__ == '__main__':
    main()
