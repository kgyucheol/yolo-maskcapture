"""Durable project storage; immutable mask revisions keep annotation saves atomic."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .document import MaskObject

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DEFAULT_CONNECTION = dict(host='192.168.123.164', port=55555, mode='sub', bind=False)


def atomic_bytes(path, data):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except (OSError, cv2.error):
        return None


def image_bytes(image, ext='.png'):
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError('이미지를 인코딩하지 못했습니다.')
    return encoded.tobytes()


class Project:
    def __init__(self, root, data):
        self.root, self.data = Path(root).resolve(), data

    @classmethod
    def create(cls, root):
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError('새 프로젝트에는 비어 있는 폴더를 선택하세요.')
        root.mkdir(parents=True, exist_ok=True)
        for name in ('images', 'annotations', 'masks', 'datasets', 'runs', 'predictions'):
            (root / name).mkdir()
        project = cls(root, dict(version=1, name=root.name, classes=[], images=[],
                                 connection=DEFAULT_CONNECTION.copy(), last_image=None))
        project.save()
        return project

    @classmethod
    def open(cls, root):
        root = Path(root)
        if root.is_file():
            root = root.parent
        try:
            data = json.loads((root / 'project.json').read_text(encoding='utf-8'))
            if data['version'] != 1 or not isinstance(data['images'], list):
                raise ValueError('지원하지 않는 프로젝트 버전입니다.')
            if [c['id'] for c in data['classes']] != list(range(len(data['classes']))):
                raise ValueError('클래스 ID가 올바르지 않습니다.')
            project = cls(root, data)
            for entry in project.images:
                project.image_path(entry['file'])
            return project
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'프로젝트 형식이 올바르지 않습니다: {exc}') from exc

    @property
    def images(self):
        return self.data['images']

    @property
    def classes(self):
        return self.data['classes']

    def save(self):
        atomic_json(self.root / 'project.json', self.data)

    def _safe(self, relative):
        relative = Path(relative)
        target = (self.root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(self.root):
            raise ValueError('프로젝트 밖의 경로에는 접근할 수 없습니다.')
        return target

    def image_path(self, image):
        if not image or Path(image).name != image:
            raise ValueError('이미지 파일명이 올바르지 않습니다.')
        return self._safe('images/' + image)

    def annotation_path(self, image):
        self.image_path(image)
        return self._safe('annotations/' + Path(image).stem + '.json')

    def add_class(self, name):
        name = name.strip()
        if not name or any(c['name'] == name for c in self.classes):
            raise ValueError('비어 있지 않은 고유한 클래스 이름을 입력하세요.')
        self.classes.append(dict(id=len(self.classes), name=name))
        self.save()
        return len(self.classes) - 1

    def rename_class(self, class_id, name):
        name = name.strip()
        if not name or any(c['name'] == name and c['id'] != class_id for c in self.classes):
            raise ValueError('고유한 클래스 이름을 입력하세요.')
        self.classes[class_id]['name'] = name
        self.save()

    def import_image(self, source):
        source = Path(source)
        frame = read_image(source)
        if frame is None:
            raise ValueError(f'이미지를 읽을 수 없습니다: {source.name}')
        return self.capture(frame, source=str(source), prefix=source.stem)

    def capture(self, frame, source='zmq', prefix='capture', received_at=None):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        # User filenames cannot become paths. A UUID prevents both file and label-stem collisions.
        prefix = ''.join(c if c.isalnum() or c in '-_' else '_' for c in prefix)[:48]
        name = f'{prefix}_{stamp}_{uuid4().hex[:8]}.png'
        path = self.image_path(name)
        atomic_bytes(path, image_bytes(frame))
        entry = dict(file=name, source=source, captured_at=stamp, received_at=received_at,
                     width=int(frame.shape[1]), height=int(frame.shape[0]))
        self.images.append(entry)
        try:
            self.save()
        except Exception:
            self.images.pop()
            path.unlink(missing_ok=True)
            raise
        return name

    def save_annotation(self, image, objects, reviewed=False):
        frame = read_image(self.image_path(image))
        if frame is None:
            raise ValueError('원본 이미지를 읽을 수 없습니다.')
        shape = frame.shape[:2]
        entries = []
        revision = uuid4().hex
        for obj in objects:
            if obj.mask.shape != shape or not np.any(obj.mask):
                raise ValueError('비어 있거나 크기가 맞지 않는 마스크입니다.')
            if obj.class_id is not None and not 0 <= obj.class_id < len(self.classes):
                raise ValueError('등록되지 않은 클래스입니다.')
            if not obj.id.isalnum():
                raise ValueError('객체 ID가 올바르지 않습니다.')
            rel = f'masks/{revision}_{obj.id}.png'
            atomic_bytes(self._safe(rel), image_bytes(np.where(obj.mask > 0, 255, 0).astype(np.uint8)))
            entries.append(dict(id=obj.id, class_id=obj.class_id, mask=rel, points=obj.points))
        atomic_json(self.annotation_path(image), dict(version=1, image=image, shape=list(shape),
                    reviewed=bool(reviewed and all(o.class_id is not None for o in objects)), objects=entries))

    def load_annotation(self, image):
        path = self.annotation_path(image)
        if not path.exists():
            return [], False
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            frame = read_image(self.image_path(image))
            if frame is None or list(frame.shape[:2]) != data['shape']:
                raise ValueError('주석과 원본 이미지 크기가 다릅니다.')
            objects = []
            for entry in data['objects']:
                mask_path = self._safe(entry['mask'])
                if mask_path.parent != self.root / 'masks':
                    raise ValueError('마스크 경로가 올바르지 않습니다.')
                mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != frame.shape[:2] or not np.any(mask):
                    raise ValueError('마스크 파일이 손상되었거나 비어 있습니다.')
                cid = entry['class_id']
                if cid is not None and (not isinstance(cid, int) or not 0 <= cid < len(self.classes)):
                    raise ValueError('주석의 클래스 ID가 올바르지 않습니다.')
                objects.append(MaskObject(np.where(mask > 0, 255, 0).astype(np.uint8), cid,
                                          entry['id'], entry.get('points', [])))
            reviewed = bool(data['reviewed'] and all(o.class_id is not None for o in objects))
            return objects, reviewed
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'주석 형식이 올바르지 않습니다: {exc}') from exc
