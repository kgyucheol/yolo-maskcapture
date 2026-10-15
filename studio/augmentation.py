"""Deterministic image/mask augmentation without touching source annotations."""
import hashlib
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

METHOD_NAMES = dict(horizontal_flip='좌우 반전', vertical_flip='상하 반전', rotation='회전',
                    scale='크기 변화', translate='위치 이동', brightness='밝기', contrast='대비',
                    saturation='채도', hue='색상', noise='노이즈', blur='흐림')


@dataclass(frozen=True)
class AugmentConfig:
    copies: int = 2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: float = 0
    scale: float = 0
    translate: float = 0
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    hue: float = 0
    noise: float = 0
    blur: bool = False
    mode: str = 'all'
    max_transforms: int = 3
    single_ratio: float = 0.5
    method_weights: tuple = ()

    def __post_init__(self):
        if not isinstance(self.copies, int) or not 0 <= self.copies <= 20:
            raise ValueError('이미지당 증강 수는 0~20이어야 합니다.')
        if self.mode not in ('single', 'random', 'all'):
            raise ValueError('증강 적용 방식이 올바르지 않습니다.')
        if not isinstance(self.max_transforms, int) or not 1 <= self.max_transforms <= len(METHOD_NAMES):
            raise ValueError('동시 적용 기법 수가 올바르지 않습니다.')
        if not isinstance(self.single_ratio, (float, int)) or not math.isfinite(self.single_ratio) or not 0 <= self.single_ratio <= 1:
            raise ValueError('단일 기법 비율은 0~1이어야 합니다.')
        try:
            weights = tuple((key, float(value)) for key, value in self.method_weights)
        except (TypeError, ValueError) as exc:
            raise ValueError('기법 선택 비중이 올바르지 않습니다.') from exc
        if (len(dict(weights)) != len(weights) or any(key not in METHOD_NAMES or not math.isfinite(value)
                                                     or value < 0 for key, value in weights)):
            raise ValueError('기법 선택 비중은 등록된 기법의 0 이상 값이어야 합니다.')
        object.__setattr__(self, 'method_weights', weights)
        for name, upper in [('rotation', 180), ('scale', 0.5), ('translate', 0.3),
                            ('brightness', 0.5), ('contrast', 0.5), ('saturation', 0.5),
                            ('hue', 30), ('noise', 25)]:
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f'{name}: 0~{upper} 범위의 값을 지정하세요.')
        if self.copies and not active_methods(self):
            raise ValueError('증강 방법을 하나 이상 선택하세요.')


def active_methods(config):
    weights = dict(config.method_weights)
    return [name for name in METHOD_NAMES if getattr(config, name) and weights.get(name, 1) > 0]


def _select_methods(config, rng, single=None):
    enabled = active_methods(config)
    if not enabled:
        raise ValueError('증강 방법을 하나 이상 선택하세요.')
    if config.mode == 'all':
        return tuple(enabled)
    cap = min(len(enabled), config.max_transforms)
    if single is None:
        single = config.mode == 'single' or cap < 2 or rng.random() < config.single_ratio
    count = 1 if single or cap < 2 else int(rng.integers(2, cap + 1))
    weights = dict(config.method_weights)
    probabilities = np.asarray([weights.get(name, 1) for name in enabled], np.float64)
    probabilities /= probabilities.sum()
    selected = set(rng.choice(enabled, size=count, replace=False, p=probabilities).tolist())
    return tuple(name for name in enabled if name in selected)


def augmentation_plan(config, count, seed):
    """Fix single/composite counts globally, then sample weighted recipes reproducibly."""
    if count <= 0:
        return []
    rng = np.random.default_rng(sample_seed(seed, '__augmentation_plan__', 0))
    if config.mode == 'all':
        return [tuple(active_methods(config))] * count
    nsingle, _ = composition_counts(config, count)
    kinds = [True] * nsingle + [False] * (count - nsingle)
    rng.shuffle(kinds)
    return [_select_methods(config, rng, single=single) for single in kinds]


def composition_counts(config, count):
    if config.mode == 'all':
        return (count, 0) if len(active_methods(config)) == 1 else (0, count)
    can_combine = len(active_methods(config)) >= 2 and config.max_transforms >= 2
    nsingle = count if config.mode == 'single' or not can_combine else int(count * config.single_ratio + 0.5)
    return nsingle, count - nsingle


def sample_seed(seed, image, index):
    payload = f'{seed}:{image}:{index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def augment_sample(frame, objects, config, seed, methods=None):
    rng = np.random.default_rng(seed)
    recipe = _select_methods(config, rng) if methods is None else tuple(methods)
    if not recipe or len(set(recipe)) != len(recipe) or not set(recipe) <= set(active_methods(config)):
        raise ValueError('선택한 증강 기법 조합이 올바르지 않습니다.')
    selected = set(recipe)
    # Keep the caller's immutable config; zero only methods not selected for this sample.
    config = replace(config, **{name: False if isinstance(getattr(config, name), bool) else 0
                                for name in METHOD_NAMES if name not in selected})
    methods = [name for name in METHOD_NAMES if name in selected]
    image = frame.copy()
    masks = [obj.mask.copy() for obj in objects]
    height, width = image.shape[:2]
    operations = {}
    flip_x, flip_y = config.horizontal_flip, config.vertical_flip
    if flip_x:
        image = cv2.flip(image, 1)
        masks = [cv2.flip(m, 1) for m in masks]
        operations['horizontal_flip'] = True
    if flip_y:
        image = cv2.flip(image, 0)
        masks = [cv2.flip(m, 0) for m in masks]
        operations['vertical_flip'] = True
    if config.rotation or config.scale or config.translate:
        angle = float(rng.uniform(-config.rotation, config.rotation))
        scale = float(rng.uniform(1 - config.scale, 1 + config.scale))
        dx, dy = rng.uniform(-config.translate, config.translate, 2) * [width, height]
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
        matrix[:, 2] += [dx, dy]
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        masks = [cv2.warpAffine(m, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in masks]
        operations['affine'] = dict(degrees=angle, scale=scale, dx=float(dx), dy=float(dy))
    if config.brightness or config.contrast:
        alpha = float(rng.uniform(1 - config.contrast, 1 + config.contrast))
        beta = float(rng.uniform(-config.brightness, config.brightness) * 255)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        operations['lighting'] = dict(contrast=alpha, brightness=beta)
    if config.saturation or config.hue:
        saturation = float(rng.uniform(1 - config.saturation, 1 + config.saturation))
        hue = float(rng.uniform(-config.hue, config.hue))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        operations['color'] = dict(saturation=saturation, hue=hue)
    if config.noise:
        image = np.clip(image.astype(np.float32) + rng.normal(0, config.noise, image.shape), 0, 255).astype(np.uint8)
        operations['noise_std'] = config.noise
    if config.blur:
        sigma = float(rng.uniform(0.3, 1.2))
        image = cv2.GaussianBlur(image, (3, 3), sigma)
        operations['blur_sigma'] = sigma
    transformed, dropped = [], []
    for obj, mask in zip(objects, masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not any(len(c) >= 3 and cv2.contourArea(c) >= 1 for c in contours):
            dropped.append(obj.id)
        else:
            transformed.append(replace(obj, mask=mask, points=[]))
    return image, transformed, dict(methods=methods, operations=operations, dropped_objects=dropped)
