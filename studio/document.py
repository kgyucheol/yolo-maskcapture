"""Image-coordinate annotation state with bounded undo history."""
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


@dataclass
class MaskObject:
    mask: np.ndarray
    class_id: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    points: list = field(default_factory=list)


class AnnotationDocument:
    def __init__(self, shape, objects=None):
        self.shape = tuple(shape[:2])
        self.objects = deepcopy(objects or [])
        self.selected = self.objects[-1].id if self.objects else None
        self._undo, self._redo = [], []
        self.dirty = False

    @property
    def current(self):
        return next((o for o in self.objects if o.id == self.selected), None)

    def _state(self):
        return deepcopy(self.objects), self.selected

    def _checkpoint(self):
        self._undo.append(self._state())
        self._undo = self._undo[-30:]
        self._redo.clear()
        self.dirty = True

    def _mask(self, mask):
        if mask.shape != self.shape or not np.any(mask):
            raise ValueError('마스크 크기가 맞지 않거나 비어 있습니다.')
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def add_mask(self, mask, points=None):
        mask = self._mask(mask)
        self._checkpoint()
        obj = MaskObject(mask, points=list(points or []))
        self.objects.append(obj)
        self.selected = obj.id
        return obj

    def assign_class(self, class_id):
        if self.current is None:
            return False
        if not isinstance(class_id, int) or class_id < 0:
            raise ValueError('클래스 번호가 올바르지 않습니다.')
        if self.current.class_id == class_id:
            return False
        self._checkpoint()
        self.current.class_id = class_id
        return True

    def replace_mask(self, mask, points=None):
        mask = self._mask(mask)
        if self.current is None:
            return self.add_mask(mask, points)
        if np.array_equal(self.current.mask, mask):
            return
        self._checkpoint()
        self.current.mask = mask
        self.current.points = list(points or [])

    def delete_selected(self):
        if self.current is not None:
            self._checkpoint()
            self.objects = [o for o in self.objects if o.id != self.selected]
            self.selected = self.objects[-1].id if self.objects else None

    def undo(self):
        if self._undo:
            self._redo.append(self._state())
            self.objects, self.selected = self._undo.pop()
            self.dirty = True

    def redo(self):
        if self._redo:
            self._undo.append(self._state())
            self.objects, self.selected = self._redo.pop()
            self.dirty = True
