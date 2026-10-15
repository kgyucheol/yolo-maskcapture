"""Mouse-only drawing paired with left-hand keyboard commands."""
import cv2
import numpy as np

from .document import AnnotationDocument
from .qt import QtCore, QtGui, QtWidgets, Qt, Signal, event_pos, qimage
from .theme import class_color


class MaskCanvas(QtWidgets.QWidget):
    changed = Signal()
    selectionChanged = Signal()
    message = Signal(str)
    command = Signal(str)
    toolChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 240)
        self.doc = None
        self.image = None
        self.class_names = []
        self.tool = 'polygon'
        self.points = []
        self.rect_start = None
        self.rect_end = None
        self.stroke = None
        self.brush_size = 18
        self.overlay = True
        self.scale = 1.0
        self.offset = QtCore.QPointF()
        self.space = False
        self.pan_start = None
        self.cursor_point = None
        self.last_point = None
        self.vertex = None
        self.vertex_points = None

    @property
    def pending(self):
        return bool(self.points or self.rect_start is not None or self.stroke is not None)

    def set_image(self, frame, objects=None):
        self.image = qimage(frame)
        self.doc = AnnotationDocument(frame.shape[:2], objects)
        self.cancel()
        self.fit()

    def clear_image(self):
        self.doc = None
        self.image = None
        self.cursor_point = None
        self.cancel()
        self.selectionChanged.emit()

    def fit(self):
        if self.image:
            self.scale = max(0.01, min((self.width() - 40) / self.image.width(),
                                     (self.height() - 40) / self.image.height()))
            self.offset = QtCore.QPointF((self.width() - self.image.width() * self.scale) / 2,
                                         (self.height() - self.image.height() * self.scale) / 2)
        self.update()

    def resizeEvent(self, event):
        self.fit()

    def set_tool(self, tool):
        if self.pending:
            self.message.emit('먼저 F로 도형을 완료하거나 C로 취소하세요.')
            return
        self.tool = tool
        if tool in ('polygon', 'rectangle') and self.doc:
            self.doc.selected = None
            self.selectionChanged.emit()
        self.toolChanged.emit(tool)
        self.setFocus()
        self.update()

    def new_object(self):
        if self.pending:
            self.message.emit('먼저 F 또는 C로 현재 도형을 정리하세요.')
            return
        if self.doc:
            self.doc.selected = None
            self.selectionChanged.emit()
        self.setFocus()
        self.update()

    def cancel(self):
        self.points = []
        self.rect_start = self.rect_end = self.stroke = None
        self.vertex = self.vertex_points = None
        self.update()

    def _point(self, pos):
        return ((pos.x() - self.offset.x()) / self.scale,
                (pos.y() - self.offset.y()) / self.scale)

    def _inside(self, point):
        return self.doc and 0 <= point[0] < self.doc.shape[1] and 0 <= point[1] < self.doc.shape[0]

    def _clamp(self, point):
        return (int(np.clip(round(point[0]), 0, self.doc.shape[1] - 1)),
                int(np.clip(round(point[1]), 0, self.doc.shape[0] - 1)))

    def _notify(self):
        self.changed.emit()
        self.selectionChanged.emit()
        self.update()

    def finish(self):
        if not self.doc:
            return
        mask = np.zeros(self.doc.shape, np.uint8)
        points = self.points
        if points:
            if len(points) < 3:
                self.message.emit('폴리곤에는 점이 최소 3개 필요합니다.')
                return
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
        elif self.rect_start is not None and self.rect_end is not None:
            x0, y0 = self.rect_start
            x1, y1 = self.rect_end
            if abs(x0 - x1) < 2 or abs(y0 - y1) < 2:
                self.message.emit('더 넓은 사각형을 그려 주세요.')
                return
            points = [(min(x0, x1), min(y0, y1)), (max(x0, x1), min(y0, y1)),
                      (max(x0, x1), max(y0, y1)), (min(x0, x1), max(y0, y1))]
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
        else:
            self.message.emit('객체를 선택하고 숫자 1–9로 태그를 지정하세요.')
            return
        self.doc.add_mask(mask, points)
        self.cancel()
        self._notify()
        self.message.emit('도형 완료 · 숫자 1–9로 클래스 태그를 지정하세요.')

    def assign_tag(self, index):
        if self.pending:
            self.message.emit('F로 도형을 완료한 뒤 숫자로 태그를 지정하세요.')
            return
        if 0 <= index < len(self.class_names) and self.doc and self.doc.current:
            self.doc.assign_class(index)
            self._notify()
            self.message.emit(f'태그 지정: {index + 1} · {self.class_names[index]}')
        else:
            self.message.emit('객체를 선택하고 등록된 태그 번호를 누르세요.')

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#0a101a'))
        if self.image is None:
            p.setPen(QtGui.QColor('#8595ac'))
            p.drawText(self.rect(), Qt.AlignCenter, '이미지를 가져오거나 ZMQ에서 수집하세요\n\nW 폴리곤   ·   E 사각형   ·   R 브러시')
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.translate(self.offset)
        p.scale(self.scale, self.scale)
        p.drawImage(QtCore.QPointF(0, 0), self.image)
        if self.overlay:
            for obj in self.doc.objects:
                color = class_color(obj.class_id)
                rgba = np.zeros((*self.doc.shape, 4), np.uint8)
                rgba[obj.mask > 0] = [color.red(), color.green(), color.blue(), 85]
                overlay = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QtGui.QImage.Format_RGBA8888)
                p.drawImage(QtCore.QPointF(0, 0), overlay)
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                p.setPen(QtGui.QPen(color, (2.5 if obj.id == self.doc.selected else 1.2) / self.scale))
                p.setBrush(Qt.NoBrush)
                for contour in contours:
                    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in contour[:, 0]]))
                if obj.id == self.doc.selected and self.tool == 'select':
                    p.setBrush(color)
                    for x, y in (self.vertex_points if self.vertex_points is not None else obj.points):
                        p.drawEllipse(QtCore.QPointF(x, y), 4 / self.scale, 4 / self.scale)
                ys, xs = np.where(obj.mask > 0)
                if len(xs):
                    text = '태그 미지정' if obj.class_id is None else f'{obj.class_id + 1} {self.class_names[obj.class_id]}'
                    p.save()
                    p.translate(float(xs.min()), max(14 / self.scale, float(ys.min())))
                    p.scale(1 / self.scale, 1 / self.scale)
                    p.setPen(color)
                    p.drawText(QtCore.QPointF(4, -5), text)
                    p.restore()
        p.setPen(QtGui.QPen(QtGui.QColor('#65f2d1'), 2 / self.scale))
        p.setBrush(Qt.NoBrush)
        if self.points:
            path = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self.points])
            p.drawPolyline(path)
            for point in path:
                p.drawEllipse(point, 3 / self.scale, 3 / self.scale)
        if self.rect_start is not None and self.rect_end is not None:
            p.drawRect(QtCore.QRectF(QtCore.QPointF(*self.rect_start), QtCore.QPointF(*self.rect_end)).normalized())
        if self.stroke is not None:
            rgba = np.zeros((*self.doc.shape, 4), np.uint8)
            rgba[self.stroke > 0] = [80, 230, 190, 110]
            p.drawImage(QtCore.QPointF(0, 0), QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                                                       rgba.strides[0], QtGui.QImage.Format_RGBA8888))
        if self.cursor_point is not None and self.tool in ('brush', 'eraser'):
            p.setPen(QtGui.QPen(QtGui.QColor('#ffb5b5' if self.tool == 'eraser' else '#ecfff7'), 1 / self.scale))
            p.drawEllipse(QtCore.QPointF(*self.cursor_point), self.brush_size / 2, self.brush_size / 2)

    def mousePressEvent(self, event):
        self.setFocus()
        pos = event_pos(event)
        if event.button() == Qt.MiddleButton or self.space:
            self.pan_start = pos
            return
        point = self._point(pos)
        if event.button() != Qt.LeftButton or not self._inside(point):
            return
        point = self._clamp(point)
        if self.tool == 'polygon':
            self.points.append(point)
        elif self.tool == 'rectangle':
            self.rect_start = self.rect_end = point
        elif self.tool in ('brush', 'eraser'):
            if self.tool == 'eraser' and self.doc.current is None:
                self.message.emit('지울 객체를 Q로 먼저 선택하세요.')
                return
            self.stroke = self.doc.current.mask.copy() if self.doc.current else np.zeros(self.doc.shape, np.uint8)
            self.last_point = point
            self._brush(point)
        elif self.tool == 'select':
            current = self.doc.current
            if current:
                for i, vertex in enumerate(current.points):
                    if np.hypot(vertex[0] - point[0], vertex[1] - point[1]) * self.scale < 10:
                        self.vertex = i
                        self.vertex_points = list(current.points)
                        return
            self.doc.selected = next((o.id for o in reversed(self.doc.objects) if o.mask[point[1], point[0]]), None)
            self.selectionChanged.emit()
        self.update()

    def _brush(self, point):
        value = 0 if self.tool == 'eraser' else 255
        cv2.line(self.stroke, self.last_point, point, value, self.brush_size)
        cv2.circle(self.stroke, point, max(1, self.brush_size // 2), value, -1)
        self.last_point = point

    def mouseMoveEvent(self, event):
        pos = event_pos(event)
        if self.pan_start is not None:
            self.offset += pos - self.pan_start
            self.pan_start = pos
        elif self.doc:
            point = self._clamp(self._point(pos))
            self.cursor_point = point
            if self.stroke is not None:
                self._brush(point)
            elif self.rect_start is not None and event.buttons() & Qt.LeftButton:
                self.rect_end = point
            elif self.vertex is not None:
                self.vertex_points[self.vertex] = point
        self.update()

    def mouseReleaseEvent(self, event):
        self.pan_start = None
        if event.button() != Qt.LeftButton or not self.doc:
            return
        if self.stroke is not None:
            if np.any(self.stroke):
                self.doc.replace_mask(self.stroke)
                self._notify()
            else:
                self.message.emit('마스크 전체 삭제에는 X를 사용하세요.')
            self.stroke = None
        if self.vertex is not None:
            mask = np.zeros(self.doc.shape, np.uint8)
            cv2.fillPoly(mask, [np.asarray(self.vertex_points, np.int32)], 255)
            if np.any(mask):
                self.doc.replace_mask(mask, self.vertex_points)
                self._notify()
            self.vertex = self.vertex_points = None
        self.update()

    def wheelEvent(self, event):
        if not self.doc:
            return
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.ControlModifier:
            self.brush_size = max(2, min(300, self.brush_size + (2 if delta > 0 else -2)))
            self.message.emit(f'브러시 크기: {self.brush_size}px')
        else:
            pos = event_pos(event)
            before = self._point(pos)
            self.scale = float(np.clip(self.scale * (1.15 if delta > 0 else 1 / 1.15), 0.03, 30))
            self.offset = pos - QtCore.QPointF(before[0] * self.scale, before[1] * self.scale)
        self.update()

    def event(self, event):
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            self.keyPressEvent(event)
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        key, modifiers = event.key(), event.modifiers()
        shift = bool(modifiers & Qt.ShiftModifier)
        if key == Qt.Key_Space:
            self.space = True
        elif key == Qt.Key_S:
            self.command.emit('save')
        elif key in (Qt.Key_A, Qt.Key_D):
            self.command.emit('previous' if key == Qt.Key_A else 'next')
        elif key in (Qt.Key_C, Qt.Key_Escape):
            self.cancel()
        elif key == Qt.Key_F:
            self.finish()
        elif Qt.Key_1 <= key <= Qt.Key_9:
            self.assign_tag(key - Qt.Key_1)
        elif key in (Qt.Key_Tab, Qt.Key_Backtab) and self.doc and self.class_names:
            cid = self.doc.current.class_id if self.doc.current and self.doc.current.class_id is not None else -1
            self.assign_tag((cid + (-1 if shift or key == Qt.Key_Backtab else 1)) % len(self.class_names))
        elif key in (Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R):
            self.set_tool({Qt.Key_Q: 'select', Qt.Key_W: 'polygon', Qt.Key_E: 'rectangle',
                           Qt.Key_R: 'eraser' if shift else 'brush'}[key])
        elif key == Qt.Key_V:
            self.overlay = not self.overlay
            self.update()
        elif self.doc and key == Qt.Key_X and not self.pending:
            self.doc.delete_selected()
            self._notify()
        elif self.doc and key == Qt.Key_Z:
            if self.points and not shift:
                self.points.pop()
                self.update()
            elif not self.pending:
                (self.doc.redo if shift else self.doc.undo)()
                self._notify()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space = False

    def focusOutEvent(self, event):
        self.space = False
        self.pan_start = None
        super().focusOutEvent(event)
