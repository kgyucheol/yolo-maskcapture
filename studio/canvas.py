"""Mouse-only drawing paired with left-hand keyboard commands."""
import cv2
import numpy as np

from .document import AnnotationDocument
from .qt import QtCore, QtGui, QtWidgets, Qt, Signal, event_pos, qimage
from .theme import class_color
from .widgets import Toast


class MaskCanvas(QtWidgets.QWidget):
    changed = Signal()
    selectionChanged = Signal()
    message = Signal(str)
    command = Signal(str)
    toolChanged = Signal(str)
    samRequested = Signal(int, object, object)
    samCleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 240)
        self.doc = None
        self.image = None
        self.frame = None
        self.image_revision = 0
        self.sam_revision = 0
        self.sam_points = []
        self.sam_labels = []
        self.sam_preview = None
        self.sam_redo = []
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
        self.zoom_toast = Toast(self)

    @property
    def pending(self):
        return bool(self.sam_points or self.points or self.rect_start is not None or self.stroke is not None)

    def set_image(self, frame, objects=None):
        self.frame = frame.copy()
        self.image_revision += 1
        self.image = qimage(frame)
        self.doc = AnnotationDocument(frame.shape[:2], objects)
        self.cancel()
        self.fit()

    def clear_image(self):
        self.doc = None
        self.image = None
        self.frame = None
        self.image_revision += 1
        self.cursor_point = None
        self.cancel()
        self.zoom_toast.hide()
        self.selectionChanged.emit()

    def fit(self):
        self.pan_start = None
        self.space = False
        if self.image:
            self.scale = max(0.01, min((self.width() - 40) / self.image.width(),
                                     (self.height() - 40) / self.image.height()))
            self.offset = QtCore.QPointF((self.width() - self.image.width() * self.scale) / 2,
                                         (self.height() - self.image.height() * self.scale) / 2)
        self.update()

    def reset_view(self):
        self.fit()
        if self.image:
            self.show_zoom()
        self.setFocus()

    def show_zoom(self):
        self.zoom_toast.flash(f'{self.scale * 100:.0f}% · 원본 픽셀 기준', ms=1100)

    def resizeEvent(self, event):
        self.fit()
        self.zoom_toast.place()

    def set_tool(self, tool):
        self.setFocus()
        if self.pending:
            self.message.emit('먼저 F로 도형을 완료하거나 C로 취소하세요.')
            self.toolChanged.emit(self.tool)
            return
        self.tool = tool
        if tool in ('polygon', 'rectangle', 'sam') and self.doc:
            self.doc.selected = None
            self.selectionChanged.emit()
        self.toolChanged.emit(tool)
        self.setFocus()
        self.update()

    def new_object(self):
        self.setFocus()
        if self.pending:
            self.message.emit('먼저 F 또는 C로 현재 도형을 정리하세요.')
            return
        if self.doc:
            self.doc.selected = None
            self.selectionChanged.emit()
        self.setFocus()
        self.update()

    def cancel(self):
        self.sam_revision += 1
        self.sam_points, self.sam_labels, self.sam_redo = [], [], []
        self.sam_preview = None
        self.samCleared.emit()
        self.points = []
        self.rect_start = self.rect_end = self.stroke = None
        self.vertex = self.vertex_points = None
        self.update()

    def _request_sam(self):
        self.sam_revision += 1
        self.sam_preview = None
        if self.sam_points:
            self.message.emit('AI 계산 중… · 좌클릭 포함 / 우클릭 제외 · C 취소')
            self.samRequested.emit(self.sam_revision, list(self.sam_points), list(self.sam_labels))
        else:
            self.samCleared.emit()
        self.update()

    def accept_sam(self, revision, mask):
        if revision != self.sam_revision or not self.sam_points or not self.doc:
            return False
        mask = np.asarray(mask)
        if mask.shape != self.doc.shape or not np.isfinite(mask).all() or not np.any(mask > 0):
            self.fail_sam(revision, '마스크가 비어 있거나 크기가 맞지 않습니다. 포함 점을 추가하거나 C로 다시 시작하세요.')
            return False
        self.sam_preview = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.message.emit('AI 미리보기 · 좌클릭 추가 / 우클릭 제외 · F 확정 → 숫자 태그 · Z 점 취소')
        self.update()
        return True

    def fail_sam(self, revision, error):
        if revision == self.sam_revision and self.sam_points:
            self.sam_preview = None
            self.message.emit('AI 마스크 실패: ' + error)
            self.update()

    def undo_sam(self, redo=False):
        if redo and self.sam_redo:
            point, label = self.sam_redo.pop()
            self.sam_points.append(point)
            self.sam_labels.append(label)
        elif not redo and self.sam_points:
            self.sam_redo.append((self.sam_points.pop(), self.sam_labels.pop()))
        else:
            return
        self._request_sam()

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
        if self.sam_points:
            if self.sam_preview is None:
                self.message.emit('최신 AI 마스크가 준비된 뒤 F로 확정하세요. 오류가 있으면 점을 추가하거나 C로 취소하세요.')
                return
            self.doc.add_mask(self.sam_preview)
            self.cancel()
            self._notify()
            self.message.emit('AI 마스크 확정 · 1–9 태그 · R 브러시 / Shift+R 지우개로 경계 보정 · T 다음 객체')
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

    def mask_label_rect(self, obj, text):
        """Anchor each badge inside an actual visible mask region, including concave masks."""
        height, width = obj.mask.shape
        x0 = max(0, int(np.floor(-self.offset.x() / self.scale)))
        y0 = max(0, int(np.floor(-self.offset.y() / self.scale)))
        x1 = min(width, int(np.ceil((self.width() - self.offset.x()) / self.scale)))
        y1 = min(height, int(np.ceil((self.height() - self.offset.y()) / self.scale)))
        if x1 <= x0 or y1 <= y0:
            return QtCore.QRectF()
        visible = obj.mask[y0:y1, x0:x1]
        if not np.any(visible):
            return QtCore.QRectF()
        padded = cv2.copyMakeBorder(visible, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
        _, _, _, (x, y) = cv2.minMaxLoc(distance)
        anchor = self.offset + QtCore.QPointF((x + x0 - 1) * self.scale, (y + y0 - 1) * self.scale)
        left, top = max(0., self.offset.x()), max(0., self.offset.y())
        right = min(float(self.width()), self.offset.x() + width * self.scale)
        bottom = min(float(self.height()), self.offset.y() + height * self.scale)
        metrics = self.fontMetrics()
        badge_width = min(metrics.horizontalAdvance(text) + 14, right - left)
        badge_height = min(metrics.height() + 8, bottom - top)
        bx = max(left, min(anchor.x() - badge_width / 2, right - badge_width))
        by = max(top, min(anchor.y() - badge_height / 2, bottom - badge_height))
        return QtCore.QRectF(bx, by, badge_width, badge_height)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#0a101a'))
        if self.image is None:
            p.setPen(QtGui.QColor('#8595ac'))
            p.drawText(self.rect(), Qt.AlignCenter, '이미지를 가져오거나 ZMQ에서 수집하세요\n\nT AI 클릭   ·   W 폴리곤   ·   E 사각형   ·   R 브러시')
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
                text = '태그 미지정' if obj.class_id is None else f'{obj.class_id + 1} {self.class_names[obj.class_id]}'
                badge = self.mask_label_rect(obj, text)
                if not badge.isEmpty():
                    p.save()
                    p.resetTransform()
                    p.setPen(QtGui.QPen(color, 1))
                    p.setBrush(QtGui.QColor(10, 20, 30, 225))
                    p.drawRoundedRect(badge, 5, 5)
                    caption = self.fontMetrics().elidedText(text, Qt.ElideRight, max(0, int(badge.width()) - 10))
                    p.drawText(badge, Qt.AlignCenter, caption)
                    p.restore()
        if self.sam_preview is not None:
            rgba = np.zeros((*self.doc.shape, 4), np.uint8)
            rgba[self.sam_preview > 0] = [80, 230, 190, 125]
            p.drawImage(QtCore.QPointF(0, 0), QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                        rgba.strides[0], QtGui.QImage.Format_RGBA8888))
            p.setPen(QtGui.QPen(QtGui.QColor('#65f2d1'), 2 / self.scale, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            contours, _ = cv2.findContours(self.sam_preview, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in contour[:, 0]]))
        for (x, y), foreground in zip(self.sam_points, self.sam_labels):
            color = QtGui.QColor('#65f2d1' if foreground else '#ff7b8d')
            p.setPen(QtGui.QPen(QtGui.QColor('#101925'), 2 / self.scale))
            p.setBrush(color)
            p.drawEllipse(QtCore.QPointF(x, y), 7 / self.scale, 7 / self.scale)
            p.setPen(QtGui.QPen(QtGui.QColor('#101925'), 1.7 / self.scale))
            p.drawLine(QtCore.QPointF(x - 3 / self.scale, y), QtCore.QPointF(x + 3 / self.scale, y))
            if foreground:
                p.drawLine(QtCore.QPointF(x, y - 3 / self.scale), QtCore.QPointF(x, y + 3 / self.scale))
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
        if self.tool == 'sam' and self._inside(point) and event.button() in (Qt.LeftButton, Qt.RightButton):
            foreground = int(event.button() == Qt.LeftButton and not event.modifiers() & Qt.ShiftModifier)
            if not self.sam_points and not foreground:
                self.message.emit('먼저 물체 안쪽을 좌클릭해 포함 점을 지정하세요.')
                return
            self.sam_points.append(self._clamp(point))
            self.sam_labels.append(foreground)
            self.sam_redo.clear()
            self._request_sam()
            return
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
        if not delta:
            return
        if event.modifiers() & Qt.ControlModifier:
            self.brush_size = max(2, min(300, self.brush_size + (2 if delta > 0 else -2)))
            self.message.emit(f'브러시 크기: {self.brush_size}px')
        else:
            pos = event_pos(event)
            before = (pos - self.offset) / self.scale
            previous = self.scale
            self.scale = float(np.clip(self.scale * (1.15 if delta > 0 else 1 / 1.15), 0.03, 30))
            self.offset = pos - before * self.scale
            if previous != self.scale:
                self.show_zoom()
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
        elif key in (Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T):
            self.set_tool({Qt.Key_Q: 'select', Qt.Key_W: 'polygon', Qt.Key_E: 'rectangle',
                           Qt.Key_R: 'eraser' if shift else 'brush', Qt.Key_T: 'sam'}[key])
        elif key == Qt.Key_V:
            self.overlay = not self.overlay
            self.update()
        elif self.doc and key == Qt.Key_X and not self.pending:
            self.doc.delete_selected()
            self._notify()
        elif self.doc and key == Qt.Key_Z:
            if self.sam_points or (self.tool == 'sam' and self.sam_redo):
                self.undo_sam(shift)
            elif self.points and not shift:
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
