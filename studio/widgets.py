from .qt import QtCore, QtWidgets, QtGui, Qt, Signal, event_pos, qimage


def button(text, callback=None, primary=False):
    widget = QtWidgets.QPushButton(text)
    if primary:
        widget.setObjectName('primary')
    if callback:
        widget.clicked.connect(callback)
    return widget


def label(text, role=None):
    widget = QtWidgets.QLabel(text)
    if role:
        widget.setObjectName(role)
    widget.setWordWrap(True)
    return widget


def unit_field(field, unit):
    """Keep editable values separate from their unit and explanatory text."""
    if isinstance(field, QtWidgets.QAbstractSpinBox):
        field.setAlignment(Qt.AlignRight)
    unit = unit.strip()
    if not unit:
        return field
    widget = QtWidgets.QWidget()
    widget.setFocusProxy(field)
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(field, 1)
    suffix = label(unit, 'numericUnit')
    suffix.setWordWrap(False)
    suffix.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
    row.addWidget(suffix)
    return widget


def help_label(text, explanation):
    """A visible, keyboard-accessible explanation beside a form label."""
    widget = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    caption = label(text)
    caption.setWordWrap(False)
    row.addWidget(caption)
    info = QtWidgets.QToolButton()
    info.setObjectName('helpIcon')
    info.setText('ⓘ')
    info.setAccessibleName(text + ' 도움말')
    info.setToolTip(explanation)
    info.setCursor(Qt.PointingHandCursor)
    info.setFixedSize(23, 23)
    info.clicked.connect(lambda: QtWidgets.QToolTip.showText(
        info.mapToGlobal(info.rect().bottomLeft()), explanation, info, info.rect(), 15000))
    row.addWidget(info)
    row.addStretch()
    return widget


def device_selector():
    field = QtWidgets.QComboBox()
    field.addItem('자동 (GPU 우선)', '')
    field.addItem('GPU 0 (CUDA)', '0')
    field.addItem('CPU', 'cpu')
    field.setToolTip('자동: CUDA GPU가 있으면 사용하고, 없으면 CPU를 사용합니다.\n'
                     'GPU 0: 첫 번째 NVIDIA GPU를 지정합니다. CUDA 버전 PyTorch가 필요합니다.')
    return field


class ShortcutButton(QtWidgets.QPushButton):
    def sizeHint(self):
        base = super().sizeHint()
        return base.expandedTo(self.layout().sizeHint() + QtCore.QSize(2, 2)) if self.layout() else base

    def minimumSizeHint(self):
        return self.sizeHint()


def shortcut_button(key, text, callback, tooltip=''):
    widget = ShortcutButton()
    widget.setObjectName('shortcutTool')
    widget.clicked.connect(callback)
    widget.setCheckable(True)
    widget.setMinimumHeight(44)
    widget.setAccessibleName(f'{key} {text}')
    widget.setToolTip(tooltip or f'{key} · {text}')
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(9, 6, 10, 6)
    row.setSpacing(7)
    for text, role in [(key, 'shortcutKey'), (text, 'shortcutCaption')]:
        child = label(text, role)
        child.setWordWrap(False)
        child.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addWidget(child)
    def highlight(checked):
        for child in widget.findChildren(QtWidgets.QLabel):
            child.setProperty('active', checked)
            child.style().unpolish(child)
            child.style().polish(child)
    widget.toggled.connect(highlight)
    return widget


def shortcut_hint(key, text):
    widget = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(0, 2, 0, 2)
    keycap = label(key, 'shortcutKey')
    keycap.setWordWrap(False)
    row.addWidget(keycap)
    row.addWidget(label(text, 'subtitle'), 1)
    return widget


def card(title=None):
    frame = QtWidgets.QFrame()
    frame.setObjectName('card')
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
    if title:
        layout.addWidget(label(title, 'section'))
    return frame, layout


def scroll_panel(widget, width=330):
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setMinimumWidth(width)
    scroll.setMaximumWidth(width + 14)
    scroll.setWidget(widget)
    return scroll


def page_header(title, description):
    box = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 8)
    layout.setSpacing(5)
    layout.addWidget(label(title, 'title'))
    layout.addWidget(label(description, 'subtitle'))
    return box


class Toast(QtWidgets.QLabel):
    """영상 위에 잠깐 떠서 결과를 알리는 알림. 새 메시지는 내용을 바꾸고 표시 시간을 다시 센다."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('toast')
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.hide()

    def flash(self, text, ms=1000, error=False):
        self.setProperty('error', bool(error))
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText(text)
        self.adjustSize()
        self.place()
        self.show()
        self.raise_()
        self.timer.start(ms)

    def place(self):
        parent = self.parentWidget()
        if parent is not None:
            self.move(max(0, (parent.width() - self.width()) // 2), max(0, parent.height() - self.height() - 28))


class ImageView(QtWidgets.QWidget):
    frameChanged = Signal()
    def __init__(self, empty='영상 또는 이미지를 선택하세요', parent=None):
        super().__init__(parent)
        self.frame = None
        self.empty = empty
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.toast = Toast(self)

    def flash(self, text, ms=1000, error=False):
        self.toast.flash(text, ms, error)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.toast.place()

    def set_frame(self, frame):
        self.frame = (frame.copy() if isinstance(frame, QtGui.QImage) else qimage(frame)) if frame is not None else None
        self.update()
        self.frameChanged.emit()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#0a101a'))
        if self.frame is None:
            painter.setPen(QtGui.QColor('#8595ac'))
            painter.drawText(self.rect().adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, self.empty)
            return
        size = self.frame.size().scaled(self.size() - QtCore.QSize(16, 16), Qt.KeepAspectRatio)
        rect = QtCore.QRect(QtCore.QPoint((self.width() - size.width()) // 2,
                                        (self.height() - size.height()) // 2), size)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawImage(rect, self.frame)


class SwipeComparison(QtWidgets.QWidget):
    """Full-size before/after comparison with a draggable divider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original = self.augmented = None
        self.fraction = .5
        self.setMinimumSize(200, 180)
        self.setCursor(Qt.SplitHCursor)
        self.setToolTip('분할선을 좌우로 드래그하면 같은 크기로 원본과 증강본을 비교합니다.')

    def image_rect(self):
        image = self.original if self.original is not None else self.augmented
        if image is None:
            return QtCore.QRect()
        size = image.size().scaled(self.size() - QtCore.QSize(16, 16), Qt.KeepAspectRatio)
        return QtCore.QRect(QtCore.QPoint((self.width() - size.width()) // 2,
                                         (self.height() - size.height()) // 2), size)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#0a101a'))
        if self.original is None:
            painter.setPen(QtGui.QColor('#8595ac'))
            painter.drawText(self.rect(), Qt.AlignCenter, '완료된 라벨 이미지가 필요합니다.')
            return
        rect = self.image_rect()
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawImage(rect, self.augmented if self.augmented is not None else self.original)
        if self.augmented is None:
            return
        divider = rect.left() + int(rect.width() * self.fraction)
        painter.save()
        painter.setClipRect(QtCore.QRect(rect.left(), rect.top(), divider - rect.left(), rect.height()))
        painter.drawImage(rect, self.original)
        painter.restore()
        painter.setPen(QtGui.QPen(QtGui.QColor('#8ef3d9'), 2))
        painter.drawLine(divider, rect.top(), divider, rect.bottom())
        painter.setBrush(QtGui.QColor('#153a38'))
        painter.drawEllipse(QtCore.QPoint(divider, rect.center().y()), 17, 17)
        painter.drawText(QtCore.QRect(divider - 17, rect.center().y() - 17, 34, 34), Qt.AlignCenter, '↔')
        for text, x in (('원본', rect.left() + 8), ('증강', rect.right() - 63)):
            badge = QtCore.QRect(x, rect.top() + 8, 55, 25)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(10, 20, 30, 225))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QtGui.QColor('#e8edf6'))
            painter.drawText(badge, Qt.AlignCenter, text)

    def move_divider(self, pos):
        rect = self.image_rect()
        if rect.width() > 0:
            self.fraction = max(0., min(1., (pos.x() - rect.left()) / rect.width()))
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.move_divider(event_pos(event))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.move_divider(event_pos(event))


class ImageComparison(QtWidgets.QWidget):
    """Large swipe comparison plus optional adaptive two-image arrangement."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['슬라이더로 크게 비교', '두 장 함께 보기'])
        layout.addWidget(self.mode)
        self.stack = QtWidgets.QStackedWidget()
        self.swipe = SwipeComparison()
        self.stack.addWidget(self.swipe)
        self.pair_panel = QtWidgets.QWidget()
        self.row = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, self.pair_panel)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(10)
        self.original_view = ImageView('완료된 라벨 이미지가 필요합니다.')
        self.augmented_view = ImageView('데이터 증강을 켜면 생성 예정 이미지를 비교할 수 있습니다.')
        for title, view in [('학습 원본', self.original_view), ('생성 예정 증강본', self.augmented_view)]:
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(4)
            column.addWidget(label(title, 'subtitle'))
            view.setMinimumSize(160, 100)
            column.addWidget(view, 1)
            self.row.addLayout(column, 1)
            view.frameChanged.connect(self.frames_changed)
        self.stack.addWidget(self.pair_panel)
        layout.addWidget(self.stack, 1)
        self.mode.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.mode.currentIndexChanged.connect(self.update_direction)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_direction()

    def update_direction(self):
        image = self.original_view.frame
        aspect = image.width() / image.height() if image is not None else 848 / 480
        width, height = self.pair_panel.width(), self.pair_panel.height()
        side = min((width - 10) / 2, max(0, height - 30) * aspect)
        stacked = min(width, max(0, (height - 70) / 2) * aspect)
        direction = QtWidgets.QBoxLayout.LeftToRight if side >= stacked else QtWidgets.QBoxLayout.TopToBottom
        if self.row.direction() != direction:
            self.row.setDirection(direction)

    def frames_changed(self):
        self.swipe.original, self.swipe.augmented = self.original_view.frame, self.augmented_view.frame
        self.swipe.update()
        self.update_direction()

    def copy_frames(self, original, augmented):
        self.original_view.set_frame(original)
        self.augmented_view.set_frame(augmented)
