from .qt import QtCore, QtWidgets, QtGui, Qt, qimage


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
        self.frame = qimage(frame) if frame is not None else None
        self.update()

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
