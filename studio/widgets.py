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


class ImageView(QtWidgets.QWidget):
    def __init__(self, empty='영상 또는 이미지를 선택하세요', parent=None):
        super().__init__(parent)
        self.frame = None
        self.empty = empty
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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
