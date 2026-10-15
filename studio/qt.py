"""Prefer Qt 6; the optional system Qt 5 fallback also works offline."""
try:
    from PySide6 import QtCore, QtGui, QtWidgets, QtTest
    Signal = QtCore.Signal
    BINDING = 'PySide6'
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
        Signal = QtCore.pyqtSignal
        BINDING = 'PyQt5'
    except ImportError as exc:
        raise ImportError('UI 실행에는 PySide6가 필요합니다: python -m pip install -r requirements-ui.txt') from exc

Qt = QtCore.Qt


def event_pos(event):
    return event.position() if hasattr(event, 'position') else QtCore.QPointF(event.pos())


def qimage(bgr):
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                        QtGui.QImage.Format_RGB888).copy()
