import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from studio.qt import Qt, QtCore, QtGui, QtWidgets


def test_zoom_shows_actual_pixel_percentage_and_fit_centers_without_changing_masks():
    from studio.canvas import MaskCanvas
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = MaskCanvas()
    canvas.resize(900, 600)
    canvas.show()
    canvas.set_image(np.zeros((480, 848, 3), np.uint8))
    canvas.doc.add_mask(np.ones((480, 848), np.uint8))
    mask = canvas.doc.current.mask.copy()
    canvas.scale = 1.
    event = QtGui.QWheelEvent(QtCore.QPointF(200, 200), QtCore.QPointF(200, 200),
        QtCore.QPoint(), QtCore.QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    canvas.wheelEvent(event)
    assert canvas.zoom_toast.isVisible()
    assert '115%' in canvas.zoom_toast.text()
    canvas.offset = QtCore.QPointF(-500, 400)
    canvas.pan_start = QtCore.QPointF(20, 20)
    canvas.space = True
    canvas.reset_view()
    center = canvas.offset + QtCore.QPointF(424 * canvas.scale, 240 * canvas.scale)
    assert abs(center.x() - 450) < .01 and abs(center.y() - 300) < .01
    assert canvas.pan_start is None and not canvas.space
    assert np.array_equal(mask, canvas.doc.current.mask)
    canvas.zoom_toast.timer.timeout.emit()
    assert not canvas.zoom_toast.isVisible()
    canvas.close()


def test_dataset_comparison_owns_the_preview_space_and_can_open_larger():
    from studio.window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.navigation.setCurrentRow(2)
    window.show()
    app.processEvents()
    page = window.dataset
    assert not page.report.isVisible()
    frame = np.zeros((480, 848, 3), np.uint8)
    page.original_view.set_frame(frame)
    page.augmented_view.set_frame(frame)
    app.processEvents()
    assert page.comparison.swipe.isVisible()
    visible_image = page.comparison.swipe.image_rect()
    assert visible_image.width() >= 500 and visible_image.height() >= 280
    page.open_comparison()
    app.processEvents()
    assert page.comparison_dialog.isVisible()
    assert page.large_comparison.original_view.frame == page.original_view.frame
    page.comparison_dialog.close()
    window.close()
