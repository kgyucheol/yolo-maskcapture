import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from studio.canvas import MaskCanvas
from studio.qt import QtCore, QtWidgets, QtTest, Qt


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def canvas(app):
    widget = MaskCanvas()
    widget.resize(600, 400)
    widget.set_image(np.zeros((100, 200, 3), np.uint8))
    widget.class_names = ['컵', '병']
    widget.show()
    widget.setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(widget, Qt.Key_T)
    yield widget
    widget.close()


def click(canvas, x, y, negative=False, shift=False):
    pos = QtCore.QPoint(round(canvas.offset.x() + x * canvas.scale),
                       round(canvas.offset.y() + y * canvas.scale))
    QtTest.QTest.mouseClick(canvas, Qt.RightButton if negative else Qt.LeftButton,
                           Qt.ShiftModifier if shift else Qt.NoModifier, pos)


def mask():
    result = np.zeros((100, 200), np.uint8)
    result[20:70, 40:110] = 255
    return result


def test_click_preview_finish_tag_and_single_step_mask_undo(canvas):
    requests = []
    canvas.samRequested.connect(lambda *args: requests.append(args))
    click(canvas, 70, 40)
    assert canvas.pending and canvas.doc.objects == []
    request, points, labels = requests[-1]
    assert points == [(70, 40)] and labels == [1]
    canvas.accept_sam(request, mask())
    QtTest.QTest.keyClick(canvas, Qt.Key_F)
    assert not canvas.pending and len(canvas.doc.objects) == 1
    assert np.array_equal(canvas.doc.current.mask, mask())
    QtTest.QTest.keyClick(canvas, Qt.Key_2)
    assert canvas.doc.current.class_id == 1
    QtTest.QTest.keyClick(canvas, Qt.Key_Z)
    assert canvas.doc.current.class_id is None
    QtTest.QTest.keyClick(canvas, Qt.Key_Z)
    assert canvas.doc.objects == []


def test_negative_points_refine_one_object_and_old_result_cannot_be_committed(canvas):
    requests = []
    canvas.samRequested.connect(lambda *args: requests.append(args))
    click(canvas, 70, 40)
    first = requests[-1][0]
    canvas.accept_sam(first, mask())
    click(canvas, 130, 40, negative=True)
    latest, points, labels = requests[-1]
    assert points == [(70, 40), (130, 40)] and labels == [1, 0]
    canvas.accept_sam(first, mask())
    QtTest.QTest.keyClick(canvas, Qt.Key_F)
    assert canvas.doc.objects == [] and canvas.pending
    canvas.accept_sam(latest, mask())
    QtTest.QTest.keyClick(canvas, Qt.Key_F)
    assert len(canvas.doc.objects) == 1


def test_shift_click_excludes_and_z_redoes_prompt_without_editing_saved_objects(canvas):
    canvas.doc.add_mask(mask())
    click(canvas, 70, 40)
    click(canvas, 120, 40, shift=True)
    assert canvas.sam_labels == [1, 0]
    QtTest.QTest.keyClick(canvas, Qt.Key_Z)
    assert canvas.sam_labels == [1] and len(canvas.doc.objects) == 1
    QtTest.QTest.keyClick(canvas, Qt.Key_Z, Qt.ShiftModifier)
    assert canvas.sam_labels == [1, 0]


def test_cancel_and_image_change_reject_inflight_results(canvas):
    requests = []
    canvas.samRequested.connect(lambda *args: requests.append(args))
    click(canvas, 70, 40)
    old = requests[-1][0]
    QtTest.QTest.keyClick(canvas, Qt.Key_C)
    canvas.accept_sam(old, mask())
    assert not canvas.pending and canvas.sam_preview is None
    click(canvas, 70, 40)
    old = requests[-1][0]
    canvas.set_image(np.zeros((100, 200, 3), np.uint8))
    canvas.accept_sam(old, mask())
    assert not canvas.pending and canvas.doc.objects == []


def test_pending_sam_blocks_tool_switch_and_empty_or_failed_result_blocks_finish(canvas):
    click(canvas, 70, 40, negative=True)
    assert not canvas.pending  # cannot start with only a background point
    click(canvas, 70, 40)
    canvas.set_tool('brush')
    assert canvas.tool == 'sam'
    canvas.accept_sam(canvas.sam_revision, np.zeros((100, 200), np.uint8))
    canvas.finish()
    assert canvas.doc.objects == []
    canvas.accept_sam(canvas.sam_revision, mask())
    click(canvas, 140, 40, negative=True)
    canvas.fail_sam(canvas.sam_revision, 'out of memory')
    canvas.finish()
    assert canvas.doc.objects == [] and canvas.sam_preview is None


def test_confirmed_mask_can_be_brushed_erased_and_next_sam_click_starts_new_object(canvas):
    click(canvas, 70, 40)
    canvas.accept_sam(canvas.sam_revision, mask())
    canvas.finish()
    first_id = canvas.doc.selected
    canvas.assign_tag(0)
    QtTest.QTest.keyClick(canvas, Qt.Key_R)
    click(canvas, 140, 40)
    assert canvas.doc.current.mask[40, 140] == 255
    assert canvas.doc.current.class_id == 0
    QtTest.QTest.keyClick(canvas, Qt.Key_R, Qt.ShiftModifier)
    click(canvas, 70, 40)
    assert canvas.doc.current.mask[40, 70] == 0
    assert canvas.doc.selected == first_id
    QtTest.QTest.keyClick(canvas, Qt.Key_T)
    click(canvas, 150, 60)
    canvas.accept_sam(canvas.sam_revision, mask())
    canvas.finish()
    assert len(canvas.doc.objects) == 2 and canvas.doc.selected != first_id


def test_tool_labels_are_readable_and_pending_draft_keeps_actual_tool_highlighted(app):
    from studio.labeling import LabelingPage
    from studio.theme import STYLESHEET
    page = LabelingPage()
    page.setStyleSheet(STYLESHEET)
    page.resize(1190, 780)
    page.canvas.set_image(np.zeros((100, 200, 3), np.uint8))
    page.show()
    app.processEvents()
    for button in page.tool_buttons.values():
        for caption in button.findChildren(QtWidgets.QLabel):
            assert caption.width() >= caption.sizeHint().width()
    page.canvas.set_tool('sam')
    page.canvas.sam_points = [(20, 30)]
    page.canvas.sam_labels = [1]
    page.tool_buttons['brush'].click()
    assert page.canvas.tool == 'sam'
    assert [name for name, button in page.tool_buttons.items() if button.isChecked()] == ['sam']
    page.canvas.cancel()
    page.close()


def test_blocked_toolbar_click_keeps_canvas_keyboard_shortcuts_available(app):
    from studio.labeling import LabelingPage
    page = LabelingPage()
    page.canvas.set_image(np.zeros((100, 200, 3), np.uint8))
    page.show()
    page.activateWindow()
    app.processEvents()
    c = page.canvas
    c.set_tool('polygon')
    c.points = [(20, 20), (40, 20), (40, 40)]
    QtTest.QTest.mouseClick(page.tool_buttons['brush'], Qt.LeftButton)
    assert app.focusWidget() is c
    QtTest.QTest.keyClick(app.focusWidget(), Qt.Key_C)
    assert not c.pending
    page.close()


@pytest.mark.parametrize('tool', ['polygon', 'rectangle', 'brush', 'sam'])
def test_releasing_ai_memory_preserves_manual_drawing_drafts(app, tool):
    from studio.labeling import LabelingPage
    page = LabelingPage()
    c = page.canvas
    c.set_image(np.zeros((100, 200, 3), np.uint8))
    c.set_tool(tool)
    if tool == 'polygon':
        c.points = [(20, 20), (40, 20), (40, 40)]
    elif tool == 'rectangle':
        c.rect_start, c.rect_end = (20, 20), (40, 40)
    elif tool == 'brush':
        c.stroke = mask()
    else:
        c.sam_points, c.sam_labels = [(70, 40)], [1]
        c.accept_sam(c.sam_revision, mask())
    page.release_sam()
    assert c.pending
    if tool == 'polygon':
        assert c.points == [(20, 20), (40, 20), (40, 40)]
    elif tool == 'rectangle':
        assert c.rect_start == (20, 20) and c.rect_end == (40, 40)
    elif tool == 'brush':
        assert np.array_equal(c.stroke, mask())
    else:
        assert c.sam_points == [(70, 40)] and np.array_equal(c.sam_preview, mask())
    c.cancel()
    page.close()


def test_mask_tag_sits_on_real_mask_instead_of_empty_bounding_box_corner(canvas):
    # An L shape has an empty upper-left bounding-box corner.
    shape = np.zeros((100, 200), np.uint8)
    shape[10:85, 140:170] = 255
    shape[65:85, 20:170] = 255
    obj = canvas.doc.add_mask(shape)
    obj.class_id = 0
    rect = canvas.mask_label_rect(obj, '1 컵')
    x, y = canvas._clamp(canvas._point(rect.center()))
    assert shape[y, x] == 255
    assert canvas.rect().contains(rect.toAlignedRect())


def test_mask_tag_remains_visible_at_image_border(canvas):
    shape = np.zeros((100, 200), np.uint8)
    shape[92:100, 192:200] = 255
    obj = canvas.doc.add_mask(shape)
    rect = canvas.mask_label_rect(obj, '2 가늘고 긴 물체')
    assert canvas.rect().contains(rect.toAlignedRect())
    right = canvas.offset.x() + 200 * canvas.scale
    bottom = canvas.offset.y() + 100 * canvas.scale
    assert rect.right() <= right + 1 and rect.bottom() <= bottom + 1
