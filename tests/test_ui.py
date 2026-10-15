import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from studio.qt import QtCore, QtWidgets, QtTest
from studio.canvas import MaskCanvas
from studio.labeling import LabelingPage
from studio.project import Project


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_polygon_confirm_then_number_assigns_and_undo(app):
    canvas = MaskCanvas()
    canvas.resize(840, 480)
    canvas.set_image(np.zeros((48, 84, 3), np.uint8))
    canvas.class_names = ['컵', '병']
    canvas.show()
    canvas.setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key_W)
    for x, y in [(200, 150), (450, 150), (450, 350), (200, 350)]:
        QtTest.QTest.mouseClick(canvas, QtCore.Qt.LeftButton, pos=QtCore.QPoint(x, y))
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key_F)
    assert len(canvas.doc.objects) == 1
    assert canvas.doc.current.class_id is None
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key_2)
    assert canvas.doc.current.class_id == 1
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key_Z)
    assert canvas.doc.current.class_id is None
    canvas.close()


def test_text_input_numbers_do_not_change_object_and_navigation_saves(app, tmp_path):
    p = Project.create(tmp_path / 'project')
    p.add_class('컵')
    p.add_class('병')
    first = p.capture(np.zeros((48, 84, 3), np.uint8))
    p.capture(np.zeros((48, 84, 3), np.uint8))
    page = LabelingPage()
    page.set_project(p)
    page.canvas.doc.add_mask(np.ones((48, 84), np.uint8))
    page.canvas.doc.assign_class(0)
    page.show()
    page.class_input.setFocus()
    app.processEvents()
    QtTest.QTest.keyClicks(page.class_input, '2wasd')
    assert page.class_input.text() == '2wasd'
    assert page.canvas.doc.current.class_id == 0
    page.navigate(1)
    objects, reviewed = p.load_annotation(first)
    assert len(objects) == 1 and reviewed
    page.close()


def test_incomplete_polygon_blocks_navigation(app, tmp_path):
    p = Project.create(tmp_path / 'project')
    p.capture(np.zeros((48, 84, 3), np.uint8))
    p.capture(np.zeros((48, 84, 3), np.uint8))
    page = LabelingPage()
    page.set_project(p)
    page.canvas.points = [(10, 10), (20, 10)]
    page.navigate(1)
    assert page.index == 0
    page.close()


def test_project_switch_clears_canvas_even_if_new_annotation_is_corrupt(app, tmp_path):
    a = Project.create(tmp_path / 'a')
    a.add_class('컵')
    a.capture(np.zeros((48, 84, 3), np.uint8))
    b = Project.create(tmp_path / 'b')
    b.add_class('컵')
    target = b.capture(np.zeros((48, 84, 3), np.uint8))
    b.annotation_path(target).write_text('{broken')
    page = LabelingPage()
    page.set_project(a)
    page.canvas.doc.add_mask(np.ones((48, 84), np.uint8))
    page.canvas.doc.assign_class(0)
    page.save_current()
    page.set_project(b)
    assert page.canvas.doc is None
    assert page.save_current()
    assert b.annotation_path(target).read_text() == '{broken'
    page.set_project(Project.create(tmp_path / 'empty'))
    assert page.canvas.doc is None and page.canvas.image is None
    page.close()
