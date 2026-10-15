import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time

import numpy as np
import pytest

from studio.capture import CapturePage
from studio.project import Project
from studio.qt import QtTest, QtWidgets


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(app, tmp_path):
    widget = CapturePage()
    widget.resize(900, 600)
    widget.show()
    widget.set_project(Project.create(tmp_path / 'p'))
    frame = np.zeros((48, 84, 3), np.uint8)
    sequence = iter(range(1, 1000))
    widget.latest = lambda: (frame, next(sequence), time.monotonic())     # a live frame without a ZMQ worker
    yield widget
    widget.close()


def toast(widget):
    return widget.view.toast


def test_saving_shows_the_file_name_then_disappears_after_a_second(page):
    page.capture_one()
    assert toast(page).isVisible()
    assert toast(page).text().startswith('✓ 저장됨 · 이번 세션 1장')
    assert page.project.images[-1]['file'] in toast(page).text()
    assert toast(page).property('error') is False
    QtTest.QTest.qWait(1150)
    assert not toast(page).isVisible()


def test_a_second_save_replaces_the_text_and_restarts_the_timer(page):
    page.capture_one()
    QtTest.QTest.qWait(700)
    page.capture_one()
    assert '이번 세션 2장' in toast(page).text()
    QtTest.QTest.qWait(700)
    assert toast(page).isVisible()          # 1.4 s after the first save, 0.7 s after the second
    QtTest.QTest.qWait(450)
    assert not toast(page).isVisible()


def test_the_toast_sits_inside_the_view_near_the_bottom_center(page):
    page.capture_one()
    box, view = toast(page).geometry(), page.view.rect()
    assert view.contains(box)
    assert abs(box.center().x() - view.center().x()) <= 1
    assert box.bottom() > view.height() * 0.7


def test_a_failed_save_shows_an_error_toast(page):
    page.latest = lambda: (None, 0, 0.0)
    page.capture_one()
    assert toast(page).isVisible() and toast(page).property('error') is True
    assert '새 프레임' in toast(page).text()
    assert page.project.images == []


def test_capturing_without_a_project_explains_why(app):
    widget = CapturePage()
    widget.resize(900, 600)
    widget.show()
    widget.capture_one()
    assert toast(widget).isVisible() and toast(widget).property('error') is True
    assert '프로젝트' in toast(widget).text()
    widget.close()
