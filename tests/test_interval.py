import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from studio.capture import CapturePage, INTERVAL_PRESETS
from studio.qt import Qt, QtTest, QtWidgets


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(app):
    widget = CapturePage()
    widget.resize(900, 600)
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    app.processEvents()
    yield widget
    widget.close()


def checked(widget):
    return [seconds for seconds, chip in widget.preset_buttons.items() if chip.isChecked()]


def test_default_interval_is_one_second_and_its_preset_is_highlighted(page):
    assert page.policy.interval == 1 and checked(page) == [1]
    assert page.interval_text.text() == '초마다 1장 (분당 60장)'
    assert page.interval.suffix() == ''           # the unit is a separate label, so the box holds only the number


@pytest.mark.parametrize('seconds', INTERVAL_PRESETS)
def test_a_preset_button_sets_the_interval_and_returns_focus_to_the_page(page, seconds):
    page.interval.setFocus()
    page.preset_buttons[seconds].click()
    assert page.policy.interval == seconds and page.interval.value() == seconds
    assert checked(page) == [seconds]
    assert page.hasFocus()


def test_clicking_the_already_selected_preset_keeps_it_selected(page):
    page.preset_buttons[1].click()
    assert checked(page) == [1]


def test_a_typed_value_replaces_the_old_one_and_enter_gives_the_page_focus_back(page):
    page.interval.setFocus()
    QtTest.QTest.qWait(30)                         # focus-in selects the whole value
    line = page.interval.lineEdit()
    assert line.selectedText() == line.text()
    QtTest.QTest.keyClicks(line, '2.5')
    QtTest.QTest.keyClick(line, Qt.Key_Return)
    assert page.policy.interval == 2.5
    assert checked(page) == [] and page.interval_text.text() == '초마다 1장 (분당 24장)'
    assert page.hasFocus()


def test_typing_does_not_apply_partial_values_until_enter(page):
    page.interval.setFocus()
    QtTest.QTest.qWait(30)
    QtTest.QTest.keyClicks(page.interval.lineEdit(), '0.5')
    assert page.policy.interval == 1               # still the old value while typing
    QtTest.QTest.keyClick(page.interval.lineEdit(), Qt.Key_Return)
    assert page.policy.interval == 0.5
