import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import re

from studio.qt import QtWidgets, QtTest, Qt


def test_all_numeric_forms_display_only_numbers():
    from studio.window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.dataset.augment.setChecked(True)
    fields = window.findChildren(QtWidgets.QAbstractSpinBox)
    assert fields
    for field in fields:
        assert re.fullmatch(r'-?\d+(?:[.,]\d+)?', field.lineEdit().text()), field.lineEdit().text()
    combo = window.dataset.augment.multiplier
    assert all(combo.itemText(i).isdigit() for i in range(combo.count()))
    window.close()
    app.processEvents()


def test_numeric_editing_keeps_units_outside_and_preserves_configuration():
    from studio.augmentation_ui import AugmentationControls
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    controls = AugmentationControls()
    controls.setChecked(True)
    controls.show()
    app.processEvents()
    for name in ('rotation', 'scale', 'brightness', 'hue'):
        field = controls.fields[name]
        field.setFocus()
        field.selectAll()
        QtTest.QTest.keyClicks(field, '12.5')
        QtTest.QTest.keyClick(field, Qt.Key_Return)
        assert field.value() == 12.5
        assert field.lineEdit().text() == '12.5'
    field = controls.single_share
    field.setFocus()
    field.selectAll()
    QtTest.QTest.keyClicks(field, '70abc')
    QtTest.QTest.keyClick(field, Qt.Key_Return)
    assert field.value() == 70 and field.lineEdit().text() == '70'
    config = controls.config()
    assert config.single_ratio == .7 and config.rotation == 12.5
    assert config.scale == .125 and config.brightness == .125 and config.hue == 12.5
    units = [item.text() for item in controls.findChildren(QtWidgets.QLabel, 'numericUnit')]
    assert '%' in units and '°' in units and 'HSV' in units and '개' in units
    controls.close()
