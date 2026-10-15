import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import numpy as np

from studio.document import MaskObject
from studio.project import Project
from studio.qt import QtWidgets, QtTest
from studio.training import DatasetPage


def test_dataset_page_previews_masks_and_generates_selected_training_augmentation(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project = Project.create(tmp_path / 'p')
    project.add_class('상자')
    frame = np.zeros((60, 100, 3), np.uint8)
    frame[10:30, 10:30] = (0, 0, 255)
    mask = np.zeros((60, 100), np.uint8)
    mask[10:30, 10:30] = 255
    for _ in range(4):
        image = project.capture(frame)
        project.save_annotation(image, [MaskObject(mask, class_id=0)], True)
    before = {p: p.read_bytes() for p in (project.root / 'images').iterdir()}
    page = DatasetPage()
    page.set_project(project)
    page.augment.setChecked(True)
    page.augment.multiplier.setCurrentIndex(page.augment.multiplier.findData(3))
    page.refresh_preview()
    assert page.original_view.frame is not None and page.augmented_view.frame is not None
    paths = []
    page.datasetReady.connect(paths.append)
    try:
        page.create_dataset()
        for _ in range(300):
            app.processEvents()
            if paths:
                break
            QtTest.QTest.qWait(10)
        assert paths
        from pathlib import Path
        report = json.loads((Path(paths[0]).parent / 'report.json').read_text())
        assert report['augmentation']['generated'] == 6
        assert len(report['splits']['train']) == 9 and len(report['splits']['val']) == 1
        assert all(p.read_bytes() == original for p, original in before.items())
        assert '6' in page.report.toPlainText()
    finally:
        if page.worker:
            assert page.worker.wait(3000)
        page.close()


def test_method_checkboxes_and_combination_mode_control_preview_and_saved_recipe(tmp_path):
    from studio.augmentation_ui import AugmentationControls
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    controls = AugmentationControls()
    controls.setChecked(True)
    for check in controls.method_checks.values():
        check.setChecked(False)
    controls.mode.setCurrentIndex(controls.mode.findData('single'))
    config = controls.config()
    assert config.mode == 'single' and config.horizontal_flip and config.brightness == 0
    assert config.noise == 0 and not controls.fields['brightness'].isEnabled()
    controls.method_checks['noise'].setChecked(True)
    controls.fields['noise'].setValue(5)
    controls.mode.setCurrentIndex(controls.mode.findData('random'))
    controls.max_transforms.setValue(2)
    config = controls.config()
    assert config.noise == 5 and config.max_transforms == 2 and config.mode == 'random'
    assert controls.fields['noise'].isEnabled()
    controls.close()
