import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from studio.augmentation import AugmentConfig
from studio.document import MaskObject
from studio.project import Project, read_image
from studio.qt import QtWidgets, QtTest, Qt, qimage


@pytest.fixture(scope='session')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mouse_enable_and_total_multiplier_are_clear_and_clickable(app):
    from studio.augmentation_ui import AugmentationControls
    from studio.theme import STYLESHEET
    controls = AugmentationControls()
    controls.setStyleSheet(STYLESHEET)
    controls.resize(440, 700)
    controls.show()
    app.processEvents()
    assert not controls.isChecked()
    QtTest.QTest.mouseClick(controls.enable_box, Qt.LeftButton)
    assert controls.isChecked()
    controls.multiplier.setCurrentIndex(controls.multiplier.findData(4))
    assert controls.config().copies == 3
    QtTest.QTest.mouseClick(controls.enable_box, Qt.LeftButton)
    assert not controls.isChecked() and controls.config().copies == 0
    controls.close()


def test_exact_single_combination_ratio_and_zero_weight_exclusion():
    from studio.augmentation import augmentation_plan
    config = AugmentConfig(copies=3, horizontal_flip=True, brightness=.2, noise=3,
        mode='random', single_ratio=.25,
        method_weights=(('horizontal_flip', 0), ('brightness', 1), ('noise', 1)))
    plan = augmentation_plan(config, 8, 42)
    assert plan == augmentation_plan(config, 8, 42)
    assert sum(len(methods) == 1 for methods in plan) == 2
    assert sum(len(methods) >= 2 for methods in plan) == 6
    assert all(set(methods) <= {'brightness', 'noise'} for methods in plan)


def test_relative_selection_weights_change_method_frequency():
    from studio.augmentation import augmentation_plan
    config = AugmentConfig(copies=1, horizontal_flip=False, brightness=.2, noise=3,
        mode='single', method_weights=(('brightness', 9), ('noise', 1)))
    counts = Counter(method for methods in augmentation_plan(config, 1000, 42) for method in methods)
    assert 800 < counts['brightness'] < 980
    assert 20 < counts['noise'] < 200


def test_one_method_falls_back_to_single_without_failing():
    from studio.augmentation import augmentation_plan
    config = AugmentConfig(copies=1, mode='random', single_ratio=0)
    assert augmentation_plan(config, 3, 42) == [('horizontal_flip',)] * 3


def test_preview_matches_generated_variant_and_global_recipe_counts(app, tmp_path):
    from studio.training import DatasetPage
    project = Project.create(tmp_path / 'p')
    project.add_class('상자')
    mask = np.zeros((60, 100), np.uint8)
    mask[10:30, 10:30] = 255
    for index in range(5):
        frame = np.full((60, 100, 3), 40 + 20 * index, np.uint8)
        frame[10:30, 10:30] = (30, 80, 200)
        image = project.capture(frame)
        project.save_annotation(image, [MaskObject(mask, class_id=0)], True)
    page = DatasetPage()
    page.set_project(project)
    page.augment.setChecked(True)
    page.augment.multiplier.setCurrentIndex(page.augment.multiplier.findData(3))
    page.augment.single_share.setValue(25)
    page.show_masks.setChecked(False)
    page.refresh_preview()
    preview = page.augmented_view.frame.copy()
    paths = []
    page.datasetReady.connect(paths.append)
    try:
        page.create_dataset()
        for _ in range(400):
            app.processEvents()
            if paths:
                break
            QtTest.QTest.qWait(10)
        assert paths, page.notice.text()
        root = Path(paths[0]).parent
        report = json.loads((root / 'report.json').read_text())
        samples = report['augmentation']['samples']
        assert len(samples) == 8
        assert sum(len(sample['methods']) == 1 for sample in samples) == 2
        assert sum(len(sample['methods']) > 1 for sample in samples) == 6
        assert report['augmentation']['multiplier'] == 3
        assert len(report['splits']['train']) == 12 and len(report['splits']['val']) == 1
        image = read_image(root / 'images/train' / samples[0]['image'])
        assert preview == qimage(image)
        assert all(sample['source'] in report['source_splits']['train'] for sample in samples)
    finally:
        if page.worker:
            assert page.worker.wait(4000)
        page.close()
