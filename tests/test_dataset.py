import json

import cv2
import numpy as np
import pytest
import yaml

from studio.document import AnnotationDocument
from studio.project import Project
from studio.dataset import build_dataset


def labeled_project(tmp_path):
    project = Project.create(tmp_path / 'project')
    project.add_class('상자')
    for i in range(5):
        image = project.capture(np.full((48, 84, 3), 60 + i, np.uint8))
        doc = AnnotationDocument((48, 84))
        mask = np.zeros((48, 84), np.uint8)
        mask[10:30, 20:60] = 255
        doc.add_mask(mask)
        if i != 4:
            doc.assign_class(0)
        project.save_annotation(image, doc.objects, True)
    return project


def test_snapshot_normalized_labels_and_reproducible_splits(tmp_path):
    project = labeled_project(tmp_path)
    a = build_dataset(project, 0.25, 7)
    b = build_dataset(project, 0.25, 7)
    ra = json.loads((a.parent / 'report.json').read_text())
    rb = json.loads((b.parent / 'report.json').read_text())
    assert ra['splits'] == rb['splits']
    assert len(ra['splits']['train']) == 3
    assert len(ra['splits']['val']) == 1
    assert len(ra['excluded']) == 1
    label = next((a.parent / 'labels/train').glob('*.txt')).read_text().split()
    assert label[0] == '0'
    assert len(label) >= 7 and len(label) % 2 == 1
    assert all(0 <= float(v) <= 1 for v in label[1:])
    data = yaml.safe_load(a.read_text())
    assert data['names'][0] == '상자'
    assert len(project.images) == 5 and a.exists() and b.exists()


@pytest.mark.parametrize('ratio', [0, 1, -0.1, float('nan')])
def test_invalid_split_rejected_without_deleting_existing(tmp_path, ratio):
    p = labeled_project(tmp_path)
    marker = p.root / 'datasets/keep.txt'
    marker.write_text('keep')
    with pytest.raises(ValueError):
        build_dataset(p, ratio)
    assert marker.read_text() == 'keep'


def test_background_is_included_only_when_explicitly_reviewed(tmp_path):
    p = labeled_project(tmp_path)
    image = p.capture(np.zeros((48, 84, 3), np.uint8))
    p.save_annotation(image, [], True)
    result = build_dataset(p)
    report = json.loads((result.parent / 'report.json').read_text())
    assert report['backgrounds'] == 1
    assert image in report['splits']['train'] + report['splits']['val']
