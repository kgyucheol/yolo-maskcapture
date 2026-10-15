import json

import cv2
import numpy as np
import pytest

from studio.document import AnnotationDocument
from studio.project import Project


@pytest.fixture
def project(tmp_path):
    p = Project.create(tmp_path / 'project')
    p.add_class('컵')
    p.add_class('병')
    return p


def image_file(path, value=100):
    cv2.imwrite(str(path), np.full((48, 84, 3), value, np.uint8))
    return path


def test_same_filename_imports_do_not_overwrite(project, tmp_path):
    source = image_file(tmp_path / 'same.png', 100)
    first = project.import_image(source)
    image_file(source, 200)
    second = project.import_image(source)
    assert first != second
    assert cv2.imread(str(project.image_path(first)))[0, 0, 0] == 100
    assert len(Project.open(project.root).images) == 2


def test_annotation_roundtrip_keeps_mask_and_class(project, tmp_path):
    image = project.import_image(image_file(tmp_path / 'source.png'))
    doc = AnnotationDocument((48, 84))
    mask = np.zeros((48, 84), np.uint8)
    mask[5:20, 10:30] = 255
    doc.add_mask(mask)
    doc.assign_class(1)
    project.save_annotation(image, doc.objects, reviewed=True)
    objects, reviewed = Project.open(project.root).load_annotation(image)
    assert reviewed and objects[0].class_id == 1
    assert np.array_equal(objects[0].mask, mask)


def test_unassigned_is_preserved_but_not_reviewed(project, tmp_path):
    image = project.import_image(image_file(tmp_path / 'source.png'))
    doc = AnnotationDocument((48, 84))
    doc.add_mask(np.ones((48, 84), np.uint8))
    project.save_annotation(image, doc.objects, reviewed=True)
    objects, reviewed = project.load_annotation(image)
    assert objects[0].class_id is None and not reviewed


def test_path_escape_and_nonempty_destination_rejected(project, tmp_path):
    with pytest.raises(ValueError):
        project.image_path('../outside.png')
    with pytest.raises(ValueError):
        Project.create(project.root)


def test_corrupt_mask_raises_instead_of_becoming_background(project, tmp_path):
    image = project.import_image(image_file(tmp_path / 'source.png'))
    doc = AnnotationDocument((48, 84))
    doc.add_mask(np.ones((48, 84), np.uint8))
    doc.assign_class(0)
    project.save_annotation(image, doc.objects, True)
    data = json.loads(project.annotation_path(image).read_text())
    (project.root / data['objects'][0]['mask']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='마스크'):
        project.load_annotation(image)
