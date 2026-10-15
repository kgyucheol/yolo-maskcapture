import numpy as np
import pytest

from studio.document import AnnotationDocument


def test_number_assigns_selected_object_and_undo_restores_unassigned():
    doc = AnnotationDocument((48, 84))
    mask = np.zeros((48, 84), np.uint8)
    mask[10:30, 20:50] = 255
    doc.add_mask(mask)
    assert doc.objects[0].class_id is None
    doc.assign_class(1)
    assert doc.objects[0].class_id == 1
    doc.undo()
    assert doc.objects[0].class_id is None
    doc.redo()
    assert doc.objects[0].class_id == 1


def test_brush_replacement_and_delete_are_undoable():
    doc = AnnotationDocument((12, 12))
    doc.add_mask(np.ones((12, 12), np.uint8) * 255)
    changed = doc.objects[0].mask.copy()
    changed[:6] = 0
    doc.replace_mask(changed)
    doc.undo()
    assert np.count_nonzero(doc.objects[0].mask) == 144
    doc.delete_selected()
    assert not doc.objects
    doc.undo()
    assert len(doc.objects) == 1


def test_invalid_mask_is_not_added():
    doc = AnnotationDocument((12, 12))
    with pytest.raises(ValueError):
        doc.add_mask(np.zeros((12, 12), np.uint8))
    with pytest.raises(ValueError):
        doc.add_mask(np.ones((6, 6), np.uint8))
    assert not doc.objects
