import numpy as np
import pytest

from studio.capture import CapturePolicy
from studio.project import Project


def test_auto_capture_saves_only_new_frames_and_obeys_interval(tmp_path):
    p = Project.create(tmp_path / 'p')
    policy = CapturePolicy(interval=2)
    policy.automatic = True
    frame = np.zeros((48, 84, 3), np.uint8)
    assert policy.maybe_save(p, frame, sequence=1, received=1.0, now=1.0)
    assert not policy.maybe_save(p, frame, sequence=1, received=1.0, now=4.0)
    assert not policy.maybe_save(p, frame, sequence=2, received=2.0, now=2.0)
    assert policy.maybe_save(p, frame, sequence=3, received=4.0, now=4.0)
    assert policy.saved == 2 and len(p.images) == 2


def test_capture_failure_does_not_increment_counter(tmp_path, monkeypatch):
    p = Project.create(tmp_path / 'p')
    policy = CapturePolicy()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(p, 'capture', fail)
    with pytest.raises(OSError):
        policy.save(p, np.zeros((4, 4, 3), np.uint8), 1, 1.0, 1.0)
    assert policy.saved == 0


@pytest.mark.parametrize('interval', [0, -1, float('nan')])
def test_invalid_capture_interval(interval):
    with pytest.raises(ValueError):
        CapturePolicy(interval)

