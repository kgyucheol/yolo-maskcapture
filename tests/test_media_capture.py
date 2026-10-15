import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio.project import Project, image_bytes
from studio.qt import QtWidgets


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / '동영상.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for value in range(10):
        writer.write(np.full((48, 64, 3), 20 * value, np.uint8))
    writer.release()
    return path


def test_image_source_keeps_original_resolution_and_can_be_captured_later(tmp_path):
    from studio.media import FileMediaSource
    image = np.full((48, 84, 3), 120, np.uint8)
    path = tmp_path / '한글 이미지.png'
    path.write_bytes(image_bytes(image))
    source = FileMediaSource(path)
    frame, sequence, received = source.latest(now=100)
    assert np.array_equal(frame, image) and received == 100
    assert not source.is_video and not source.playing
    assert source.latest(now=200)[1] == sequence
    source.close()
    assert source.latest()[0] is None


def test_video_play_pause_seek_and_eof_are_deterministic(movie):
    from studio.media import FileMediaSource
    source = FileMediaSource(movie)
    assert source.is_video and source.total_frames == 10
    source.play(now=0)
    source.advance(now=.05)
    assert source.frame_index == 0
    source.advance(now=.1)
    assert source.frame_index == 1
    source.pause()
    source.advance(now=100)
    assert source.frame_index == 1
    source.seek(5)
    assert source.frame_index == 5 and not source.playing
    assert np.mean(source.latest()[0]) > 70
    source.play(now=100)
    for step in range(1, 8):
        source.advance(now=100 + step * .11)
    assert not source.playing and source.ended
    assert source.frame_index == 9
    source.close()


def test_media_page_captures_paused_frames_and_clears_on_source_switch(tmp_path, movie):
    from studio.capture import CapturePage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = CapturePage()
    page.set_project(Project.create(tmp_path / 'project'))
    assert page.open_media(movie)
    assert page.source_mode.currentIndex() == 1
    page.capture_one()
    assert len(page.project.images) == 1
    assert page.project.images[0]['width'] == 64
    page.media.pause()
    page.seek_media(4)
    page.capture_one()
    assert len(page.project.images) == 2
    page.source_mode.setCurrentIndex(0)
    assert page.latest()[0] is None
    assert not page.policy.automatic
    page.stop()
    page.close()


def test_open_failure_reports_error_without_throwing_or_retaining_old_frame(tmp_path):
    from studio.capture import CapturePage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = CapturePage()
    assert not page.open_media(tmp_path / '없는파일.mp4')
    assert page.latest()[0] is None
    assert '파일' in page.notice.text()
    page.stop()
    page.close()


def test_slider_keyboard_actions_seek_and_programmatic_refresh_does_not_pause(movie):
    from studio.capture import CapturePage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = CapturePage()
    page.open_media(movie)
    page.seek_slider.triggerAction(QtWidgets.QAbstractSlider.SliderSingleStepAdd)
    assert page.media.frame_index == 1
    page.media.play()
    page.update_playback()
    assert page.media.playing
    page.stop()
    page.close()


def test_replacing_a_still_image_updates_shared_input_prediction_and_provenance(tmp_path):
    from studio.capture import CapturePage
    from studio.prediction import PredictionPage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    capture = CapturePage()
    prediction = PredictionPage(capture)
    submitted = []
    class Worker:
        running = True
        def isRunning(self):
            return self.running
        def submit(self, frame, source):
            submitted.append((frame, source))
        def stop(self):
            self.running = False
    prediction.worker = Worker()
    prediction.loaded = True
    prediction.gate.start()
    for index in range(2):
        path = tmp_path / f'{index}.png'
        path.write_bytes(image_bytes(np.full((48, 84, 3), index * 200, np.uint8)))
        capture.open_media(path)
        prediction.pump()
    assert len(submitted) == 2
    assert submitted[1][1] == str(tmp_path / '1.png')
    assert np.mean(submitted[1][0]) == 200
    prediction.stop()
    capture.stop()
    prediction.close()
    capture.close()
