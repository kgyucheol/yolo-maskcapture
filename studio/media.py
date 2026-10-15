"""Local image/video playback through the same frame interface as ZMQ."""
import math
import time
from pathlib import Path

import cv2

from .project import IMAGE_EXTENSIONS, read_image

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}
MEDIA_FILTER = ('이미지 및 동영상 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff '
                '*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.mpg *.mpeg)')


class FileMediaSource:
    def __init__(self, path):
        self.path = Path(path)
        self.capture = None
        self.frame = None
        self.sequence = 1
        self.frame_index = 0
        self.total_frames = 1
        self.fps = 1.
        self.playing = self.ended = False
        self.next_frame_at = 0.
        self.is_video = self.path.suffix.lower() in VIDEO_EXTENSIONS
        try:
            if not self.path.is_file():
                raise ValueError('파일을 찾을 수 없습니다.')
            if self.is_video:
                self.capture = cv2.VideoCapture(str(self.path))
                if not self.capture.isOpened():
                    raise ValueError('동영상을 열 수 없습니다. 파일과 코덱을 확인하세요.')
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                self.fps = fps if math.isfinite(fps) and fps > 0 else 30.
                count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                self.total_frames = max(0, int(count)) if math.isfinite(count) else 0
                ok, self.frame = self.capture.read()
                if not ok:
                    raise ValueError('동영상의 첫 프레임을 읽지 못했습니다.')
            elif self.path.suffix.lower() in IMAGE_EXTENSIONS:
                self.frame = read_image(self.path)
                if self.frame is None:
                    raise ValueError('이미지 파일을 읽지 못했습니다.')
            else:
                raise ValueError('지원하는 이미지 또는 동영상 파일을 선택하세요.')
        except Exception:
            self.close()
            raise

    def latest(self, now=None):
        return self.frame, self.sequence, time.monotonic() if now is None else now

    def play(self, now=None):
        if not self.is_video or self.capture is None:
            return
        if self.ended:
            self.seek(0)
        self.playing = True
        self.next_frame_at = (time.monotonic() if now is None else now) + 1 / self.fps

    def pause(self):
        self.playing = False

    def advance(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.playing or now < self.next_frame_at:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.playing, self.ended = False, True
            return
        self.frame = frame
        self.frame_index += 1
        self.sequence += 1
        self.next_frame_at = max(self.next_frame_at + 1 / self.fps, now)

    def seek(self, index):
        self.pause()
        if not self.is_video or self.capture is None or self.total_frames <= 0:
            return
        target = max(0, min(self.total_frames - 1, int(index)))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise ValueError('이 동영상은 재생 위치 이동을 지원하지 않습니다.')
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError('지정한 위치의 프레임을 읽지 못했습니다.')
        self.frame, self.frame_index = frame, target
        self.sequence += 1
        self.ended = False

    def close(self):
        self.playing = False
        self.frame = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def media_time(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'
