"""Incremental process output and epoch-based training time estimates."""
import codecs
import math
import re
import time
from collections import deque


class TrainingLog:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.line = ''
        self.carriage = False

    @staticmethod
    def clean(text):
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
        text = re.sub(r'\x1b\][^\x07]*(?:\x07|\x1b\\)', '', text)
        # Incomplete terminal escapes remain buffered until the next read.
        text = re.sub(r'\x1b[^\x1b]*$', '', text)
        return ''.join(char for char in text if char == '\t' or ord(char) >= 32)

    def feed(self, data, final=False):
        completed = []
        for char in self.decoder.decode(data, final=final):
            if char == '\r':
                self.carriage = True
            elif char == '\n':
                completed.append(self.clean(self.line))
                self.line, self.carriage = '', False
            else:
                if self.carriage:
                    self.line, self.carriage = '', False
                self.line += char
        if final and self.line:
            completed.append(self.clean(self.line))
            self.line = ''
        return completed, self.clean(self.line)


class TrainingETA:
    def __init__(self):
        self.active = False
        self.durations = deque(maxlen=5)
        self.epoch = self.epochs = 0
        self.last_end = None

    def start(self, epochs, now=None):
        self.active = True
        self.epoch, self.epochs = 0, epochs
        self.durations.clear()
        self.last_end = time.monotonic() if now is None else now

    def update(self, epoch, epochs, seconds=None, now=None):
        if not self.active or not 0 < epoch <= epochs or epoch <= self.epoch:
            return
        now = time.monotonic() if now is None else now
        duration = seconds if seconds is not None else (now - self.last_end) / (epoch - self.epoch)
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
            self.durations.append(duration)
        self.epoch, self.epochs, self.last_end = epoch, epochs, now

    def remaining(self, now=None):
        if not self.active or not self.durations:
            return None
        now = time.monotonic() if now is None else now
        return max(0., sum(self.durations) / len(self.durations) * (self.epochs - self.epoch)
                   - max(0, now - self.last_end))

    def finish(self):
        self.active = False


def duration_text(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}시간 {minutes:02d}분 {seconds:02d}초' if hours else f'{minutes:02d}분 {seconds:02d}초'
