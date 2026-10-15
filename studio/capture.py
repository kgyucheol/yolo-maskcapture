import math
import threading
import time
from datetime import datetime, timezone

from .qt import QtCore, QtWidgets, Qt, Signal
from .media import FileMediaSource, MEDIA_FILTER, media_time
from .file_dialogs import choose_file
from .widgets import ImageView, button, card, label, page_header, scroll_panel


class CapturePolicy:
    def __init__(self, interval=1.0):
        self.set_interval(interval)
        self.automatic = False
        self.last_save = None
        self.last_sequence = None
        self.saved = 0

    def set_interval(self, interval):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('촬영 간격은 0보다 커야 합니다.')
        self.interval = interval

    def save(self, project, frame, sequence, received, now):
        if frame is None or now - received > 1.5:
            raise ValueError('새 프레임을 수신한 뒤 촬영하세요.')
        image = project.capture(frame, received_at=datetime.now(timezone.utc).isoformat())
        self.last_save, self.last_sequence = now, sequence
        self.saved += 1
        return image

    def maybe_save(self, project, frame, sequence, received, now):
        if not self.automatic or sequence == self.last_sequence or frame is None or now - received > 1.5:
            return None
        if self.last_save is None or now - self.last_save >= self.interval:
            return self.save(project, frame, sequence, received, now)
        return None


class CaptureWorker(QtCore.QThread):
    status = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0
        self._received = 0.0

    def latest(self):
        with self._lock:
            return self._frame, self._sequence, self._received

    def run(self):
        source = None
        try:
            from zmq_source import ZmqImageSource
            source = ZmqImageSource(**self.settings, timeout_ms=200)
            self.status.emit('접속 요청됨 · 첫 프레임 대기 중')
            last_notice = 0.0
            while not self.isInterruptionRequested():
                try:
                    frame = source.read()
                    now = time.monotonic()
                    if frame is not None:
                        with self._lock:
                            self._frame = frame
                            self._sequence += 1
                            self._received = now
                        if now - last_notice > 1:
                            self.status.emit(f'수신 중 · {frame.shape[1]} × {frame.shape[0]}')
                            last_notice = now
                    elif now - last_notice > 3:
                        self.status.emit('프레임 대기 중 · 송신 주소와 방식을 확인하세요')
                        last_notice = now
                except ValueError as exc:
                    self.status.emit(str(exc))
                    self.msleep(50)
        except ImportError:
            self.status.emit('pyzmq가 필요합니다: python -m pip install pyzmq')
        except Exception as exc:
            self.status.emit(f'연결 오류: {exc}')
        finally:
            if source:
                source.close()


INTERVAL_PRESETS = (0.5, 1, 2, 5, 10)


class IntervalSpin(QtWidgets.QDoubleSpinBox):
    """클릭하거나 탭으로 들어오면 값 전체가 선택되어 바로 덮어쓸 수 있는 간격 입력 칸."""

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QtCore.QTimer.singleShot(0, self.selectAll)


class CapturePage(QtWidgets.QWidget):
    imagesAdded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.worker = None
        self.media = None
        self.source_generation = 0
        self.policy = CapturePolicy()
        self.displayed_sequence = 0
        self.setFocusPolicy(Qt.StrongFocus)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('데이터 수집', 'ZMQ 스트림 또는 이미지·동영상 파일을 확인하고 필요한 순간을 원본 해상도로 저장하세요.'))
        row = QtWidgets.QHBoxLayout()
        self.view = ImageView('ZMQ 영상이 여기에 표시됩니다\n\n192.168.123.164:55555')
        row.addWidget(self.view, 1)
        frame, box = card('입력과 촬영')
        frame.setMaximumWidth(330)
        self.source_mode = QtWidgets.QComboBox()
        self.source_mode.addItems(['ZMQ 스트림', '이미지 / 동영상 파일'])
        box.addWidget(self.source_mode)
        self.source_stack = QtWidgets.QStackedWidget()
        stream_controls = QtWidgets.QWidget()
        stream_box = QtWidgets.QVBoxLayout(stream_controls)
        stream_box.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        self.host = QtWidgets.QLineEdit('192.168.123.164')
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(55555)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['sub', 'pull', 'req'])
        self.bind = QtWidgets.QCheckBox('이 PC에서 bind')
        form.addRow('호스트', self.host)
        form.addRow('포트', self.port)
        form.addRow('수신 방식', self.mode)
        form.addRow('', self.bind)
        stream_box.addLayout(form)
        self.connect_button = button('스트림 연결', self.toggle_connection, True)
        stream_box.addWidget(self.connect_button)
        stream_box.addStretch()
        self.source_stack.addWidget(stream_controls)
        file_controls = QtWidgets.QWidget()
        file_box = QtWidgets.QVBoxLayout(file_controls)
        file_box.setContentsMargins(0, 0, 0, 0)
        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText('PNG, JPG, MP4, AVI, MOV…')
        file_box.addWidget(self.file_path)
        file_box.addWidget(button('이미지 / 동영상 파일 열기', self.browse_media, True))
        playback = QtWidgets.QHBoxLayout()
        self.play_button = button('재생', self.toggle_playback)
        self.rewind_button = button('처음으로', lambda: self.seek_media(0))
        playback.addWidget(self.play_button)
        playback.addWidget(self.rewind_button)
        file_box.addLayout(playback)
        self.seek_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.valueChanged.connect(self.seek_media)
        file_box.addWidget(self.seek_slider)
        self.media_position = label('파일을 선택하세요.', 'subtitle')
        file_box.addWidget(self.media_position)
        file_box.addStretch()
        self.source_stack.addWidget(file_controls)
        box.addWidget(self.source_stack)
        self.update_playback()
        self.source_mode.currentIndexChanged.connect(self.switch_source)
        box.addWidget(label('입력 상태', 'section'))
        self.connection = label('연결되지 않음', 'notice')
        box.addWidget(self.connection)
        box.addWidget(label('자동 촬영', 'section'))
        self.interval = IntervalSpin()
        self.interval.setRange(0.1, 3600)
        self.interval.setDecimals(1)
        self.interval.setSingleStep(0.5)
        self.interval.setKeyboardTracking(False)
        self.interval.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.interval.setFixedWidth(96)
        self.interval.setValue(1)
        self.interval.setToolTip('숫자를 입력하고 Enter. 0.1 ~ 3600초')
        self.interval.valueChanged.connect(self.policy.set_interval)
        self.interval.valueChanged.connect(self.show_interval)
        self.interval.lineEdit().returnPressed.connect(self.setFocus)
        self.interval_text = label('', 'subtitle')
        interval_row = QtWidgets.QHBoxLayout()
        interval_row.addWidget(self.interval)
        interval_row.addWidget(self.interval_text, 1)
        box.addLayout(interval_row)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(5)
        self.preset_buttons = {}
        for seconds in INTERVAL_PRESETS:
            chip = button(f'{seconds:g}초', lambda checked=False, value=seconds: self.set_interval_preset(value))
            chip.setObjectName('chip')
            chip.setCheckable(True)
            self.preset_buttons[seconds] = chip
            preset_row.addWidget(chip)
        box.addLayout(preset_row)
        self.show_interval(self.interval.value())
        self.auto_button = button('Ctrl+Space  자동 촬영', self.toggle_auto)
        self.auto_button.setCheckable(True)
        box.addWidget(self.auto_button)
        box.addWidget(button('Space  지금 한 장 촬영', self.capture_one, True))
        self.count = label('이번 세션 0장', 'badge')
        box.addWidget(self.count)
        box.addWidget(label('이미지: 파일을 열고 한 장 촬영\n동영상: 재생·일시정지·위치 이동 후 촬영\n자동 촬영은 재생 중인 새 프레임만 저장\n\nZMQ: JPEG/PNG 또는 multipart 지원', 'subtitle'))
        box.addStretch()
        row.addWidget(scroll_panel(frame, 330))
        layout.addLayout(row, 1)
        self.notice = label('스트림에 연결하면 영상을 바로 볼 수 있습니다. 촬영 파일은 프로젝트를 만든 뒤 PNG로 저장합니다.', 'notice')
        layout.addWidget(self.notice)
        layout.addWidget(label('Space 한 장 촬영   ·   Ctrl+Space 자동 촬영 시작/중지   ·   영상 클릭 후 키보드 사용', 'subtitle'))
        self.view.setFocusPolicy(Qt.NoFocus)
        self.view.mousePressEvent = lambda event: self.setFocus()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

    def switch_source(self, index):
        self.stop()
        self.source_stack.setCurrentIndex(index)
        self.file_path.clear()
        self.displayed_sequence = None
        self.policy.last_sequence = None
        self.policy.last_save = None
        self.connection.setText('파일을 선택하세요.' if index else '연결되지 않음')
        self.update_playback()

    def browse_media(self):
        path = choose_file(self, '이미지 또는 동영상 열기', MEDIA_FILTER, 'media', self.project, self.file_path.text())
        if path:
            self.open_media(path)

    def open_media(self, path):
        self.source_mode.setCurrentIndex(1)
        self.stop()
        self.displayed_sequence = None
        self.policy.last_sequence = None
        try:
            self.media = FileMediaSource(path)
            self.file_path.setText(str(self.media.path))
            self.file_path.setToolTip(str(self.media.path))
            self.view.set_frame(self.media.frame)
            self.connection.setText(f'파일 준비됨 · {self.media.frame.shape[1]} × {self.media.frame.shape[0]}')
            self.notice.setText('동영상은 재생한 뒤 자동 촬영하거나, 일시정지 후 한 장씩 촬영하세요.'
                                if self.media.is_video else '이미지 준비됨 · Space 또는 촬영 버튼으로 프로젝트에 저장하세요.')
            blocked = self.seek_slider.blockSignals(True)
            self.seek_slider.setRange(0, max(0, self.media.total_frames - 1))
            self.seek_slider.setValue(0)
            self.seek_slider.blockSignals(blocked)
            self.update_playback()
            self.setFocus()
            return True
        except (ValueError, OSError) as exc:
            self.media = None
            self.file_path.clear()
            self.notice.setText('파일 열기 실패: ' + str(exc))
            self.update_playback()
            return False

    def toggle_playback(self):
        if self.media and self.media.is_video:
            if self.media.playing:
                self.media.pause()
                self.stop_auto()
            else:
                self.media.play()
            self.update_playback()
            self.setFocus()

    def seek_media(self, index):
        if self.media:
            try:
                self.stop_auto()
                self.media.seek(index)
                self.view.set_frame(self.media.frame)
                self.update_playback()
            except ValueError as exc:
                self.notice.setText(str(exc))

    def update_playback(self):
        video = bool(self.media and self.media.is_video)
        self.play_button.setEnabled(video)
        self.rewind_button.setEnabled(video and self.media.total_frames > 0)
        self.seek_slider.setEnabled(video and self.media.total_frames > 0)
        self.play_button.setText('일시정지' if video and self.media.playing else '재생')
        if self.media:
            if video:
                if not self.seek_slider.isSliderDown():
                    blocked = self.seek_slider.blockSignals(True)
                    self.seek_slider.setValue(self.media.frame_index)
                    self.seek_slider.blockSignals(blocked)
                self.media_position.setText(f'{media_time(self.media.frame_index / self.media.fps)} / '
                    f'{media_time(self.media.total_frames / self.media.fps)} · {self.media.frame_index + 1} 프레임')
            else:
                self.media_position.setText('정지 이미지 · 원본 해상도로 촬영')
        else:
            self.media_position.setText('파일을 선택하세요.')

    def stop_auto(self):
        self.policy.automatic = False
        self.auto_button.setChecked(False)

    def set_project(self, project):
        self.project = project
        self.policy.automatic = False
        self.auto_button.setChecked(False)
        self.policy.saved = 0
        settings = project.data.get('connection', {})
        self.host.setText(settings.get('host', '192.168.123.164'))
        self.port.setValue(settings.get('port', 55555))
        self.mode.setCurrentText(settings.get('mode', 'sub'))
        self.bind.setChecked(settings.get('bind', False))
        self.update_count()

    def settings(self):
        return dict(host=self.host.text().strip(), port=self.port.value(),
                    mode=self.mode.currentText(), bind=self.bind.isChecked())

    def toggle_connection(self):
        if self.worker and self.worker.isRunning():
            self.stop()
            return
        if self.project:
            previous = self.project.data.get('connection')
            try:
                self.project.data['connection'] = self.settings()
                self.project.save()
            except (ValueError, OSError) as exc:
                self.project.data['connection'] = previous
                self.notice.setText(f'연결 설정을 저장하지 못했습니다: {exc}')
                return
        else:
            self.notice.setText('프로젝트 없이 영상만 확인합니다 · 촬영하려면 프로젝트를 만들거나 여세요.')
        self.displayed_sequence = 0
        self.policy.last_sequence = None
        self.worker = CaptureWorker(self.settings(), self)
        self.source_generation += 1
        self.worker.status.connect(self.connection.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.connect_button.setText('연결 해제')
        for widget in (self.host, self.port, self.mode, self.bind):
            widget.setEnabled(False)
        self.setFocus()

    def on_finished(self):
        self.connect_button.setText('스트림 연결')
        self.auto_button.setChecked(False)
        self.policy.automatic = False
        for widget in (self.host, self.port, self.mode, self.bind):
            widget.setEnabled(True)

    def stop(self):
        self.source_generation += 1
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(1000)
        if self.media:
            self.media.close()
            self.media = None
        self.policy.automatic = False
        self.auto_button.setChecked(False)
        self.connection.setText('연결 해제됨')
        self.view.set_frame(None)

    def latest(self):
        if self.source_mode.currentIndex() == 1:
            if self.media:
                frame, sequence, received = self.media.latest()
                return frame, (self.source_generation, sequence), received
            return None, 0, 0.
        if self.worker and self.worker.isRunning():
            frame, sequence, received = self.worker.latest()
            return frame, (self.source_generation, sequence), received
        return None, 0, 0.0

    def source_name(self):
        return str(self.media.path) if self.source_mode.currentIndex() == 1 and self.media else 'zmq'

    def toggle_auto(self):
        ready = bool(self.media and self.media.is_video and self.media.playing) if self.source_mode.currentIndex() == 1 else bool(self.worker and self.worker.isRunning())
        if not self.project or not ready:
            self.auto_button.setChecked(False)
            self.notice.setText('프로젝트를 열고 스트림 연결 또는 동영상 재생을 먼저 준비하세요.')
            return
        self.policy.automatic = self.auto_button.isChecked()
        self.policy.last_save = None
        self.notice.setText('자동 촬영 중 · 새 프레임만 저장합니다.' if self.policy.automatic else '자동 촬영 중지')
        self.setFocus()

    def show_interval(self, seconds):
        self.interval_text.setText(f'초마다 1장 (분당 {60 / seconds:.0f}장)')
        for preset, chip in self.preset_buttons.items():
            chip.setChecked(abs(preset - seconds) < 1e-9)

    def set_interval_preset(self, seconds):
        self.interval.setValue(seconds)
        self.show_interval(self.interval.value())      # re-check the chip even when the value did not change
        self.setFocus()

    def saved_text(self, image):
        return f'✓ 저장됨 · 이번 세션 {self.policy.saved}장\n{image}'

    def update_count(self):
        total = len(self.project.images) if self.project else 0
        self.count.setText(f'이번 세션 {self.policy.saved}장   ·   전체 {total}장')

    def capture_one(self):
        if not self.project:
            self.notice.setText('촬영하려면 프로젝트를 만들거나 여세요.')
            self.view.flash('촬영하려면 프로젝트를 만들거나 여세요.', error=True)
            return
        frame, sequence, received = self.latest()
        try:
            image = self.policy.save(self.project, frame, sequence, received, time.monotonic())
            self.notice.setText(f'저장됨 · {image}')
            self.view.flash(self.saved_text(image))
            self.update_count()
            self.imagesAdded.emit()
        except (ValueError, OSError) as exc:
            self.notice.setText(str(exc))
            self.view.flash(str(exc), error=True)
        self.setFocus()

    def tick(self):
        if self.media:
            playing = self.media.playing
            self.media.advance()
            if playing and self.media.ended:
                self.stop_auto()
                self.notice.setText('동영상 재생 완료 · 자동 촬영을 중지했습니다.')
            self.update_playback()
        frame, sequence, received = self.latest()
        if frame is None:
            return
        if sequence != self.displayed_sequence:
            self.view.set_frame(frame)
            self.displayed_sequence = sequence
        try:
            image = self.project and self.policy.maybe_save(self.project, frame, sequence, received, time.monotonic())
            if image:
                self.view.flash(self.saved_text(image))
                self.update_count()
                self.imagesAdded.emit()
        except (ValueError, OSError) as exc:
            self.policy.automatic = False
            self.auto_button.setChecked(False)
            self.notice.setText(f'자동 촬영 중지: {exc}')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if event.modifiers() & Qt.ControlModifier:
                self.auto_button.setChecked(not self.auto_button.isChecked())
                self.toggle_auto()
            else:
                self.capture_one()
        else:
            super().keyPressEvent(event)
