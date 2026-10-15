import math
import threading
import time
from datetime import datetime, timezone

from .qt import QtCore, QtWidgets, Qt, Signal
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


class CapturePage(QtWidgets.QWidget):
    imagesAdded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.worker = None
        self.policy = CapturePolicy()
        self.displayed_sequence = 0
        self.setFocusPolicy(Qt.StrongFocus)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('데이터 수집', '실시간 영상을 확인하고 필요한 순간을 저장하세요. 원본 해상도를 그대로 보존합니다.'))
        row = QtWidgets.QHBoxLayout()
        self.view = ImageView('ZMQ 영상이 여기에 표시됩니다\n\n192.168.123.164:55555')
        row.addWidget(self.view, 1)
        frame, box = card('스트림 연결')
        frame.setMaximumWidth(330)
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
        box.addLayout(form)
        self.connect_button = button('스트림 연결', self.toggle_connection, True)
        box.addWidget(self.connect_button)
        box.addWidget(label('연결 상태', 'section'))
        self.connection = label('연결되지 않음', 'notice')
        box.addWidget(self.connection)
        box.addWidget(label('자동 촬영', 'section'))
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(0.1, 3600)
        self.interval.setDecimals(1)
        self.interval.setValue(1)
        self.interval.setSuffix(' 초마다 1장')
        self.interval.valueChanged.connect(self.policy.set_interval)
        box.addWidget(self.interval)
        self.auto_button = button('Ctrl+Space  자동 촬영', self.toggle_auto)
        self.auto_button.setCheckable(True)
        box.addWidget(self.auto_button)
        box.addWidget(button('Space  지금 한 장 촬영', self.capture_one, True))
        self.count = label('이번 세션 0장', 'badge')
        box.addWidget(self.count)
        box.addWidget(label('SUB: PUB 송신기에 연결\nPULL: PUSH 송신기에 연결\nREQ: REP 송신기에 요청\n\nJPEG/PNG 바이트 또는\n마지막 파트가 이미지인 multipart', 'subtitle'))
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
        self.timer.start(40)

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
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(1000)
        self.policy.automatic = False
        self.auto_button.setChecked(False)
        self.connection.setText('연결 해제됨')
        self.view.set_frame(None)

    def latest(self):
        if self.worker and self.worker.isRunning():
            return self.worker.latest()
        return None, 0, 0.0

    def toggle_auto(self):
        if not self.project or not self.worker or not self.worker.isRunning():
            self.auto_button.setChecked(False)
            self.notice.setText('프로젝트와 스트림 연결을 먼저 준비하세요.')
            return
        self.policy.automatic = self.auto_button.isChecked()
        self.policy.last_save = None
        self.notice.setText('자동 촬영 중 · 새 프레임만 저장합니다.' if self.policy.automatic else '자동 촬영 중지')
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
