import threading
import time
from pathlib import Path
from uuid import uuid4

from .inference import SegmentationPredictor, ResultGate
from .project import IMAGE_EXTENSIONS, atomic_bytes, atomic_json, image_bytes, read_image
from .qt import QtCore, QtWidgets, Signal
from .widgets import ImageView, button, card, label, page_header, scroll_panel


class PredictionWorker(QtCore.QThread):
    loaded = Signal(int)
    result = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, generation, config, parent=None):
        super().__init__(parent)
        self.generation, self.config = generation, config
        self.condition = threading.Condition()
        self.pending = None

    def submit(self, frame, source):
        with self.condition:
            self.pending = (frame, source)
            self.condition.notify()

    def stop(self):
        self.requestInterruption()
        with self.condition:
            self.pending = None
            self.condition.notify()

    def run(self):
        try:
            predictor = SegmentationPredictor(**self.config)
            self.loaded.emit(self.generation)
            while not self.isInterruptionRequested():
                with self.condition:
                    if self.pending is None:
                        self.condition.wait(0.1)
                    item, self.pending = self.pending, None
                if item is None or self.isInterruptionRequested():
                    continue
                frame, source = item
                prediction = predictor.predict(frame)
                if not self.isInterruptionRequested():
                    self.result.emit(self.generation, (prediction, source))
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class PredictionPage(QtWidgets.QWidget):
    def __init__(self, capture_page):
        super().__init__()
        self.project = None
        self.capture = capture_page
        self.worker = None
        self.gate = ResultGate()
        self.loaded = False
        self.last_sequence = None
        self.last_result = None
        self.last_source = None
        self.files = []
        self.file_index = 0
        self.requested_source = None
        self.run_config = None
        self.result_config = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('추론 확인', 'PT와 ONNX 세그먼테이션 모델의 마스크, 태그, 신뢰도와 처리 시간을 확인하세요.'))
        row = QtWidgets.QHBoxLayout()
        self.view = ImageView('학습한 .pt 또는 내보낸 .onnx 모델을 선택하세요')
        row.addWidget(self.view, 1)
        panel, box = card('모델과 입력')
        panel.setMaximumWidth(350)
        self.model = QtWidgets.QLineEdit()
        self.model.setPlaceholderText('best.pt 또는 best.onnx')
        box.addWidget(self.model)
        box.addWidget(button('모델 파일 선택', self.browse_model))
        self.source_mode = QtWidgets.QComboBox()
        self.source_mode.addItems(['이미지', '이미지 폴더', 'ZMQ 스트림'])
        box.addWidget(self.source_mode)
        self.source = QtWidgets.QLineEdit()
        self.source.setPlaceholderText('추론할 이미지 또는 폴더')
        box.addWidget(self.source)
        box.addWidget(button('입력 선택', self.browse_source))
        form = QtWidgets.QFormLayout()
        self.conf = QtWidgets.QDoubleSpinBox()
        self.conf.setRange(0.01, 1)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.01, 1)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.45)
        self.imgsz = QtWidgets.QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(864)
        self.device = QtWidgets.QLineEdit('cpu')
        form.addRow('신뢰도', self.conf)
        form.addRow('IoU', self.iou)
        form.addRow('입력 크기', self.imgsz)
        form.addRow('Device', self.device)
        box.addLayout(form)
        box.addWidget(label('고정 크기 ONNX는 내보낼 때 지정한 입력 크기를 사용합니다.', 'subtitle'))
        self.start_button = button('모델 로드 / 추론 시작', self.start, True)
        self.stop_button = button('추론 중지', self.stop)
        self.stop_button.setEnabled(False)
        box.addWidget(self.start_button)
        box.addWidget(self.stop_button)
        navigation = QtWidgets.QHBoxLayout()
        self.previous = button('이전 이미지', lambda: self.navigate(-1))
        self.next = button('다음 이미지', lambda: self.navigate(1))
        navigation.addWidget(self.previous)
        navigation.addWidget(self.next)
        box.addLayout(navigation)
        self.position = label('이미지 미선택', 'subtitle')
        box.addWidget(self.position)
        self.stats = label('객체 0개 · 대기 중', 'badge')
        box.addWidget(self.stats)
        self.objects = QtWidgets.QListWidget()
        box.addWidget(self.objects, 1)
        box.addWidget(button('결과 이미지 + JSON 저장', self.save_result))
        row.addWidget(scroll_panel(panel, 350))
        layout.addLayout(row, 1)
        self.notice = label('Ultralytics 세그먼테이션 PT/ONNX 모델을 지원합니다.', 'notice')
        layout.addWidget(self.notice)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.pump)
        self.timer.start(80)
        self.pending_restart = False
        for field in (self.model, self.source, self.device):
            field.textChanged.connect(self.settings_changed)
        for field in (self.conf, self.iou, self.imgsz):
            field.valueChanged.connect(self.settings_changed)
        self.source_mode.currentIndexChanged.connect(self.settings_changed)

    def settings_changed(self):
        had_result = self.last_result is not None or self.gate.active
        self.stop()
        self.last_result = None
        self.result_config = None
        if had_result:
            self.notice.setText('설정이 변경되었습니다. 추론 시작을 눌러 적용하세요.')

    def set_project(self, project):
        self.project = project
        self.last_result = None
        self.result_config = None
        self.objects.clear()
        self.view.set_frame(None)

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '세그먼테이션 모델', '', 'Models (*.pt *.onnx)')
        if path:
            self.model.setText(path)
            if self.busy():
                self.stop()
                self.notice.setText('모델이 변경되었습니다. 추론 시작을 누르세요.')

    def browse_source(self):
        if self.source_mode.currentIndex() == 1:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, '이미지 폴더')
        elif self.source_mode.currentIndex() == 0:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '이미지', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)')
        else:
            self.notice.setText('수집 화면에서 연결한 ZMQ 스트림을 공유합니다.')
            return
        if path:
            self.source.setText(path)

    def start(self):
        if self.busy():
            self.pending_restart = True
            self.stop(preserve_restart=True)
            self.notice.setText('이전 추론 정리 후 새 설정으로 시작합니다…')
            return
        try:
            path = Path(self.model.text().strip())
            if not path.is_file():
                raise ValueError('모델 파일을 선택하세요.')
            self.files = []
            self.file_index = 0
            self.last_sequence = None
            if self.source_mode.currentIndex() == 2:
                if self.capture.latest()[0] is None:
                    raise ValueError('수집 화면에서 스트림을 연결하고 프레임을 수신하세요.')
            else:
                source = Path(self.source.text().strip())
                if self.source_mode.currentIndex() == 1 and source.is_dir():
                    self.files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
                elif source.is_file():
                    self.files = [source]
                if not self.files:
                    raise ValueError('추론할 이미지를 선택하세요.')
            self.loaded = False
            self.last_result = None
            self.objects.clear()
            generation = self.gate.start()
            config = dict(path=str(path), conf=self.conf.value(), iou=self.iou.value(),
                          device=self.device.text().strip() or 'cpu', imgsz=self.imgsz.value())
            self.run_config = config.copy()
            self.worker = PredictionWorker(generation, config, self)
            self.worker.loaded.connect(self.on_loaded)
            self.worker.result.connect(self.on_result)
            self.worker.failed.connect(self.on_error)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
            self.stop_button.setEnabled(True)
            self.notice.setText('모델을 불러오고 있습니다…')
            self.stats.setText('모델 로드 중')
        except (ValueError, OSError) as exc:
            self.notice.setText(str(exc))

    def on_loaded(self, generation):
        if self.gate.accepts(generation):
            self.loaded = True
            self.notice.setText('모델 준비 완료')
            if self.files:
                self.submit_file()

    def submit_file(self):
        if not self.loaded or not self.files or not self.busy():
            return
        path = self.files[self.file_index]
        image = read_image(path)
        if image is None:
            self.notice.setText('이미지를 읽을 수 없습니다: ' + path.name)
            return
        self.requested_source = str(path)
        self.worker.submit(image, str(path))
        self.position.setText(f'{self.file_index + 1} / {len(self.files)} · {path.name}')

    def navigate(self, direction):
        if self.files:
            self.file_index = max(0, min(len(self.files) - 1, self.file_index + direction))
            self.submit_file()

    def pump(self):
        if not self.loaded or not self.gate.active or self.files or not self.busy():
            return
        frame, sequence, received = self.capture.latest()
        if frame is not None and sequence != self.last_sequence and time.monotonic() - received < 1.5:
            self.last_sequence = sequence
            self.worker.submit(frame, 'zmq')

    def on_result(self, generation, value):
        if not self.gate.accepts(generation):
            return
        result, source = value
        if self.files and source != self.requested_source:
            return
        self.last_result, self.last_source = result, source
        self.result_config = self.run_config.copy() if self.run_config else {}
        self.view.set_frame(result.image)
        self.objects.clear()
        for obj in result.objects:
            self.objects.addItem(f"{obj['name']}   {obj['confidence']:.1%}")
        fps = 1000 / max(0.01, result.elapsed_ms)
        self.stats.setText(f'{len(result.objects)}개 객체 · {result.elapsed_ms:.1f} ms · 처리 {fps:.1f} FPS')
        self.notice.setText('추론 완료 · ' + (Path(source).name if source != 'zmq' else 'ZMQ 실시간 스트림'))

    def on_error(self, generation, error):
        if self.gate.accepts(generation):
            self.gate.stop()
            self.loaded = False
            self.notice.setText('추론 실패: ' + error)
            self.stats.setText('추론 실패')

    def on_finished(self):
        self.stop_button.setEnabled(False)
        if self.pending_restart:
            self.pending_restart = False
            self.start()

    def stop(self, preserve_restart=False):
        if not preserve_restart:
            self.pending_restart = False
        self.gate.stop()
        self.loaded = False
        if self.worker:
            self.worker.stop()
        self.notice.setText('추론 중지 요청됨')

    def busy(self):
        return bool(self.worker and self.worker.isRunning())

    def save_result(self):
        if not self.project or self.last_result is None:
            self.notice.setText('프로젝트와 추론 결과가 필요합니다.')
            return
        stem = 'prediction_' + uuid4().hex[:12]
        image_path = self.project.root / 'predictions' / (stem + '.png')
        try:
            atomic_bytes(image_path, image_bytes(self.last_result.image))
            atomic_json(image_path.with_suffix('.json'), dict(source=self.last_source,
                model=(self.result_config or {}).get('path'), settings=self.result_config,
                elapsed_ms=self.last_result.elapsed_ms, objects=self.last_result.objects))
            self.notice.setText('결과 저장됨 · ' + str(image_path))
        except OSError as exc:
            self.notice.setText('결과 저장 실패: ' + str(exc))
