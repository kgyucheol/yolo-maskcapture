import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .dataset import build_dataset
from .jobs import validate_train_config
from .project import Project, atomic_json
from .qt import QtCore, QtGui, QtWidgets, Qt, Signal
from .widgets import button, card, label, page_header, scroll_panel

REPO_ROOT = Path(__file__).resolve().parent.parent


class DatasetWorker(QtCore.QThread):
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, project, ratio, seed, parent=None):
        super().__init__(parent)
        self.project = Project(project.root, copy.deepcopy(project.data))
        self.ratio, self.seed = ratio, seed

    def run(self):
        try:
            self.ready.emit(str(build_dataset(self.project, self.ratio, self.seed)))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetPage(QtWidgets.QWidget):
    datasetReady = Signal(str)

    def __init__(self):
        super().__init__()
        self.project = None
        self.worker = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('학습 데이터셋', '작업 완료 이미지를 검증하고, 원본을 보존하는 새 데이터셋을 만듭니다.'))
        top, box = card('데이터 현황')
        self.summary = label('프로젝트를 열어 주세요.', 'badge')
        box.addWidget(self.summary)
        box.addWidget(label('태그 미지정·미작업 이미지는 제외됩니다. 빈 이미지는 라벨링 화면에서 배경으로 확인해야 포함됩니다.', 'subtitle'))
        layout.addWidget(top)
        panel, box = card('생성 설정')
        form = QtWidgets.QFormLayout()
        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setRange(0.01, 0.99)
        self.ratio.setSingleStep(0.05)
        self.ratio.setValue(0.2)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        form.addRow('검증 데이터 비율', self.ratio)
        form.addRow('분할 seed', self.seed)
        box.addLayout(form)
        self.create_button = button('검증하고 데이터셋 생성', self.create_dataset, True)
        box.addWidget(self.create_button)
        layout.addWidget(panel)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText('생성 결과, 제외된 이미지, 마스크 변환 보고서가 여기에 표시됩니다.')
        layout.addWidget(self.report, 1)
        self.notice = label('학습·검증 이미지가 각각 최소 1장 필요합니다.', 'notice')
        layout.addWidget(self.notice)

    def set_project(self, project):
        self.project = project
        self.report.clear()
        self.refresh()

    def refresh(self):
        if not self.project:
            return
        completed, background, pending = 0, 0, 0
        for entry in self.project.images:
            path = self.project.annotation_path(entry['file'])
            try:
                data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if data.get('reviewed'):
                    completed += 1
                    background += not data.get('objects')
                else:
                    pending += 1
            except (OSError, ValueError):
                pending += 1
        self.summary.setText(f'전체 {len(self.project.images)}장   ·   완료 {completed}장 (배경 {background}장)   ·   미완료 {pending}장   ·   클래스 {len(self.project.classes)}개')

    def create_dataset(self):
        if not self.project or self.busy():
            return
        self.worker = DatasetWorker(self.project, self.ratio.value(), self.seed.value(), self)
        self.worker.ready.connect(self.on_ready)
        self.worker.failed.connect(lambda text: self.notice.setText('생성 실패: ' + text))
        self.worker.finished.connect(lambda: self.create_button.setEnabled(True))
        self.create_button.setEnabled(False)
        self.notice.setText('이미지와 라벨을 검사하고 있습니다…')
        self.worker.start()

    def on_ready(self, path):
        data = json.loads((Path(path).parent / 'report.json').read_text(encoding='utf-8'))
        lines = [f'저장 위치: {path}', '', f"학습 {len(data['splits']['train'])}장 / 검증 {len(data['splits']['val'])}장",
                 f"배경 {data['backgrounds']}장 / 제외 {len(data['excluded'])}장", '', '제외된 이미지']
        lines.extend(f"• {entry['image']} — {entry['reason']}" for entry in data['excluded'])
        lines.extend(['', '마스크 변환 안내: 분리 영역은 개별 인스턴스, 내부 구멍은 채워진 폴리곤으로 내보냅니다.'])
        lines.extend(f"• {w['image']} — 영역 {w['components']}개 / 채운 구멍 {w['filled_holes']}개" for w in data['warnings'])
        self.report.setPlainText('\n'.join(lines))
        self.notice.setText('데이터셋 생성 완료 · 학습 화면에 연결했습니다.')
        self.datasetReady.emit(path)

    def busy(self):
        return bool(self.worker and self.worker.isRunning())


class TrainingPage(QtWidgets.QWidget):
    artifactReady = Signal(str)

    def __init__(self):
        super().__init__()
        self.project = None
        self.process = None
        self.job_dir = None
        self.best_model = None
        self.stopping = False
        self.output_buffer = ''
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('세그먼테이션 학습', 'YOLO11s-seg로 학습하고 best.pt 또는 ONNX를 추론 화면에서 확인하세요.'))
        row = QtWidgets.QHBoxLayout()
        self.panel, box = card('학습 설정')
        self.panel.setMaximumWidth(390)
        form = QtWidgets.QFormLayout()
        self.model = QtWidgets.QLineEdit('yolo11s-seg.pt')
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model)
        model_row.addWidget(button('파일', self.browse_model))
        form.addRow('초기 모델', model_row)
        self.data = QtWidgets.QLineEdit()
        self.data.setPlaceholderText('데이터셋 생성 후 자동 입력')
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(self.data)
        data_row.addWidget(button('파일', self.browse_data))
        form.addRow('data.yaml', data_row)
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(100)
        self.imgsz = QtWidgets.QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(864)
        self.batch = QtWidgets.QSpinBox()
        self.batch.setRange(-1, 1024)
        self.batch.setValue(4)
        self.batch.setToolTip('-1: GPU 메모리에 맞춘 자동 batch')
        self.device = QtWidgets.QLineEdit('cpu')
        self.device.setPlaceholderText('cpu 또는 GPU 번호 0')
        self.rect = QtWidgets.QCheckBox('직사각형 학습 (rect)')
        form.addRow('Epoch', self.epochs)
        form.addRow('학습 크기 imgsz', self.imgsz)
        form.addRow('Batch', self.batch)
        form.addRow('Device', self.device)
        form.addRow('', self.rect)
        box.addLayout(form)
        box.addWidget(label('848×480 원본을 그대로 사용하세요.\nimgsz=864는 비율을 유지하며 패딩합니다.\nGPU 학습은 device에 0을 입력하세요.', 'subtitle'))
        self.start_button = button('학습 시작', self.start_training, True)
        self.export_button = button('PT → ONNX 내보내기', self.start_export)
        box.addWidget(self.start_button)
        box.addWidget(self.export_button)
        box.addWidget(label('초기 모델이 로컬에 없으면 Ultralytics가 다운로드합니다. 학습에는 requirements.txt 패키지가 필요합니다.', 'subtitle'))
        box.addStretch()
        row.addWidget(scroll_panel(self.panel, 390))
        result, box = card('진행 상황')
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        box.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2500)
        self.log.setPlaceholderText('학습 로그와 마스크 지표가 여기에 표시됩니다.')
        box.addWidget(self.log, 1)
        self.stop_button = button('현재 작업 중지', self.stop)
        self.stop_button.setObjectName('danger')
        self.stop_button.setEnabled(False)
        box.addWidget(self.stop_button)
        row.addWidget(result, 1)
        layout.addLayout(row, 1)
        self.notice = label('준비됨 · 데이터셋 생성 후 학습을 시작하세요.', 'notice')
        self.notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.notice)

    def set_project(self, project):
        self.project = project
        datasets = sorted((project.root / 'datasets').glob('*/data.yaml'))
        self.data.setText(str(datasets[-1]) if datasets else '')
        self.best_model = None
        results = sorted((project.root / 'runs').glob('*/result.json'))
        for result in results:
            try:
                manifest = json.loads(result.read_text(encoding='utf-8'))
                if manifest.get('best') and Path(manifest['best']).is_file():
                    self.best_model = manifest['best']
            except (OSError, ValueError):
                pass

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '초기 모델', '', 'PyTorch (*.pt)')
        if path:
            self.model.setText(path)

    def browse_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '데이터셋 YAML', '', 'YAML (*.yaml *.yml)')
        if path:
            self.data.setText(path)

    def start_training(self):
        self.launch(dict(mode='train', model=self.model.text().strip(), data=self.data.text().strip(),
                         epochs=self.epochs.value(), batch=self.batch.value(), imgsz=self.imgsz.value(),
                         device=self.device.text().strip(), rect=self.rect.isChecked()))

    def start_export(self):
        model = self.best_model
        if not model:
            model, _ = QtWidgets.QFileDialog.getOpenFileName(self, '내보낼 세그먼테이션 PT', '', 'PyTorch (*.pt)')
        if model:
            self.launch(dict(mode='export', model=model, imgsz=self.imgsz.value()))

    def launch(self, config):
        if not self.project or self.busy():
            self.notice.setText('프로젝트를 열고 현재 작업이 끝난 뒤 시작하세요.')
            return
        try:
            validate_train_config(config)
            name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
            self.job_dir = self.project.root / 'runs' / name
            self.job_dir.mkdir()
            config.update(project=str(self.job_dir), name='segment')
            atomic_json(self.job_dir / 'config.json', config)
            self.process = QtCore.QProcess(self)
            self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.process.setWorkingDirectory(str(REPO_ROOT))
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.stopping = False
            self.output_buffer = ''
            self.log.clear()
            self.progress.setRange(0, config.get('epochs', 0))
            self.progress.setValue(0)
            self.panel.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.notice.setText('학습 중…' if config['mode'] == 'train' else 'ONNX 내보내기 중…')
            self.process.start(sys.executable, ['-u', '-m', 'studio.jobs', '--config', str(self.job_dir / 'config.json')])
        except (ValueError, OSError) as exc:
            self.notice.setText(str(exc))

    def read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        self.output_buffer += text
        while '\n' in self.output_buffer:
            line, self.output_buffer = self.output_buffer.split('\n', 1)
            if line.startswith('STUDIO_PROGRESS '):
                try:
                    data = json.loads(line[len('STUDIO_PROGRESS '):])
                    self.progress.setRange(0, data['epochs'])
                    self.progress.setValue(data['epoch'])
                except (ValueError, KeyError):
                    pass

    def process_error(self, error):
        if self.process and self.process.state() == QtCore.QProcess.NotRunning:
            self.panel.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.notice.setText('작업 프로세스를 시작하지 못했습니다: ' + self.process.errorString())

    def finished(self, code, status):
        self.read_output()
        self.panel.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, max(1, self.progress.maximum()))
        result = {}
        try:
            result = json.loads((self.job_dir / 'result.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        if self.stopping or result.get('status') == 'cancelled':
            self.notice.setText('작업을 중지했습니다. 저장된 중간 가중치는 실행 폴더에 남습니다.')
        elif code == 0 and result.get('status') == 'success':
            artifact = result.get('best') or result.get('onnx')
            if result.get('best'):
                self.best_model = artifact
            self.progress.setValue(self.progress.maximum())
            self.notice.setText('완료 · ' + artifact)
            self.artifactReady.emit(artifact)
        else:
            self.notice.setText('작업 실패 · ' + result.get('error', f'종료 코드 {code}. 로그를 확인하세요.'))

    def stop(self):
        if self.busy():
            self.stopping = True
            process = self.process
            process.terminate()
            self.stop_button.setEnabled(False)
            self.notice.setText('작업 중지 요청 중…')
            QtCore.QTimer.singleShot(4000, lambda: process.kill() if process.state() != QtCore.QProcess.NotRunning else None)

    def busy(self):
        return bool(self.process and self.process.state() != QtCore.QProcess.NotRunning)
