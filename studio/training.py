import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .dataset import build_dataset, split_sources
from .augmentation import METHOD_NAMES, augmentation_plan, augment_sample, composition_counts, sample_seed
from .augmentation_ui import AugmentationControls, mask_preview
from .jobs import validate_train_config
from .project import Project, atomic_json, read_image
from .qt import QtCore, QtGui, QtWidgets, Qt, Signal
from .training_options import TrainingOptions
from .training_progress import TrainingETA, TrainingLog, duration_text
from .file_dialogs import choose_file
from .widgets import ImageView, ImageComparison, button, card, label, page_header, scroll_panel, help_label

REPO_ROOT = Path(__file__).resolve().parent.parent


class DatasetWorker(QtCore.QThread):
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, project, ratio, seed, parent=None, augmentation=None):
        super().__init__(parent)
        self.project = Project(project.root, copy.deepcopy(project.data))
        self.ratio, self.seed = ratio, seed
        self.augmentation = augmentation

    def run(self):
        try:
            self.ready.emit(str(build_dataset(self.project, self.ratio, self.seed, self.augmentation)))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetPage(QtWidgets.QWidget):
    datasetReady = Signal(str)

    def __init__(self):
        super().__init__()
        self.project = None
        self.worker = None
        self.preview_index = 0
        self.completed = 0
        self.completed_names = []
        self._preview_plan_key = None
        self._preview_plan = []
        self.comparison_dialog = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('학습 데이터셋', '작업 완료 이미지를 검증하고, 원본을 보존하는 새 데이터셋을 만듭니다.'))
        self.summary = label('프로젝트를 열어 주세요.', 'badge')
        self.summary.setToolTip('태그 미지정·미작업 이미지는 제외됩니다. 빈 이미지는 라벨링 화면에서 배경으로 확인해야 포함됩니다.')
        layout.addWidget(self.summary)
        row = QtWidgets.QHBoxLayout()
        panel, box = card('01  원본 분할')
        form = QtWidgets.QFormLayout()
        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setRange(0.01, 0.99)
        self.ratio.setSingleStep(0.05)
        self.ratio.setValue(0.2)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        form.addRow('검증 데이터 비율', self.ratio)
        self.seed.setToolTip('무작위 분할과 증강을 재현하기 위한 기준 숫자입니다.\n같은 원본·설정·시드이면 같은 결과가 나옵니다. 보통 42를 유지하세요.')
        form.addRow('재현 시드', self.seed)
        box.addLayout(form)
        box.addWidget(label('시드: 무작위 분할·증강의 기준 숫자입니다.\n같은 데이터와 시드이면 같은 결과를 재현합니다.', 'subtitle'))
        self.augment = AugmentationControls()
        box.addWidget(self.augment)
        self.estimate = label('완료 이미지가 최소 2장 필요합니다.', 'badge')
        box.addWidget(self.estimate)
        self.create_button = button('현재 설정으로 데이터셋 생성', self.create_dataset, True)
        row.addWidget(scroll_panel(panel, 410))
        result, box = card('03  미리보기 확인')
        box.setSpacing(8)
        self.preview_tabs = QtWidgets.QTabWidget()
        preview = QtWidgets.QWidget()
        preview_box = QtWidgets.QVBoxLayout(preview)
        preview_box.setContentsMargins(8, 8, 8, 8)
        toolbar = QtWidgets.QHBoxLayout()
        self.show_masks = QtWidgets.QCheckBox('마스크 함께 표시')
        self.show_masks.setChecked(True)
        toolbar.addWidget(self.show_masks)
        toolbar.addStretch()
        toolbar.addWidget(button('크게 비교', self.open_comparison))
        preview_box.addLayout(toolbar)
        self.comparison = ImageComparison()
        self.original_view = self.comparison.original_view
        self.augmented_view = self.comparison.augmented_view
        preview_box.addWidget(self.comparison, 1)
        self.preview_status = label('미리보기는 원본과 주석을 변경하지 않습니다.', 'subtitle')
        preview_box.addWidget(self.preview_status)
        preview_box.addWidget(button('다음 생성 예정 이미지 보기', self.next_preview))
        self.preview_tabs.addTab(preview, '원본 / 증강 비교')
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText('생성 결과, 제외된 이미지, 마스크 변환 보고서가 여기에 표시됩니다.')
        self.report.setMinimumHeight(80)
        self.preview_tabs.addTab(self.report, '생성 결과')
        box.addWidget(self.preview_tabs, 1)
        row.addWidget(result, 1)
        layout.addLayout(row, 1)
        layout.addWidget(self.create_button)
        self.notice = label('학습·검증 이미지가 각각 최소 1장 필요합니다.', 'notice')
        layout.addWidget(self.notice)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(150)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.augment.changed.connect(self.schedule_preview)
        self.ratio.valueChanged.connect(self.schedule_preview)
        self.seed.valueChanged.connect(self.schedule_preview)
        self.show_masks.toggled.connect(self.refresh_preview)

    def schedule_preview(self):
        self.preview_index = 0
        self.update_estimate()
        self.preview_timer.start()

    def next_preview(self):
        self.preview_index += 1
        self.refresh_preview()

    def open_comparison(self):
        if self.original_view.frame is None:
            self.notice.setText('라벨링 완료 이미지가 있어야 크게 비교할 수 있습니다.')
            return
        if self.comparison_dialog:
            self.comparison_dialog.raise_()
            return
        dialog = QtWidgets.QDialog(self)
        self.comparison_dialog = dialog
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle('학습 원본 / 생성 예정 증강본 크게 비교')
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        area = self.screen().availableGeometry()
        dialog.resize(int(area.width() * .94), int(area.height() * .9))
        box = QtWidgets.QVBoxLayout(dialog)
        self.large_comparison = ImageComparison()
        box.addWidget(self.large_comparison, 1)
        self.large_preview_status = label(self.preview_status.text(), 'subtitle')
        box.addWidget(self.large_preview_status)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(button('다음 생성 예정 이미지 보기', self.next_preview))
        controls.addWidget(button('닫기', dialog.close))
        box.addLayout(controls)
        dialog.finished.connect(lambda _: setattr(self, 'comparison_dialog', None))
        self.sync_comparison()
        dialog.show()

    def sync_comparison(self):
        if self.comparison_dialog:
            self.large_comparison.copy_frames(self.original_view.frame, self.augmented_view.frame)
            self.large_preview_status.setText(self.preview_status.text())

    def update_estimate(self):
        if self.completed < 2:
            self.estimate.setText('완료 이미지가 최소 2장 필요합니다.')
            return
        nval = min(self.completed - 1, max(1, int(self.completed * self.ratio.value())))
        ntrain = self.completed - nval
        try:
            config = self.augment.config()
            total = ntrain * config.copies
            single, composite = composition_counts(config, total)
            self.estimate.setText(f'학습 원본 {ntrain}장 ×{config.copies + 1} = {ntrain * (1 + config.copies)}장\n'
                                 f'추가 {total}장 · 단일 {single} / 복합 {composite}\n검증 {nval}장 · 원본 유지')
        except ValueError as exc:
            self.estimate.setText(str(exc))

    def refresh_preview(self):
        self.original_view.set_frame(None)
        self.augmented_view.set_frame(None)
        if not self.project:
            self.sync_comparison()
            return
        try:
            config = self.augment.config()
            train = split_sources(self.completed_names, self.ratio.value(), self.seed.value())['train'] if len(self.completed_names) >= 2 else self.completed_names
            if not train:
                self.preview_status.setText('라벨링을 완료한 이미지가 생기면 미리보기를 표시합니다.')
                return
            total = len(train) * config.copies
            key = (config, tuple(train), self.seed.value())
            if self._preview_plan_key != key:
                self._preview_plan = augmentation_plan(config, total, self.seed.value())
                self._preview_plan_key = key
            selected = self.preview_index % max(1, total)
            name = train[selected // config.copies] if config.copies else train[0]
            if name:
                objects, reviewed = self.project.load_annotation(name)
                if not reviewed:
                    raise ValueError('생성할 이미지의 라벨이 변경되었습니다. 데이터셋 화면을 다시 여세요.')
                frame = read_image(self.project.image_path(name))
                if frame is None:
                    raise ValueError(f'{name}: 이미지를 읽을 수 없습니다.')
                self.original_view.set_frame(mask_preview(frame, objects) if self.show_masks.isChecked() else frame)
                if config.copies:
                    index = selected % config.copies
                    image, masks, info = augment_sample(frame, objects, config,
                        sample_seed(self.seed.value(), name, index), methods=self._preview_plan[selected])
                    self.augmented_view.set_frame(mask_preview(image, masks) if self.show_masks.isChecked() else image)
                    methods = ' + '.join(METHOD_NAMES[name] for name in info['methods'])
                    kind = '단일' if len(info['methods']) == 1 else '복합'
                    self.preview_status.setText(f"예정 {selected + 1}/{total} · {kind} · {methods}\n원본: {name}\n"
                        f"{frame.shape[1]}×{frame.shape[0]} · 객체 {len(objects)} → {len(masks)}")
                else:
                    self.preview_status.setText('증강 꺼짐 · 원본 이미지로만 데이터셋을 만듭니다.')
                return
        except (ValueError, OSError) as exc:
            self.preview_status.setText('미리보기: ' + str(exc))
        finally:
            self.sync_comparison()

    def set_project(self, project):
        self.project = project
        self.preview_index = 0
        self.report.clear()
        self.refresh()

    def refresh(self):
        if not self.project:
            return
        completed, background, pending = 0, 0, 0
        self.completed_names = []
        for entry in self.project.images:
            path = self.project.annotation_path(entry['file'])
            try:
                data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if data.get('reviewed'):
                    completed += 1
                    self.completed_names.append(entry['file'])
                    background += not data.get('objects')
                else:
                    pending += 1
            except (OSError, ValueError):
                pending += 1
        self.summary.setText(f'전체 {len(self.project.images)}장   ·   완료 {completed}장 (배경 {background}장)   ·   미완료 {pending}장   ·   클래스 {len(self.project.classes)}개')
        self.completed = completed
        self.update_estimate()
        self.refresh_preview()

    def create_dataset(self):
        if not self.project or self.busy():
            return
        try:
            config = self.augment.config()
        except ValueError as exc:
            self.notice.setText(str(exc))
            return
        self.worker = DatasetWorker(self.project, self.ratio.value(), self.seed.value(), self, augmentation=config)
        self.worker.ready.connect(self.on_ready)
        self.worker.failed.connect(lambda text: self.notice.setText('생성 실패: ' + text))
        self.worker.finished.connect(lambda: self.create_button.setEnabled(True))
        self.create_button.setEnabled(False)
        self.notice.setText('이미지·라벨 검증 및 학습 이미지 증강 중…')
        self.worker.start()

    def on_ready(self, path):
        data = json.loads((Path(path).parent / 'report.json').read_text(encoding='utf-8'))
        lines = [f'저장 위치: {path}', '', f"학습 {len(data['splits']['train'])}장 / 검증 {len(data['splits']['val'])}장",
                 f"원본 배경 {data['backgrounds']}장 / 제외 {len(data['excluded'])}장"]
        augmentation = data.get('augmentation', {})
        samples = augmentation.get('samples', [])
        lines.extend([f"추가 증강 {augmentation.get('generated', 0)}장 · 검증 이미지는 원본만 유지",
                      f"증강 중 사라진 작은/화면 밖 객체 {sum(len(s['dropped_objects']) for s in samples)}개", '', '제외된 이미지'])
        lines.insert(3, f"학습 ×{augmentation.get('multiplier', 1)} · 단일 {augmentation.get('single', 0)}장 / 복합 {augmentation.get('composite', 0)}장")
        if samples:
            from collections import Counter
            counts = Counter(name for sample in samples for name in sample.get('methods', []))
            lines.insert(6, '실제 적용: ' + ' / '.join(f'{METHOD_NAMES[name]} {count}장' for name, count in counts.items()))
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
        self.log_stream = TrainingLog()
        self.eta = TrainingETA()
        self.eta_timer = QtCore.QTimer(self)
        self.eta_timer.setInterval(1000)
        self.eta_timer.timeout.connect(self.update_eta)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('세그먼테이션 학습', 'YOLO11s-seg로 학습하고 best.pt 또는 ONNX를 추론 화면에서 확인하세요.'))
        row = QtWidgets.QHBoxLayout()
        self.panel, box = card('학습 설정')
        self.panel.setMaximumWidth(380)
        form = QtWidgets.QFormLayout()
        self.model = QtWidgets.QLineEdit('yolo11s-seg.pt')
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model)
        model_row.addWidget(button('파일', self.browse_model))
        form.addRow(help_label('초기 모델', '사전 학습된 세그먼테이션 PT 가중치입니다.\n'
                               '기본 yolo11s-seg.pt. 파일이 없으면 Ultralytics가 다운로드합니다.'), model_row)
        self.data = QtWidgets.QLineEdit()
        self.data.setPlaceholderText('데이터셋 생성 후 자동 입력')
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(self.data)
        data_row.addWidget(button('파일', self.browse_data))
        form.addRow(help_label('데이터셋', '학습·검증 이미지 경로와 클래스 이름이 들어 있는 data.yaml입니다.\n'
                               '데이터셋 화면에서 생성하면 자동으로 연결됩니다.'), data_row)
        box.addLayout(form)
        self.options = TrainingOptions()
        for name in ('epochs', 'imgsz', 'batch', 'device', 'rect'):
            setattr(self, name, getattr(self.options, name))
        box.addWidget(self.options)
        box.addWidget(label('ⓘ에 마우스를 올리거나 클릭하면 항목별 설명을 볼 수 있습니다.', 'subtitle'))
        self.start_button = button('학습 시작', self.start_training, True)
        self.export_button = button('PT → ONNX 내보내기', self.start_export)
        box.addWidget(self.start_button)
        box.addWidget(self.export_button)
        box.addWidget(label('초기 모델이 로컬에 없으면 Ultralytics가 다운로드합니다. 학습에는 requirements.txt 패키지가 필요합니다.', 'subtitle'))
        box.addStretch()
        row.addWidget(scroll_panel(self.panel, 380))
        result, box = card()
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label('진행 상황', 'section'))
        self.eta_label = label('예상 남은 시간 —', 'badge')
        self.eta_label.setToolTip('최근 완료한 최대 5개 epoch의 학습·검증 시간을 기준으로 추정합니다.\n'
                                  '첫 epoch 완료 후 계산하며, 조기 종료 시 더 빨리 끝날 수 있습니다.')
        heading.addWidget(self.eta_label)
        heading.addStretch()
        box.addLayout(heading)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        box.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log.setStyleSheet("font-family: 'Noto Sans Mono', 'Noto Sans CJK KR', monospace;")
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
        path = choose_file(self, '초기 모델', 'PyTorch (*.pt)', 'initial_model', self.project, self.model.text())
        if path:
            self.model.setText(path)

    def browse_data(self):
        path = choose_file(self, '데이터셋 YAML', 'YAML (*.yaml *.yml)', 'dataset', self.project, self.data.text())
        if path:
            self.data.setText(path)

    def start_training(self):
        self.launch(dict(mode='train', model=self.model.text().strip(), data=self.data.text().strip(),
                         **self.options.config()))

    def start_export(self):
        model = self.best_model
        if not model:
            model = choose_file(self, '내보낼 세그먼테이션 PT', 'PyTorch (*.pt)', 'export_model', self.project)
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
            environment = QtCore.QProcessEnvironment.systemEnvironment()
            for key, value in dict(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', NO_COLOR='1', TERM='dumb').items():
                environment.insert(key, value)
            self.process.setProcessEnvironment(environment)
            self.process.setWorkingDirectory(str(REPO_ROOT))
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.stopping = False
            self.output_buffer = ''
            self.log_stream = TrainingLog()
            self.eta.finish()
            if config['mode'] == 'train':
                self.eta.start(config['epochs'])
                self.eta_timer.start()
                self.update_eta()
            else:
                self.eta_label.setText('내보내기 진행 중')
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
        self.consume_output(bytes(self.process.readAllStandardOutput()))

    def consume_output(self, output, final=False):
        lines, current = self.log_stream.feed(output, final=final)
        visible = []
        for line in lines:
            if line.startswith('STUDIO_PROGRESS '):
                try:
                    data = json.loads(line[len('STUDIO_PROGRESS '):])
                    self.progress.setRange(0, data['epochs'])
                    self.progress.setValue(data['epoch'])
                    self.eta.update(data['epoch'], data['epochs'], data.get('epoch_seconds'))
                    self.update_eta()
                except (ValueError, KeyError, TypeError):
                    pass
            elif not line.startswith('STUDIO_RESULT '):
                visible.append(line)
        if current.startswith(('STUDIO_PROGRESS ', 'STUDIO_RESULT ')):
            current = ''
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(''.join(line + '\n' for line in visible) + current)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def update_eta(self):
        remaining = self.eta.remaining()
        self.eta_label.setText('예상 남은 시간 계산 중…' if remaining is None else
                               '예상 남은 시간 약 ' + duration_text(remaining))

    def finish_eta(self, text):
        self.eta.finish()
        self.eta_timer.stop()
        self.eta_label.setText(text)

    def process_error(self, error):
        if self.process and self.process.state() == QtCore.QProcess.NotRunning:
            self.finish_eta('작업 실패')
            self.panel.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.notice.setText('작업 프로세스를 시작하지 못했습니다: ' + self.process.errorString())

    def finished(self, code, status):
        self.read_output()
        self.consume_output(b'', final=True)
        self.panel.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, max(1, self.progress.maximum()))
        result = {}
        try:
            result = json.loads((self.job_dir / 'result.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        if self.stopping or result.get('status') == 'cancelled':
            self.finish_eta('작업 중지됨')
            self.notice.setText('작업을 중지했습니다. 저장된 중간 가중치는 실행 폴더에 남습니다.')
        elif code == 0 and result.get('status') == 'success':
            self.finish_eta('완료')
            artifact = result.get('best') or result.get('onnx')
            if result.get('best'):
                self.best_model = artifact
            self.progress.setValue(self.progress.maximum())
            self.notice.setText('완료 · ' + artifact)
            self.artifactReady.emit(artifact)
        else:
            self.finish_eta('작업 실패')
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
