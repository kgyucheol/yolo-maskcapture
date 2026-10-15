"""Training controls kept separate from the process and dataset lifecycle."""
from .jobs import TRAIN_DEFAULTS, OPTIMIZERS
from .qt import QtWidgets
from .widgets import help_label, unit_field, device_selector, label


class TrainingOptions(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        basic_widget, basic = self.section()
        layout.addWidget(basic_widget)
        self.advanced_toggle = QtWidgets.QPushButton('▸ 상세 설정')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setAccessibleName('상세 학습 설정 펼치기 또는 접기')
        self.advanced_toggle.setToolTip('Optimizer, 학습률, 마스크 해상도와 성능 설정을 표시합니다.\n'
                                       '접어도 변경한 값은 유지되며 학습에 적용됩니다.')
        layout.addWidget(self.advanced_toggle)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        for title in ('최적화', '마스크·성능'):
            widget, form = self.section()
            self.tabs.addTab(widget, title)
            if title == '최적화':
                advanced = form
            else:
                masks = form
        self.tabs.hide()
        self.advanced_toggle.toggled.connect(self.show_advanced)

        def add(form, title, field, explanation, unit=''):
            field.setToolTip(explanation)
            field.setAccessibleName(title)
            form.addRow(help_label(title, explanation), unit_field(field, unit))

        self.epochs = self.integer(1, 100000, 100)
        add(basic, '반복 횟수', self.epochs,
            'epochs · 전체 학습 이미지를 몇 번 반복해서 학습할지 정합니다.\n'
            '기본 100. 조기 종료 조건을 만족하면 이보다 먼저 끝날 수 있습니다.', '회')
        self.imgsz = self.integer(32, 4096, 864, 32)
        add(basic, '입력 크기', self.imgsz,
            'imgsz · 학습 시 입력 이미지 크기입니다. 32의 배수로 입력하세요.\n'
            '848×480 원본은 864부터 시작할 수 있습니다. 비율을 유지해 크기를 맞추고 패딩합니다.\n'
            '큰 값은 작은 물체 표현에 유리할 수 있지만 GPU 메모리와 시간이 더 필요합니다.', 'px')
        self.batch = self.integer(1, 1024, 4)
        self.auto_batch = QtWidgets.QCheckBox('자동')
        self.auto_batch.setToolTip('batch=-1 · GPU 메모리에 맞게 배치 크기를 추정합니다.')
        self.auto_batch.toggled.connect(lambda checked: self.batch.setEnabled(not checked))
        batch_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(batch_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        row.addWidget(unit_field(self.batch, '장'), 1)
        row.addWidget(self.auto_batch)
        add(basic, '배치 크기', batch_row,
            'batch · 한 번의 가중치 갱신에 사용하는 이미지 수입니다. 기본 4.\n'
            '메모리 부족 오류가 나면 낮추세요. 자동을 켜면 GPU 메모리 기준으로 추정합니다.')
        self.device = device_selector()
        add(basic, '실행 장치', self.device, self.device.toolTip())
        self.patience = self.integer(0, 100000, TRAIN_DEFAULTS['patience'])
        add(basic, '조기 종료 대기', self.patience,
            'patience · 검증 성능이 개선되지 않아도 기다릴 epoch 수입니다.\n'
            '기본 100. 20~50으로 낮추면 불필요한 반복을 줄일 수 있습니다. 0은 조기 종료를 끕니다.', '회')
        self.rect = QtWidgets.QCheckBox('사용')
        add(basic, '직사각형 배치', self.rect,
            'rect · 비슷한 종횡비의 이미지를 직사각형 배치로 묶어 패딩을 줄입니다.\n'
            '켜면 mosaic 증강이 비활성화될 수 있습니다. 기본은 끔입니다.')

        self.optimizer = QtWidgets.QComboBox()
        self.optimizer.addItems(OPTIMIZERS)
        add(advanced, 'Optimizer', self.optimizer,
            'optimizer · 가중치를 갱신하는 방법입니다.\n'
            'auto는 학습 규모에 맞춰 방법과 초기 학습률·모멘텀을 자동 결정합니다.\n'
            '직접 학습률을 조절하려면 SGD, Adam 또는 AdamW를 선택하세요.')
        self.lr0 = self.decimal(.000001, 1, TRAIN_DEFAULTS['lr0'], 6, .001)
        add(advanced, '초기 학습률', self.lr0,
            'lr0 · 가중치를 한 번에 얼마나 바꿀지 정합니다.\n'
            '너무 높으면 불안정하고 너무 낮으면 학습이 느립니다.\n'
            '시작 예: SGD 0.01, Adam/AdamW 0.001. auto에서는 직접 설정값을 사용하지 않습니다.')
        self.lrf = self.decimal(0, 1, TRAIN_DEFAULTS['lrf'], 3, .01)
        add(advanced, '최종 학습률 비율', self.lrf,
            'lrf · 학습 마지막의 학습률을 초기 학습률 대비 비율로 정합니다.\n'
            '기본 0.01: 마지막 학습률은 초기 학습률의 1%입니다.')
        self.momentum = self.decimal(0, .999, TRAIN_DEFAULTS['momentum'], 3, .001)
        add(advanced, '모멘텀', self.momentum,
            'momentum · 이전 갱신 방향을 얼마나 반영할지 정합니다.\n'
            'SGD 모멘텀 또는 Adam 계열 beta1에 사용합니다. 기본 0.937.\n'
            'SGD는 0보다 커야 합니다. auto에서는 자동으로 결정합니다.')
        self.weight_decay = self.decimal(0, 1, TRAIN_DEFAULTS['weight_decay'], 6, .0001)
        add(advanced, '가중치 감쇠', self.weight_decay,
            'weight_decay · 과도하게 큰 가중치를 억제하는 규제 강도입니다.\n'
            '과적합 완화에 사용하며 기본 0.0005. 너무 크면 학습을 방해할 수 있습니다.')
        self.warmup_epochs = self.decimal(0, 1000, TRAIN_DEFAULTS['warmup_epochs'], 1, .5)
        add(advanced, '워밍업', self.warmup_epochs,
            'warmup_epochs · 초반에 학습률을 서서히 올려 안정화하는 기간입니다.\n'
            '기본 3.0 epoch. 0이면 워밍업을 생략합니다.', '회')
        self.cos_lr = QtWidgets.QCheckBox('사용')
        add(advanced, '코사인 학습률', self.cos_lr,
            'cos_lr · 코사인 곡선으로 학습률을 낮춥니다.\n'
            '끄면 기본 선형 감소를 사용합니다. 최종 비율은 lrf로 정합니다.')
        self.optimizer_note = label('auto에서는 초기 학습률과 모멘텀을 자동으로 정합니다.', 'subtitle')
        advanced.addRow(self.optimizer_note)
        self.optimizer.currentTextChanged.connect(self.optimizer_changed)
        self.optimizer_changed()

        self.mask_ratio = QtWidgets.QComboBox()
        for value in (1, 2, 4, 8, 16):
            self.mask_ratio.addItem(str(value), value)
        self.mask_ratio.setCurrentIndex(self.mask_ratio.findData(TRAIN_DEFAULTS['mask_ratio']))
        add(masks, '마스크 축소 비율', self.mask_ratio,
            'mask_ratio · 학습용 마스크의 가로·세로 해상도를 나누는 값입니다.\n'
            '기본 4. 1~2는 더 세밀한 경계를 보존하지만 메모리를 더 사용합니다.\n'
            '원본 라벨 파일의 해상도는 변경하지 않습니다.', '배')
        self.overlap_mask = QtWidgets.QCheckBox('사용')
        self.overlap_mask.setChecked(TRAIN_DEFAULTS['overlap_mask'])
        add(masks, '겹친 마스크 병합', self.overlap_mask,
            'overlap_mask · 한 이미지의 객체 마스크들을 학습용 단일 마스크 맵에 합칩니다.\n'
            '겹친 영역에서는 작은 객체를 큰 객체 위에 놓습니다. 기본 켬. 클래스가 합쳐지는 것은 아닙니다.')
        self.amp = QtWidgets.QCheckBox('사용')
        self.amp.setChecked(TRAIN_DEFAULTS['amp'])
        add(masks, '혼합 정밀도', self.amp,
            'amp · 지원하는 GPU에서 혼합 정밀도로 메모리 사용량을 줄이고 학습을 빠르게 합니다.\n'
            '기본 켬. 수치 불안정 문제가 있으면 끄세요. 장치에 따라 자동 비활성화됩니다.')
        self.workers = self.integer(0, 32, TRAIN_DEFAULTS['workers'])
        add(masks, '로더 작업 수', self.workers,
            'workers · 이미지를 병렬로 읽는 작업 프로세스 수입니다.\n'
            '앱 기본 0은 현재 학습 프로세스에서 읽습니다. 데이터 로딩이 병목이면 2~4를 시도하세요.', '개')
        self.seed = self.integer(0, 2147483647, TRAIN_DEFAULTS['seed'])
        add(masks, '학습 시드', self.seed,
            'seed · 학습의 무작위 동작을 재현하기 위한 기준 숫자입니다. 기본 0.\n'
            '데이터셋 분할·증강 시드와 별개입니다. 같은 시드도 장치·라이브러리가 다르면 결과가 달라질 수 있습니다.')

    @staticmethod
    def section():
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 12, 10, 10)
        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        layout.addStretch()
        return widget, form

    def show_advanced(self, expanded):
        self.tabs.setVisible(expanded)
        self.advanced_toggle.setText('▾ 상세 설정' if expanded else '▸ 상세 설정')

    @staticmethod
    def integer(minimum, maximum, value, step=1):
        field = QtWidgets.QSpinBox()
        field.setRange(minimum, maximum)
        field.setSingleStep(step)
        field.setValue(value)
        return field

    @staticmethod
    def decimal(minimum, maximum, value, decimals, step):
        field = QtWidgets.QDoubleSpinBox()
        field.setDecimals(decimals)
        field.setRange(minimum, maximum)
        field.setSingleStep(step)
        field.setValue(value)
        return field

    def optimizer_changed(self):
        manual = self.optimizer.currentText() != 'auto'
        self.lr0.setEnabled(manual)
        self.momentum.setEnabled(manual)
        self.momentum.setMinimum(.001 if self.optimizer.currentText() == 'SGD' else 0)
        self.optimizer_note.setText('초기 학습률과 모멘텀에 직접 입력한 값을 사용합니다.' if manual else
                                   'auto에서는 초기 학습률과 모멘텀을 자동으로 정합니다.')

    def config(self):
        result = {name: getattr(self, name).value() for name in (
            'epochs', 'imgsz', 'patience', 'lr0', 'lrf', 'momentum', 'weight_decay',
            'warmup_epochs', 'workers', 'seed')}
        result.update({name: getattr(self, name).isChecked() for name in (
            'rect', 'cos_lr', 'amp', 'overlap_mask')})
        result.update(batch=-1 if self.auto_batch.isChecked() else self.batch.value(),
                      device=self.device.currentData(), optimizer=self.optimizer.currentText(),
                      mask_ratio=self.mask_ratio.currentData())
        return result
