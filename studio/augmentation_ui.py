"""Dataset augmentation controls and mask previews."""
import cv2
import numpy as np

from .augmentation import AugmentConfig
from .qt import QtWidgets, Signal
from .theme import class_color
from .widgets import label, unit_field


class AugmentToggle(QtWidgets.QCheckBox):
    def hitButton(self, position):
        return self.rect().contains(position)


class AugmentationControls(QtWidgets.QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        self.enable_box = AugmentToggle('02  데이터 증강 사용')
        self.enable_box.setObjectName('augmentEnable')
        self.enable_box.setMinimumHeight(40)
        root.addWidget(self.enable_box)
        self.enabled_note = label('꺼짐 · 체크하면 아래 설정으로 학습 이미지를 늘립니다.', 'subtitle')
        root.addWidget(self.enabled_note)
        self.settings = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(self.settings)
        box.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.settings)
        form = QtWidgets.QFormLayout()
        self.multiplier = QtWidgets.QComboBox()
        for value in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]:
            self.multiplier.addItem(str(value), value)
        self.multiplier.setCurrentIndex(self.multiplier.findData(3))
        form.addRow('학습 생성 배수', unit_field(self.multiplier, '배 · 원본 포함'))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem('무작위 조합 (권장)', 'random')
        self.mode.addItem('한 가지씩 적용', 'single')
        self.mode.addItem('선택한 기법 모두 적용', 'all')
        form.addRow('적용 방식', self.mode)
        share_row = QtWidgets.QHBoxLayout()
        self.single_share = QtWidgets.QSpinBox()
        self.single_share.setRange(0, 100)
        self.single_share.setValue(50)
        self.single_share.setToolTip('추가 증강본 중 한 가지 기법만 적용할 비율입니다.\n전체 장수로 반올림하고 나머지는 2개 이상 조합합니다.')
        self.composite_share = label('복합 50%', 'subtitle')
        share_row.addWidget(label('단일', 'subtitle'))
        share_row.addWidget(unit_field(self.single_share, '%'), 1)
        share_row.addWidget(self.composite_share)
        form.addRow('단일 / 복합 비율', share_row)
        self.max_transforms = QtWidgets.QSpinBox()
        self.max_transforms.setRange(2, 11)
        self.max_transforms.setValue(3)
        form.addRow('동시 적용 최대', unit_field(self.max_transforms, '개'))
        box.addLayout(form)
        box.addWidget(label('종류별로 기법을 선택하고 강도와 선택 비중을 정하세요.', 'subtitle'))
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        box.addWidget(self.tabs)
        grids = {}
        for name in ('기하 변환', '색상·조명', '노이즈·화질'):
            page = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(page)
            grid.setContentsMargins(5, 12, 5, 8)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(9)
            for column, title in enumerate(('기법 선택', '강도', '선택 비중')):
                grid.addWidget(label(title, 'subtitle'), 0, column)
            self.tabs.addTab(page, name)
            grids[name] = grid
        self.fields = {}
        self.method_checks = {}
        self.checks = {}
        self.weights = {}
        specs = [
            ('기하 변환', 'horizontal_flip', '좌우 반전', None, True, ''),
            ('기하 변환', 'vertical_flip', '상하 반전', None, False, ''),
            ('기하 변환', 'rotation', '회전 ±', 180, 10, '°'),
            ('기하 변환', 'scale', '크기 ±', 50, 10, '%'),
            ('기하 변환', 'translate', '이동 ±', 30, 5, '%'),
            ('색상·조명', 'brightness', '밝기 ±', 50, 15, '%'),
            ('색상·조명', 'contrast', '대비 ±', 50, 15, '%'),
            ('색상·조명', 'saturation', '채도 ±', 50, 15, '%'),
            ('색상·조명', 'hue', '색상 ±', 30, 3, ' HSV'),
            ('노이즈·화질', 'noise', '노이즈', 25, 3, ''),
            ('노이즈·화질', 'blur', '약한 흐림', None, False, '')]
        for category, name, title, maximum, default, suffix in specs:
            grid = grids[category]
            row = grid.rowCount()
            check = QtWidgets.QCheckBox(title)
            check.setChecked(bool(default))
            self.checks[name] = check
            grid.addWidget(check, row, 0)
            if maximum is None:
                grid.addWidget(label('3×3' if name == 'blur' else '—', 'subtitle'), row, 1)
            else:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0, maximum)
                spin.setDecimals(1)
                spin.setValue(default)
                spin.setMinimumWidth(75)
                self.fields[name] = spin
                self.method_checks[name] = check
                grid.addWidget(unit_field(spin, suffix), row, 1)
                spin.valueChanged.connect(self.changed)
            weight = QtWidgets.QSpinBox()
            weight.setRange(1, 100)
            weight.setValue(50)
            weight.setMinimumWidth(60)
            weight.setToolTip('상대적인 선택 비중입니다. 클수록 더 자주 선택됩니다.\n합계를 100으로 맞출 필요는 없습니다.')
            self.weights[name] = weight
            grid.addWidget(weight, row, 2)
            weight.valueChanged.connect(self.changed)
            check.toggled.connect(lambda checked, key=name: self.method_toggled(key, checked))
        self.horizontal_flip = self.checks['horizontal_flip']
        self.vertical_flip = self.checks['vertical_flip']
        self.blur = self.checks['blur']
        box.addWidget(label('선택 비중이 클수록 자주 뽑힙니다. 합계 100은 필요 없습니다.\n한 기법만 고르면 단일 100%로 생성됩니다.\n검증 이미지는 원본을 유지합니다.', 'subtitle'))
        self.enable_box.toggled.connect(self.enable_changed)
        self.enable_changed(False)
        self.multiplier.currentIndexChanged.connect(self.changed)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.max_transforms.valueChanged.connect(self.changed)
        self.single_share.valueChanged.connect(self.share_changed)
        self.mode_changed()

    def isChecked(self):
        return self.enable_box.isChecked()

    def setChecked(self, checked):
        self.enable_box.setChecked(checked)

    def enable_changed(self, checked):
        self.settings.setEnabled(checked)
        self.enabled_note.setText('켜짐 · 기법 선택 → 미리보기 확인 → 데이터셋 생성' if checked else
                                  '꺼짐 · 체크하면 아래 설정으로 학습 이미지를 늘립니다.')
        self.changed.emit()

    def share_changed(self, value):
        self.composite_share.setText(f'복합 {100 - value}%')
        self.changed.emit()

    def method_toggled(self, name, checked):
        if name in self.fields:
            self.fields[name].setEnabled(checked)
        self.weights[name].setEnabled(checked and self.mode.currentData() != 'all')
        self.changed.emit()

    def mode_changed(self):
        self.max_transforms.setEnabled(self.mode.currentData() == 'random')
        self.single_share.setEnabled(self.mode.currentData() == 'random')
        for name, check in self.checks.items():
            if name in self.fields:
                self.fields[name].setEnabled(check.isChecked())
            self.weights[name].setEnabled(check.isChecked() and self.mode.currentData() != 'all')
        self.changed.emit()

    def config(self):
        values = {name: (spin.value() if self.method_checks[name].isChecked() else 0) / (100 if name in ('scale', 'translate', 'brightness', 'contrast', 'saturation') else 1)
                  for name, spin in self.fields.items()}
        return AugmentConfig(copies=self.multiplier.currentData() - 1 if self.isChecked() else 0,
            horizontal_flip=self.horizontal_flip.isChecked(), vertical_flip=self.vertical_flip.isChecked(),
            blur=self.blur.isChecked(), mode=self.mode.currentData(), max_transforms=self.max_transforms.value(),
            single_ratio=self.single_share.value() / 100,
            method_weights=tuple((name, spin.value()) for name, spin in self.weights.items()), **values)


def mask_preview(frame, objects):
    image = frame.copy()
    for obj in objects:
        color = class_color(obj.class_id)
        bgr = np.array([color.blue(), color.green(), color.red()])
        selected = obj.mask > 0
        image[selected] = (image[selected].astype(np.float32) * 0.55 + bgr * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, tuple(int(v) for v in bgr), 1)
    return image
