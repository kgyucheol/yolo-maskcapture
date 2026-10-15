from pathlib import Path

from .capture import CapturePage
from .labeling import LabelingPage
from .prediction import PredictionPage
from .project import Project
from .qt import QtCore, QtWidgets, Qt
from .theme import STYLESHEET
from .training import DatasetPage, TrainingPage
from .widgets import button, label


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Segmentation Studio')
        self.resize(1440, 940)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(STYLESHEET)
        self.project = None
        self._closing = False
        self._switching = False
        base = QtWidgets.QWidget()
        self.setCentralWidget(base)
        root = QtWidgets.QHBoxLayout(base)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(205)
        sidebar.setStyleSheet('QFrame { background: #111b29; }')
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(14, 26, 14, 18)
        side.addWidget(label('SEGMENTATION', 'eyebrow'))
        brand = label('Studio', 'title')
        brand.setStyleSheet('font-size: 29px; padding-bottom: 12px;')
        side.addWidget(brand)
        side.addWidget(label('COLLECT  /  LABEL  /  TRAIN', 'subtitle'))
        self.navigation = QtWidgets.QListWidget()
        self.navigation.setObjectName('navigation')
        self.navigation.addItems(['01    데이터 수집', '02    마스크 라벨링', '03    데이터셋', '04    모델 학습', '05    추론 확인'])
        side.addWidget(self.navigation, 1)
        side.addWidget(label('LOCAL WORKSPACE', 'eyebrow'))
        self.project_badge = label('프로젝트 미선택', 'subtitle')
        side.addWidget(self.project_badge)
        side.addWidget(button('사용 방법 / 단축키', self.show_help))
        root.addWidget(sidebar)
        main = QtWidgets.QVBoxLayout()
        main.setContentsMargins(24, 20, 24, 12)
        main.setSpacing(18)
        top = QtWidgets.QHBoxLayout()
        self.project_title = label('새 프로젝트로 시작하세요', 'section')
        top.addWidget(self.project_title, 1)
        top.addWidget(button('프로젝트 열기', self.open_project))
        top.addWidget(button('새 프로젝트', self.new_project, True))
        main.addLayout(top)
        self.stack = QtWidgets.QStackedWidget()
        self.capture = CapturePage()
        self.labeling = LabelingPage()
        self.dataset = DatasetPage()
        self.training = TrainingPage()
        self.prediction = PredictionPage(self.capture)
        for page in (self.capture, self.labeling, self.dataset, self.training, self.prediction):
            self.stack.addWidget(page)
        main.addWidget(self.stack, 1)
        root.addLayout(main, 1)
        self.statusBar().showMessage('프로젝트를 만들거나 열면 수집 → 라벨링 → 데이터셋 → 학습 → 추론을 이어서 진행할 수 있습니다.')
        self.navigation.currentRowChanged.connect(self.change_page)
        self.navigation.setCurrentRow(0)
        self.capture.imagesAdded.connect(self.on_images_added)
        self.labeling.projectChanged.connect(self.on_project_changed)
        self.dataset.datasetReady.connect(self.training.data.setText)
        self.training.artifactReady.connect(self.on_artifact)

    def set_project(self, project):
        self.project = project
        self.capture.set_project(project)
        self.labeling.set_project(project)
        self.dataset.set_project(project)
        self.training.set_project(project)
        self.prediction.set_project(project)
        self.project_title.setText(project.data['name'])
        self.project_badge.setText(project.root.name)
        self.project_badge.setToolTip(str(project.root))
        self.setWindowTitle(f"{project.data['name']} — Segmentation Studio")
        self.on_project_changed()

    def prepare_switch(self):
        if self.training.busy() or self.dataset.busy() or self.prediction.busy():
            self.statusBar().showMessage('현재 실행 중인 학습·생성·추론 작업을 완료하거나 중지한 뒤 프로젝트를 전환하세요.')
            return False
        if not self.labeling.save_current():
            self.statusBar().showMessage('현재 도형을 F로 완료하거나 C로 취소한 뒤 전환하세요.')
            return False
        self.capture.stop()
        return True

    def new_project(self):
        if not self.prepare_switch():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '새 프로젝트용 빈 폴더 선택')
        if path:
            try:
                self.set_project(Project.create(path))
            except (ValueError, OSError) as exc:
                QtWidgets.QMessageBox.warning(self, '프로젝트 생성 실패', str(exc))

    def open_project(self):
        if not self.prepare_switch():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '프로젝트 열기', '', 'Studio project (project.json)')
        if path:
            try:
                self.set_project(Project.open(path))
            except (ValueError, OSError) as exc:
                QtWidgets.QMessageBox.warning(self, '프로젝트 열기 실패', str(exc))

    def change_page(self, index):
        if index < 0 or self._switching:
            return
        previous = self.stack.currentIndex()
        if previous == 1 and index != 1 and not self.labeling.save_current():
            self._switching = True
            self.navigation.setCurrentRow(previous)
            self._switching = False
            return
        self.stack.setCurrentIndex(index)
        if index == 1:
            self.labeling.refresh_images()
            if self.project and self.project.images and self.labeling.canvas.doc is None:
                self.labeling.load_image(0)
            self.labeling.canvas.setFocus()
        elif index == 2:
            self.dataset.refresh()
        elif index == 0:
            self.capture.setFocus()

    def on_images_added(self):
        self.labeling.refresh_images()
        if self.project and self.project.images and self.labeling.canvas.doc is None:
            self.labeling.load_image(0)
        self.on_project_changed()

    def on_project_changed(self):
        if self.project:
            self.statusBar().showMessage(f'{self.project.root}   ·   이미지 {len(self.project.images)}장   ·   태그 {len(self.project.classes)}개')
            self.capture.update_count()

    def on_artifact(self, path):
        self.prediction.stop()
        self.prediction.model.setText(path)
        self.statusBar().showMessage('추론 화면에 모델을 연결했습니다: ' + path)

    def show_help(self):
        QtWidgets.QMessageBox.information(self, '사용 방법과 왼손 단축키',
            '1. 수집: 스트림 연결 후 Space 촬영 / Ctrl+Space 자동 촬영\n'
            '2. 라벨링: 클래스 이름을 태그로 등록\n'
            '   W 폴리곤 / E 사각형 / R 브러시 / Shift+R 지우개\n'
            '   F 도형 완료 → 1–9 숫자로 선택 객체 태그 지정\n'
            '   Q 선택·꼭짓점 이동 / X 객체 삭제 / C 그리기 취소\n'
            '   Z 실행 취소 / Shift+Z 다시 실행 / A·D 이미지 이동\n'
            '   S 저장 / V 마스크 표시 / Space+드래그 화면 이동\n'
            '   휠 확대·축소 / Ctrl+휠 브러시 크기\n'
            '   클래스 10개 이상은 태그 목록 클릭으로 지정\n'
            '3. 데이터셋: 검증 비율 설정 후 새 데이터셋 생성\n'
            '4. 학습: yolo11s-seg.pt / imgsz=864 기본값\n'
            '5. 추론: 모델과 이미지·폴더·ZMQ 입력 선택\n\n'
            '잘린 물체는 보이는 부분만 마스크로 지정하세요.\n'
            '태그 미지정 이미지는 학습 데이터셋에 포함되지 않습니다.')

    def closeEvent(self, event):
        if not self._closing:
            if not self.labeling.save_current():
                self.statusBar().showMessage('F로 도형 완료 또는 C로 취소한 뒤 종료하세요.')
                event.ignore()
                return
            if self.training.busy():
                answer = QtWidgets.QMessageBox.question(self, '학습 작업 진행 중',
                    '진행 중인 학습/내보내기를 중지하고 종료할까요?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if answer != QtWidgets.QMessageBox.Yes:
                    event.ignore()
                    return
            self._closing = True
            self.training.stop()
            self.prediction.stop()
            self.capture.stop()
        if self.dataset.busy() or self.prediction.busy() or self.training.busy() or (self.capture.worker and self.capture.worker.isRunning()):
            self.statusBar().showMessage('작업을 정리하는 중입니다. 완료 후 종료합니다…')
            event.ignore()
            QtCore.QTimer.singleShot(250, self.close)
            return
        self.labeling.timer.stop()
        event.accept()
