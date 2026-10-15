import json

from .canvas import MaskCanvas
from .project import read_image
from .qt import QtCore, QtWidgets, Qt, Signal
from .theme import class_color
from .widgets import button, card, label, page_header, scroll_panel


class LabelingPage(QtWidgets.QWidget):
    projectChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.index = 0
        self.current_image = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('마스크 라벨링', '그리기 → F로 도형 완료 → 숫자로 태그 지정. 왼손은 키보드, 오른손은 마우스.'))
        tools = QtWidgets.QHBoxLayout()
        self.tool_buttons = {}
        self.canvas = MaskCanvas()
        for text, tool in [('Q  선택', 'select'), ('W  폴리곤', 'polygon'), ('E  사각형', 'rectangle'),
                           ('R  브러시', 'brush'), ('⇧R  지우개', 'eraser')]:
            btn = button(text, lambda checked=False, t=tool: self.canvas.set_tool(t))
            btn.setCheckable(True)
            btn.setChecked(tool == 'polygon')
            tools.addWidget(btn)
            self.tool_buttons[tool] = btn
        tools.addWidget(button('새 객체', self.canvas.new_object))
        tools.addStretch()
        tools.addWidget(button('화면 맞춤', self.canvas.fit))
        tools.addWidget(button('S  저장', self.save_current, True))
        layout.addLayout(tools)

        split = QtWidgets.QSplitter(Qt.Horizontal)
        left, box = card('이미지')
        self.image_list = QtWidgets.QListWidget()
        self.image_list.setMinimumWidth(150)
        box.addWidget(self.image_list)
        box.addWidget(button('이미지 가져오기', self.import_images))
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(button('A  이전', lambda: self.navigate(-1)))
        nav.addWidget(button('D  다음', lambda: self.navigate(1)))
        box.addLayout(nav)
        split.addWidget(left)
        split.addWidget(self.canvas)
        right, box = card('클래스 태그')
        box.addWidget(label('마스크를 선택한 뒤 숫자를 누르세요.', 'subtitle'))
        self.class_list = QtWidgets.QListWidget()
        self.class_list.setMinimumHeight(145)
        self.class_list.setMaximumHeight(240)
        box.addWidget(self.class_list)
        self.class_input = QtWidgets.QLineEdit()
        self.class_input.setPlaceholderText('클래스 이름 입력')
        box.addWidget(self.class_input)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button('태그 추가', self.add_class))
        row.addWidget(button('이름 변경', self.rename_class))
        box.addLayout(row)
        box.addWidget(label('이미지의 객체', 'section'))
        self.object_list = QtWidgets.QListWidget()
        box.addWidget(self.object_list)
        box.addWidget(button('X  선택 객체 삭제', self.delete_object))
        self.background = QtWidgets.QCheckBox('객체 없는 배경 이미지로 확인')
        self.background.setToolTip('객체가 전혀 없는 이미지만 체크하세요. 학습용 음성 예제로 포함됩니다.')
        box.addWidget(self.background)
        box.addWidget(label('브러시: 마스크 영역 추가\n지우개: 선택 마스크의 영역 제거\nSpace+드래그: 화면 이동\n휠: 확대 · Ctrl+휠: 붓 크기', 'subtitle'))
        split.addWidget(scroll_panel(right, 265))
        split.setSizes([205, 760, 265])
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.status = label('프로젝트를 만들거나 열어 주세요.', 'notice')
        layout.addWidget(self.status)
        layout.addWidget(label('F 완료   C 취소   1–9 태그   Z 취소 / ⇧Z 다시 실행   A/D 이미지   V 마스크 표시', 'subtitle'))

        self.canvas.changed.connect(self.on_changed)
        self.canvas.selectionChanged.connect(self.refresh_objects)
        self.canvas.toolChanged.connect(self.sync_tools)
        self.canvas.message.connect(self.status.setText)
        self.canvas.command.connect(self.command)
        self.image_list.currentRowChanged.connect(self.choose_image)
        self.class_list.itemClicked.connect(self.assign_selected_tag)
        self.object_list.itemClicked.connect(self.select_object)
        self.class_input.returnPressed.connect(self.add_class)
        self.background.toggled.connect(self.mark_dirty)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.autosave)
        self.timer.start()

    def sync_tools(self, tool):
        for name, btn in self.tool_buttons.items():
            btn.setChecked(name == tool)

    def set_project(self, project):
        self.current_image = None
        self.canvas.clear_image()
        self.background.blockSignals(True)
        self.background.setChecked(False)
        self.background.blockSignals(False)
        self.project = project
        self.index = 0
        self.refresh_classes()
        self.refresh_images()
        previous = project.data.get('last_image')
        names = [e['file'] for e in project.images]
        self.load_image(names.index(previous) if previous in names else 0)

    def refresh_classes(self):
        self.class_list.clear()
        classes = self.project.classes if self.project else []
        self.canvas.class_names = [c['name'] for c in classes]
        for c in classes:
            item = QtWidgets.QListWidgetItem(f"{c['id'] + 1}    {c['name']}")
            item.setForeground(class_color(c['id']))
            self.class_list.addItem(item)
        self.canvas.update()

    def add_class(self):
        if not self.project:
            self.status.setText('먼저 프로젝트를 만들어 주세요.')
            return
        try:
            self.project.add_class(self.class_input.text())
            self.class_input.clear()
            self.refresh_classes()
            self.projectChanged.emit()
            self.canvas.setFocus()
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def rename_class(self):
        index = self.class_list.currentRow()
        if not self.project or index < 0:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, '태그 이름 변경', '클래스 이름',
                                                text=self.project.classes[index]['name'])
        if ok:
            try:
                self.project.rename_class(index, text)
                self.refresh_classes()
                self.projectChanged.emit()
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def assign_selected_tag(self, item):
        self.canvas.assign_tag(self.class_list.row(item))
        self.canvas.setFocus()

    def select_object(self, item):
        if self.canvas.pending:
            self.status.setText('먼저 F로 도형을 완료하거나 C로 취소하세요.')
            return
        self.canvas.doc.selected = item.data(Qt.UserRole)
        self.canvas.setFocus()
        self.canvas.update()

    def delete_object(self):
        if self.canvas.doc and not self.canvas.pending:
            self.canvas.doc.delete_selected()
            self.canvas._notify()
            self.canvas.setFocus()

    def refresh_objects(self):
        self.object_list.clear()
        if not self.canvas.doc:
            return
        for i, obj in enumerate(self.canvas.doc.objects):
            name = '태그 미지정 · 숫자를 누르세요' if obj.class_id is None else self.canvas.class_names[obj.class_id]
            item = QtWidgets.QListWidgetItem(f'{i + 1:02d}    {name}')
            item.setData(Qt.UserRole, obj.id)
            item.setForeground(class_color(obj.class_id))
            self.object_list.addItem(item)
            if obj.id == self.canvas.doc.selected:
                self.object_list.setCurrentItem(item)
        self.background.setEnabled(not self.canvas.doc.objects)
        if self.canvas.doc.objects and self.background.isChecked():
            self.background.setChecked(False)

    def refresh_images(self):
        self.image_list.blockSignals(True)
        self.image_list.clear()
        if self.project:
            for i, entry in enumerate(self.project.images):
                image = entry['file']
                symbol = '○'
                path = self.project.annotation_path(image)
                if path.exists():
                    try:
                        data = json.loads(path.read_text(encoding='utf-8'))
                        symbol = ('✓' if data['objects'] else '◇') if data.get('reviewed') else '◐'
                    except (OSError, ValueError):
                        symbol = '!'
                item = QtWidgets.QListWidgetItem(f'{symbol}  {i + 1:04d}   {image[:23]}')
                item.setToolTip(image + '\n○ 미작업 · ◐ 작업 중 · ✓ 완료 · ◇ 배경')
                self.image_list.addItem(item)
            self.image_list.setCurrentRow(self.index)
        self.image_list.blockSignals(False)

    def load_image(self, index):
        self.current_image = None
        self.canvas.clear_image()
        self.object_list.clear()
        self.background.blockSignals(True)
        self.background.setChecked(False)
        self.background.blockSignals(False)
        if not self.project or not self.project.images:
            self.status.setText('수집 화면에서 촬영하거나 이미지를 가져오세요.')
            return
        index = max(0, min(index, len(self.project.images) - 1))
        self.index = index
        image = self.project.images[index]['file']
        try:
            frame = read_image(self.project.image_path(image))
            if frame is None:
                raise ValueError('원본 이미지를 읽을 수 없습니다.')
            objects, reviewed = self.project.load_annotation(image)
            self.index = index
            self.canvas.set_image(frame, objects)
            self.current_image = image
            self.background.blockSignals(True)
            self.background.setChecked(reviewed and not objects)
            self.background.blockSignals(False)
            self.refresh_objects()
            self.image_list.blockSignals(True)
            self.image_list.setCurrentRow(index)
            self.image_list.blockSignals(False)
            self.project.data['last_image'] = image
            self.project.save()
            self.status.setText(f'{index + 1} / {len(self.project.images)}   ·   {frame.shape[1]} × {frame.shape[0]}   ·   W로 그리기 시작')
            self.canvas.setFocus()
        except (ValueError, OSError) as exc:
            self.status.setText(f'{image}: {exc}')

    def save_current(self):
        if not self.project or not self.canvas.doc or self.current_image is None:
            return True
        if self.canvas.pending:
            self.status.setText('이동 전에 F로 도형을 완료하거나 C로 취소하세요.')
            return False
        if not self.canvas.doc.dirty:
            return True
        image = self.current_image
        try:
            objects = self.canvas.doc.objects
            reviewed = bool(objects) or self.background.isChecked()
            self.project.save_annotation(image, objects, reviewed)
            self.canvas.doc.dirty = False
            self.refresh_images()
            unassigned = sum(o.class_id is None for o in objects)
            self.status.setText('자동 저장됨' + (f' · 태그 미지정 {unassigned}개' if unassigned else ' · 작업 완료' if reviewed else ' · 미작업'))
            self.projectChanged.emit()
            return True
        except (ValueError, OSError) as exc:
            self.status.setText(f'저장 실패: {exc}')
            return False

    def mark_dirty(self):
        if self.canvas.doc:
            self.canvas.doc.dirty = True

    def on_changed(self):
        self.refresh_objects()
        self.status.setText('변경됨 · 잠시 후 자동 저장')

    def autosave(self):
        if self.project and self.canvas.doc and self.canvas.doc.dirty and not self.canvas.pending:
            self.save_current()

    def choose_image(self, index):
        if index >= 0 and index != self.index:
            if self.save_current():
                self.load_image(index)
            else:
                self.image_list.blockSignals(True)
                self.image_list.setCurrentRow(self.index)
                self.image_list.blockSignals(False)

    def navigate(self, direction):
        if self.project and self.project.images and self.save_current():
            self.load_image(self.index + direction)

    def command(self, command):
        if command == 'save':
            self.save_current()
        else:
            self.navigate(-1 if command == 'previous' else 1)

    def import_images(self):
        if not self.project or not self.save_current():
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, '이미지 가져오기', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)')
        errors = []
        start = len(self.project.images)
        for filename in files:
            try:
                self.project.import_image(filename)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        self.refresh_images()
        if files and start < len(self.project.images):
            self.load_image(start)
            self.projectChanged.emit()
        if errors:
            self.status.setText(' / '.join(errors[:3]))
