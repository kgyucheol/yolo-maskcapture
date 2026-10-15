import json

from .canvas import MaskCanvas
from .project import read_image
from .qt import QtCore, QtWidgets, Qt, Signal
from .sam import SamWorker
from .file_dialogs import choose_file
from .theme import class_color
from .widgets import button, card, label, page_header, scroll_panel, shortcut_button, shortcut_hint


class LabelingPage(QtWidgets.QWidget):
    projectChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.index = 0
        self.current_image = None
        self.sam_worker = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page_header('마스크 라벨링', 'AI로 물체 클릭 → 필요하면 영역 보정 → F 확정 → 숫자로 태그 지정'))
        tools = QtWidgets.QHBoxLayout()
        self.tool_buttons = {}
        self.canvas = MaskCanvas()
        for key, text, tool, tip in [
                ('T', 'AI 클릭', 'sam', 'SAM2 · 좌클릭 포함 / 우클릭 또는 Shift+좌클릭 제외'),
                ('Q', '선택', 'select', '객체 선택 · 폴리곤 꼭짓점 이동'),
                ('W', '폴리곤', 'polygon', '점으로 직접 그리기 · F 확정'),
                ('E', '사각형', 'rectangle', '드래그로 사각형 그리기 · F 확정'),
                ('R', '브러시 +', 'brush', '선택 마스크에 영역 추가 · Ctrl+휠 크기 조절'),
                ('⇧R', '지우개 −', 'eraser', '선택 마스크에서 영역 제거 · 원본 이미지는 유지')]:
            btn = shortcut_button(key, text, lambda checked=False, t=tool: self.canvas.set_tool(t), tip)
            btn.setChecked(tool == 'polygon')
            tools.addWidget(btn)
            self.tool_buttons[tool] = btn
        tools.addWidget(button('새 객체', self.canvas.new_object))
        tools.addStretch()
        tools.addWidget(button('화면 맞춤 · 중앙', self.canvas.reset_view))
        tools.addWidget(button('S  저장', self.save_current, True))
        layout.addLayout(tools)

        ai_bar = QtWidgets.QHBoxLayout()
        ai_bar.addWidget(label('SAM 2.1 Tiny', 'badge'))
        self.sam_status = label('가벼운 AI 클릭 마스크 · 첫 사용 시 모델 다운로드 · GPU 자동 선택', 'subtitle')
        ai_bar.addWidget(self.sam_status, 1)
        self.sam_load_button = button('AI 준비', self.prepare_sam)
        self.sam_release_button = button('AI 메모리 해제', self.release_sam)
        ai_bar.addWidget(self.sam_load_button)
        ai_bar.addWidget(self.sam_release_button)
        layout.addLayout(ai_bar)

        threshold_row = QtWidgets.QHBoxLayout()
        threshold_row.addWidget(label('AI 경계 임계값', 'section'))
        threshold_row.addWidget(label('넓게', 'subtitle'))
        self.sam_threshold = QtWidgets.QSlider(Qt.Horizontal)
        self.sam_threshold.setRange(-50, 50)
        self.sam_threshold.setValue(0)
        self.sam_threshold.setSingleStep(1)
        self.sam_threshold.setPageStep(5)
        self.sam_threshold.setMinimumWidth(160)
        self.sam_threshold.setMaximumWidth(360)
        self.sam_threshold.setToolTip('기본 0.0 · 음수: 포함 영역 확대 / 양수: 축소\nF 확정 전 AI 초안에만 적용합니다.')
        threshold_row.addWidget(self.sam_threshold, 1)
        threshold_row.addWidget(label('좁게', 'subtitle'))
        self.sam_threshold_value = label('0.0', 'badge')
        self.sam_threshold_value.setMinimumWidth(55)
        threshold_row.addWidget(self.sam_threshold_value)
        threshold_row.addWidget(button('기본값', self.reset_sam_threshold))
        threshold_row.addWidget(label('F 확정 전 AI 미리보기에 적용', 'subtitle'), 1)
        layout.addLayout(threshold_row)
        self.sam_threshold.valueChanged.connect(self.change_sam_threshold)
        self.sam_threshold.sliderReleased.connect(self.canvas.setFocus)

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
        box.addWidget(label('AI 클릭 보정', 'section'))
        box.addWidget(shortcut_hint('T', 'AI 도구 선택'))
        box.addWidget(shortcut_hint('좌 / 우', '포함 + / 제외 − 클릭'))
        box.addWidget(label('초록 점은 포함, 분홍 점은 제외.\nF로 확정할 때까지 같은 물체를 보정합니다.', 'subtitle'))
        box.addWidget(label('확정 후 직접 보정', 'section'))
        box.addWidget(shortcut_hint('R / ⇧R', '브러시 추가 / 지우개 제거'))
        box.addWidget(shortcut_hint('Ctrl+휠', '붓 크기 조절'))
        split.addWidget(scroll_panel(right, 265))
        split.setSizes([205, 760, 265])
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.status = label('프로젝트를 만들거나 열어 주세요.', 'notice')
        layout.addWidget(self.status)
        shortcuts = QtWidgets.QHBoxLayout()
        for key, text in [('F', '마스크 확정'), ('1–9', '태그 지정'), ('C', '초안 취소'),
                          ('Z / ⇧Z', '취소 / 다시 실행'), ('A / D', '이전 / 다음 이미지')]:
            shortcuts.addWidget(shortcut_hint(key, text), 1)
        layout.addLayout(shortcuts)

        self.canvas.changed.connect(self.on_changed)
        self.canvas.selectionChanged.connect(self.refresh_objects)
        self.canvas.toolChanged.connect(self.sync_tools)
        self.canvas.message.connect(self.status.setText)
        self.canvas.command.connect(self.command)
        self.canvas.samRequested.connect(self.request_sam)
        self.canvas.samCleared.connect(self.clear_sam_request)
        self.image_list.currentRowChanged.connect(self.choose_image)
        self.class_list.itemClicked.connect(self.assign_selected_tag)
        self.object_list.itemClicked.connect(self.select_object)
        self.class_input.returnPressed.connect(self.add_class)
        self.background.toggled.connect(self.mark_dirty)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.autosave)
        self.timer.start()

    def prepare_sam(self):
        if self.sam_worker and self.sam_worker.isRunning():
            if self.sam_worker.isInterruptionRequested():
                self.sam_status.setText('AI 정리 중… 완료 후 다시 시도하세요.')
                return False
            self.queue_sam_draft()
            return True
        worker = SamWorker(self)
        self.sam_worker = worker
        self.sam_status.setText('AI 준비 중… · 첫 실행은 모델 다운로드와 초기화에 시간이 걸립니다.')
        self.sam_load_button.setEnabled(False)
        worker.loaded.connect(self.sam_loaded)
        worker.result.connect(self.canvas.accept_sam)
        worker.failed.connect(self.sam_failed)
        worker.finished.connect(lambda: self.sam_finished(worker))
        self.queue_sam_draft()
        worker.start()
        self.canvas.setFocus()
        return True

    def sam_loaded(self, device):
        if self.sender() is not self.sam_worker or self.sam_worker.isInterruptionRequested():
            return
        self.sam_status.setText(f'AI 준비됨 · {device} · 같은 이미지의 특징을 재사용합니다.')

    def request_sam(self, revision, points, labels):
        if self.canvas.frame is None:
            return
        if not self.prepare_sam():
            self.canvas.fail_sam(revision, 'AI 정리 후 포함 점을 다시 추가하세요.')

    def queue_sam_draft(self):
        c = self.canvas
        if self.sam_worker and c.sam_points and c.frame is not None:
            self.sam_worker.submit(c.sam_revision, c.image_revision, c.frame, c.sam_points, c.sam_labels,
                                   mask_threshold=self.sam_threshold.value() / 10)

    def change_sam_threshold(self, value):
        self.sam_threshold_value.setText(f'{value / 10:.1f}')
        if self.canvas.sam_points:
            self.canvas._request_sam()

    def reset_sam_threshold(self):
        self.sam_threshold.setValue(0)
        self.canvas.setFocus()

    def clear_sam_request(self):
        if self.sam_worker:
            self.sam_worker.clear_pending()

    def sam_failed(self, revision, error):
        if self.sender() is not self.sam_worker or self.sam_worker.isInterruptionRequested():
            return
        if revision == -1:
            self.sam_status.setText('AI 준비 실패 · AI 준비 버튼으로 재시도하세요: ' + error)
            revision = self.canvas.sam_revision
        self.canvas.fail_sam(revision, error)

    def sam_finished(self, worker):
        if self.sam_worker is worker:
            self.sam_worker = None
            self.sam_load_button.setEnabled(True)
        worker.deleteLater()

    def stop_sam(self):
        if self.sam_worker:
            self.sam_worker.stop()

    def release_sam(self):
        # Releasing the model must not discard either manual geometry or AI prompts.
        # Already computed previews remain available for F; in-flight results expire.
        self.canvas.sam_revision += 1
        self.stop_sam()
        self.sam_status.setText('AI 메모리 해제 · 다음 사용 시 다시 불러옵니다.')
        self.canvas.setFocus()

    def sam_busy(self):
        return bool(self.sam_worker and self.sam_worker.isRunning())

    def closeEvent(self, event):
        self.stop_sam()
        if self.sam_busy():
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
            return
        self.timer.stop()
        super().closeEvent(event)

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
            self.canvas.setFocus()
            return
        self.canvas.doc.selected = item.data(Qt.UserRole)
        self.canvas.setFocus()
        self.canvas.update()

    def delete_object(self):
        self.canvas.setFocus()
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
            self.status.setText(f'{index + 1} / {len(self.project.images)}   ·   {frame.shape[1]} × {frame.shape[0]}   ·   T AI 클릭 / W 직접 그리기')
            self.canvas.setFocus()
        except (ValueError, OSError) as exc:
            self.status.setText(f'{image}: {exc}')

    def save_current(self):
        if not self.project or not self.canvas.doc or self.current_image is None:
            return True
        if self.canvas.pending:
            self.status.setText('이동 전에 F로 도형을 완료하거나 C로 취소하세요.')
            self.canvas.setFocus()
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
        files = choose_file(self, '이미지 가져오기', 'Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)',
                            'import_image', self.project, multiple=True)
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
