import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path

from studio.project import Project
from studio.qt import QtCore, QtWidgets


def test_dialog_roles_have_existing_defaults_and_project_scoped_memory(tmp_path):
    from studio.file_dialogs import DialogPaths
    app_root = tmp_path / 'app'
    app_root.mkdir()
    (app_root / 'yolo11s-seg.pt').touch()
    settings = QtCore.QSettings(str(tmp_path / 'paths.ini'), QtCore.QSettings.IniFormat)
    locations = DialogPaths(settings=settings, app_root=app_root)
    project = Project.create(tmp_path / 'project')
    other = Project.create(tmp_path / 'other')
    assert locations.start('initial_model', project) == str(app_root)
    assert locations.start('model', project) == str(project.root / 'runs')
    assert locations.start('dataset', project) == str(project.root / 'datasets')
    assert locations.start('media', project) == str(project.root / 'images')
    assert locations.start('project', project) == str(project.root.parent)
    selected = tmp_path / '선택한 폴더'
    selected.mkdir()
    model = selected / 'best.pt'
    model.touch()
    locations.remember('model', project, model)
    assert locations.start('model', project) == str(selected)
    assert locations.start('model', other) == str(other.root / 'runs')
    assert locations.start('dataset', project) == str(project.root / 'datasets')
    locations.remember('model', project, '')
    assert locations.start('model', project) == str(selected)


def test_current_model_path_overrides_fallback_and_missing_memory_is_ignored(tmp_path):
    from studio.file_dialogs import DialogPaths
    settings = QtCore.QSettings(str(tmp_path / 'paths.ini'), QtCore.QSettings.IniFormat)
    paths = DialogPaths(settings=settings, app_root=tmp_path)
    project = Project.create(tmp_path / 'project')
    selected = tmp_path / 'deleted'
    selected.mkdir()
    paths.remember('model', project, selected, directory=True)
    selected.rmdir()
    assert paths.start('model', project) == str(project.root / 'runs')
    current = project.root / 'runs' / 'current'
    current.mkdir()
    (current / 'best.pt').touch()
    assert paths.start('model', project, str(current / 'best.pt')) == str(current)


def test_every_file_picker_opens_in_an_existing_absolute_directory(tmp_path, monkeypatch):
    from studio.window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    project = Project.create(tmp_path / 'project')
    window.set_project(project)
    directories = []
    def single(parent, caption, directory, *args):
        directories.append(directory)
        return '', ''
    def multiple(parent, caption, directory, *args):
        directories.append(directory)
        return [], ''
    def folder(parent, caption, directory, *args):
        directories.append(directory)
        return ''
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', single)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileNames', multiple)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', folder)
    window.training.browse_model()
    window.training.browse_data()
    window.training.start_export()
    window.prediction.browse_model()
    window.prediction.browse_source()
    window.prediction.source_mode.setCurrentIndex(1)
    window.prediction.browse_source()
    window.capture.browse_media()
    window.labeling.import_images()
    window.open_project()
    window.new_project()
    assert len(directories) == 10
    assert all(Path(path).is_absolute() and Path(path).is_dir() for path in directories)
    assert directories[1] == str(project.root / 'datasets')
    assert directories[4] == str(project.root / 'images')
    window.close()
