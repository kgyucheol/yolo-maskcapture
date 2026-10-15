"""Predictable picker locations with per-project, per-purpose history."""
import hashlib
from pathlib import Path

from .qt import QtCore, QtWidgets


class DialogPaths:
    def __init__(self, settings=None, app_root=None):
        self.settings = settings if settings is not None else QtCore.QSettings('SegmentationStudio', 'FileDialogs')
        self.app_root = Path(app_root or Path(__file__).resolve().parent.parent)

    @staticmethod
    def key(role, project):
        scope = hashlib.sha256(str(project.root).encode('utf-8')).hexdigest()[:24] if project and role != 'project' else 'global'
        return f'directories/{scope}/{role}'

    def start(self, role, project=None, current=''):
        candidates = []
        if current:
            path = Path(current).expanduser()
            if path.is_absolute():
                candidates.append(path if path.is_dir() else path.parent)
        remembered = self.settings.value(self.key(role, project), '', type=str)
        if remembered:
            candidates.append(Path(remembered))
        if current and not Path(current).is_absolute():
            path = self.app_root / current
            if path.exists():
                candidates.append(path if path.is_dir() else path.parent)
        if role == 'project' and project:
            candidates.append(project.root.parent)
        elif role == 'initial_model':
            if (self.app_root / 'yolo11s-seg.pt').is_file():
                candidates.append(self.app_root)
            candidates.append(self.app_root / 'models')
        if project:
            subdirectory = {'model': 'runs', 'export_model': 'runs', 'dataset': 'datasets',
                            'media': 'images', 'image': 'images', 'import_image': 'images'}.get(role)
            if subdirectory:
                base = project.root / subdirectory
                if role == 'dataset':
                    datasets = sorted(base.glob('*/data.yaml'))
                    if datasets:
                        candidates.append(datasets[-1].parent)
                candidates.append(base)
            if role != 'project':
                candidates.append(project.root)
        if role in ('model', 'export_model'):
            candidates.append(self.app_root / 'models')
        candidates.append(self.app_root)
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate.resolve())
        return str(Path.home())

    def remember(self, role, project, selection, directory=False):
        if selection:
            path = Path(selection).resolve()
            folder = path if directory else path.parent
            if folder.is_dir():
                self.settings.setValue(self.key(role, project), str(folder))


paths = DialogPaths()


def choose_file(parent, caption, filters, role, project=None, current='', multiple=False):
    directory = paths.start(role, project, current)
    picker = QtWidgets.QFileDialog.getOpenFileNames if multiple else QtWidgets.QFileDialog.getOpenFileName
    selected, _ = picker(parent, caption, directory, filters)
    if selected:
        paths.remember(role, project, selected[0] if multiple else selected)
    return selected


def choose_directory(parent, caption, role, project=None, current=''):
    selected = QtWidgets.QFileDialog.getExistingDirectory(parent, caption, paths.start(role, project, current))
    paths.remember(role, project, selected, directory=True)
    return selected
