#!/usr/bin/env python3
"""Run the desktop workspace: python app.py [--project /path/to/project]."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='ZMQ Segmentation Studio')
    parser.add_argument('--project', help='프로젝트 폴더 또는 project.json')
    args = parser.parse_args()
    try:
        from studio.qt import QtWidgets
        from studio.project import Project
        from studio.window import MainWindow
    except ImportError as exc:
        print(f'{exc}\n설치: python -m pip install -r requirements-ui.txt', file=sys.stderr)
        return 1
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Segmentation Studio')
    window = MainWindow()
    if args.project:
        try:
            window.set_project(Project.open(args.project))
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.critical(window, '프로젝트 열기 실패', str(exc))
            return 1
    window.show()
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == '__main__':
    sys.exit(main())
