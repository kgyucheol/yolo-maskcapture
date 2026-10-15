import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json

import pytest

from studio.qt import QtWidgets


def test_split_utf8_and_terminal_updates_preserve_korean():
    from studio.training_progress import TrainingLog
    stream = TrainingLog()
    payload = ('경로: /용기공급/학습\n\x1b[32m진행 10%\x1b[0m\r진행 20%\r진행 100%\r\n'
               'STUDIO_PROGRESS {"epoch": 1, "epochs": 10}\n완료').encode('utf-8')
    lines = []
    for value in payload:
        completed, current = stream.feed(bytes([value]))
        lines.extend(completed)
    completed, current = stream.feed(b'', final=True)
    lines.extend(completed)
    assert lines == ['경로: /용기공급/학습', '진행 100%',
                     'STUDIO_PROGRESS {"epoch": 1, "epochs": 10}', '완료']
    assert not current


def test_eta_uses_completed_epoch_durations_and_counts_down():
    from studio.training_progress import TrainingETA
    eta = TrainingETA()
    eta.start(10, now=0)
    assert eta.remaining(now=3) is None
    eta.update(1, 10, 20, now=30)
    assert eta.remaining(now=30) == 180
    assert eta.remaining(now=35) == 175
    eta.update(2, 10, 10, now=40)
    assert eta.remaining(now=40) == 120
    eta.finish()
    assert eta.remaining(now=50) is None


def test_training_page_log_and_eta_consume_structured_progress():
    from studio.training import TrainingPage
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = TrainingPage()
    page.eta.start(10)
    encoded = ('한글 학습 경로\n' + 'STUDIO_PROGRESS ' + json.dumps(
        dict(epoch=1, epochs=10, epoch_seconds=2)) + '\n').encode('utf-8')
    for value in encoded:
        page.consume_output(bytes([value]))
    assert '한글 학습 경로' in page.log.toPlainText()
    assert '\ufffd' not in page.log.toPlainText()
    assert 'STUDIO_PROGRESS' not in page.log.toPlainText()
    assert page.progress.value() == 1
    assert '예상 남은 시간' in page.eta_label.text() and '계산' not in page.eta_label.text()
    page.close()
