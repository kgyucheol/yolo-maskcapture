from .qt import QtGui

COLORS = ['#42d6ba', '#7f9cff', '#ffb85c', '#f483b4', '#c39aff', '#7cd783', '#65c8ee', '#f18572', '#d8d36c']


def class_color(class_id):
    return QtGui.QColor('#a6b0c3' if class_id is None else COLORS[class_id % len(COLORS)])


STYLESHEET = """
QWidget { background: #0e1420; color: #e8edf6; font-family: 'Noto Sans CJK KR', 'Noto Sans', sans-serif; font-size: 13px; }
QMainWindow { background: #0e1420; }
QFrame#card { background: #161f2e; border: 1px solid #263247; border-radius: 12px; }
QLabel { background: transparent; }
QLabel#eyebrow { color: #56d6bd; font-size: 11px; font-weight: 700; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#subtitle { color: #97a6bd; font-size: 12px; }
QLabel#numericUnit { color: #a8b8cd; font-size: 12px; background: transparent; padding: 0 1px; }
QToolButton#helpIcon { color: #8da8c9; background: transparent; border: none; border-radius: 11px; font-size: 16px; }
QToolButton#helpIcon:hover, QToolButton#helpIcon:focus { color: #8ef3d9; background: #23473f; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #69788f; background: #161e2b; border-color: #283246; }
QLabel#section { font-size: 14px; font-weight: 700; color: #e0e8f6; }
QLabel#badge { color: #59d7bd; background: #153a38; border-radius: 7px; padding: 5px 10px; }
QLabel#shortcutKey { color: #d0e5ff; background: #26364b; border: 1px solid #48617b; border-bottom: 3px solid #142035; border-radius: 5px; padding: 3px 7px; font-family: monospace; font-size: 12px; font-weight: 700; }
QLabel#shortcutCaption { color: #d5e0ef; background: transparent; font-size: 12px; }
QLabel#shortcutKey[active="true"] { color: #092720; background: #65e9c9; border-color: #43b59c; }
QLabel#shortcutCaption[active="true"] { color: #91f5d9; }
QLabel#toast { color: #8ef3d9; background: #153a38; border: 1px solid #3e9d8b; border-radius: 9px; padding: 10px 18px; font-weight: 700; }
QLabel#toast[error="true"] { color: #ffb3b3; background: #3a1b22; border-color: #8a3d4a; }
QLabel#notice { color: #abbad0; padding: 9px; background: #172232; border-radius: 7px; }
QPushButton { background: #202c3e; border: 1px solid #334158; border-radius: 7px; padding: 9px 13px; font-weight: 600; }
QPushButton#shortcutTool { padding: 0; }
QPushButton:hover { background: #293b51; border-color: #5c7a96; }
QPushButton:pressed { background: #31465e; }
QPushButton:checked { color: #61e4c7; background: #173b3a; border-color: #3e9d8b; }
QPushButton:disabled { color: #627087; border-color: #283246; background: #182131; }
QPushButton#chip { padding: 6px 2px; font-weight: 600; }
QPushButton#primary { background: #45d4b5; color: #092720; border: none; }
QPushButton#primary:hover { background: #6ee5ca; }
QPushButton#primary:disabled { background: #244f48; color: #789990; }
QPushButton#danger { color: #ffa6a6; border-color: #6e3947; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #101827; border: 1px solid #354158; border-radius: 6px; padding: 7px; selection-background-color: #276e66; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #4dd8ba; }
QComboBox QAbstractItemView { background: #1c293b; selection-background-color: #285450; }
QListWidget { background: #101827; border: 1px solid #263247; border-radius: 8px; outline: none; padding: 5px; }
QListWidget::item { padding: 10px 7px; border-radius: 5px; }
QListWidget::item:selected { background: #23473f; color: #8ef3d9; }
QListWidget::item:hover { background: #1e3040; }
QListWidget#navigation { border: none; background: #111b29; padding: 9px; }
QListWidget#navigation::item { padding: 15px 8px; margin: 3px 0; }
QPlainTextEdit { background: #0b1220; color: #b9c7da; border: 1px solid #263247; border-radius: 8px; padding: 8px; font-family: monospace; font-size: 12px; }
QProgressBar { background: #1c293a; border: none; border-radius: 6px; height: 14px; text-align: center; }
QProgressBar::chunk { background: #35b89c; border-radius: 6px; }
QCheckBox { spacing: 8px; padding: 4px 0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:unchecked { background: #101827; border: 1px solid #64758d; border-radius: 3px; }
QCheckBox::indicator:checked { background: #45d4b5; border: 2px solid #b3ffe9; border-radius: 3px; }
QCheckBox::indicator:disabled { background: #172131; border: 1px solid #344257; }
QGroupBox { border: 1px solid #34465d; border-radius: 8px; margin-top: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8ce7cd; }
QCheckBox#augmentEnable { font-size: 14px; font-weight: 700; color: #8ce7cd; background: #173b3a; border: 1px solid #3e9d8b; border-radius: 7px; padding: 7px 10px; }
QCheckBox#augmentEnable::indicator { width: 20px; height: 20px; }
QTabWidget::pane { border: 1px solid #34465d; border-radius: 6px; background: #111b29; }
QTabBar::tab { background: #202c3e; color: #a9b9cc; padding: 9px 11px; border: 1px solid #34465d; }
QTabBar::tab:selected { color: #8ef3d9; background: #23473f; border-bottom: 2px solid #45d4b5; }
QSlider::groove:horizontal { height: 5px; background: #34465d; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #347d72; border-radius: 2px; }
QSlider::handle:horizontal { background: #65e9c9; border: 2px solid #b1ffe9; width: 13px; margin: -5px 0; border-radius: 7px; }
QScrollArea { background: transparent; border: none; }
QSplitter::handle { background: #253147; width: 3px; height: 3px; }
QStatusBar { background: #111b29; color: #91a4bf; border-top: 1px solid #263247; }
QToolTip { background: #26364a; color: #f0f4fa; border: 1px solid #4f687f; padding: 6px; }
QScrollBar:vertical { background: #131c2a; width: 9px; border: none; }
QScrollBar::handle:vertical { background: #34445a; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
