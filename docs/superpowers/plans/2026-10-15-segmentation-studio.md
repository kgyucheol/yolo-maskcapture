# Segmentation Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** ZMQ 수집부터 왼손 단축키 마스크 라벨링, 데이터셋, 학습, PT/ONNX 추론까지 연결한다.

**Architecture:** PySide6 UI와 프로젝트/주석 저장, 마스크 모델, 데이터셋 서비스를 분리한다. 수집과 추론은 제한된 대기열의 작업 스레드, 학습과 export는 별도 프로세스에서 수행한다. UI와 데이터셋은 같은 원본 좌표계와 클래스 매핑을 사용한다.

**Tech Stack:** Python 3.10+, PySide6, NumPy, OpenCV, pyzmq, PyYAML, Ultralytics, ONNX Runtime.

**Spec:** `docs/superpowers/specs/2026-10-15-segmentation-studio-design.md`

## Global Constraints

- 기본 수신 주소는 `tcp://192.168.123.164:55555`다.
- 기본 순서는 `마스크 그리기 → F로 도형 완료 → 숫자 키로 태그 지정`이다.
- 숫자는 현재 선택 객체에 태그를 지정하며 화면 인덱스는 1부터, 저장 ID는 0부터다.
- 텍스트/숫자 입력 중에는 캔버스 단축키를 실행하지 않는다.
- 원본 이미지와 이전 데이터셋을 덮어쓰거나 삭제하지 않는다.
- 실제 연결/학습/모델 검증의 수행 여부를 구분해서 보고한다.

## Review Focus

- 비정상 프로젝트 경로/손상된 마스크: 다른 파일을 덮어쓰지 않고 원인을 알린다. Task 1.
- 클래스 미지정 또는 이미지 크기가 다른 주석: 내보내기에서 오류를 밝히고 조용히 배경으로 변환하지 않는다. Task 2.
- 입력 필드에서 숫자/단축키: 입력을 보존하고 주석을 변경하지 않는다. Task 3.
- 끊긴 ZMQ와 multipart: 예전 프레임을 반복 저장하거나 잘못 디코딩하지 않는다. Task 4.
- 추론 중지/모델 교체: 이전 결과를 표시하지 않고 UI를 계속 사용할 수 있다. Task 5.

## Task 1: 프로젝트와 주석

**Files:** `studio/project.py`, `studio/document.py`, `tests/test_project.py`, `tests/test_document.py`.

**Interfaces:** `Project.create(path)`, `Project.open(path)`, `add_class(name)`, `import_image(path)`, `save_annotation(image, objects, reviewed)`; `AnnotationDocument(shape)` provides `add_mask`, `assign_class`, `undo`, `redo`, `replace_mask`.

- [x] 저장/복원, 이름 충돌, 경로 탈출, 태그 미지정, undo/redo 테스트를 먼저 작성한다.
- [x] `/usr/bin/python3 -m pytest tests/test_project.py tests/test_document.py -q`가 구현 부재로 실패하는지 확인한다.
- [x] 원자적 JSON/PNG 저장과 객체별 마스크, 안정적인 클래스 ID를 구현한다.
- [x] 같은 명령으로 모든 테스트가 통과하는지 확인한다.

## Task 2: 데이터셋

**Files:** `studio/dataset.py`, `tests/test_dataset.py`.

**Interfaces:** `build_dataset(project, val_ratio=0.2, seed=0) -> Path`; 프로젝트 주석에서 YOLO segmentation 스냅샷을 만든다.

- [x] train/val 재현성, 정규화 좌표, 배경/미작업 구분, 미지정 객체, 무손실 원본 보존과 잘못된 크기 검증 테스트를 먼저 작성한다.
- [x] `tests/test_dataset.py`를 실행해 실패를 확인한다.
- [x] 윤곽 추출, 분리 영역/구멍 보고, validation과 staging→rename 생성을 구현한다.
- [x] 같은 테스트와 이전 테스트가 통과하는지 확인한다.

## Task 3: 캔버스와 UI

**Files:** `studio/qt.py`, `studio/theme.py`, `studio/canvas.py`, `studio/labeling.py`, `studio/widgets.py`, `studio/window.py`, `app.py`, `tests/test_ui.py`.

**Interfaces:** `MaskCanvas` emits `changed`/`message`; `LabelingPage(project)` handles annotation autosave and classes; `MainWindow` owns project and all workflow pages.

- [x] offscreen 테스트로 도형 생성→숫자 태그 지정, 선택 객체 변경, 입력 필드 단축키 차단, 이미지 이동 저장을 검증한다.
- [x] 테스트 실패를 확인하고 원본 좌표 캔버스, 폴리곤/사각형/브러시/지우개, 확대/이동을 구현한다.
- [x] 어두운 테마, 단계 탐색, 클래스/객체 패널과 상태/단축키 안내를 연결한다.
- [x] `QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m pytest tests/test_ui.py -q`를 실행한다.

## Task 4: ZMQ 수집

**Files:** `zmq_source.py`, `studio/capture.py`, `tests/test_capture.py`, `tests/test_zmq.py`.

**Interfaces:** `ZmqImageSource.read() -> ndarray|None`; `CaptureWorker` owns source/socket and a latest-frame slot; `CapturePage` exposes latest frame for inference.

- [x] multipart, timeout 재설정, 프레임 신선도, interval 검증과 저장 실패 카운트 테스트를 먼저 실행한다.
- [x] 소켓 소유 스레드, 짧은 timeout, REQ 재설정과 중지 처리를 구현한다.
- [x] 미리보기, 수동/자동 촬영, 설정과 메타데이터 저장을 연결한다.
- [x] 수집 테스트와 전체 suite를 실행한다.

## Task 5: 학습·추론과 실행 문서

**Files:** `studio/jobs.py`, `studio/training.py`, `studio/inference.py`, `studio/prediction.py`, `scripts/train.py`, `scripts/infer_segment.py`, `README.md`, `requirements.txt`, `requirements-ui.txt`.

**Interfaces:** `SegmentationPredictor(path, conf, iou, device).predict(frame)` returns mask visual and object metadata; `JobPage` manages a QProcess; training worker CLI writes a JSON result manifest.

- [x] segmentation-only 모델 검증, 작업별 출력 경로, 실패/중지 구분, 모델 교체 후 결과 무시 테스트를 먼저 실행한다.
- [x] Ultralytics segment 경로로 PT/ONNX 예측과 ONNX export를 구현한다. 추론 어댑터가 ONNX Runtime 의존성 부재를 명확히 알린다.
- [x] 학습 로그/epoch, 데이터셋 생성, PT/ONNX 선택, 이미지/폴더/ZMQ 입력과 저장을 연결한다.
- [x] 실행법, 단축키, 지원 모델 범위와 현재 환경 검증 한계를 README에 기록한다.
- [x] 전체 테스트, compileall, offscreen 실행과 screenshot을 확인한다.
- [x] 전체 코드를 새 검토자로 검토하고 중요한 문제를 회귀 테스트와 함께 수정한다.

## 실행 기록

사용자의 `구현해줘` 요청에 따라 이 세션에서 직접 구현한다. 현재 디렉터리는 git 저장소가 아니므로 worktree/commit 스크립트를 실행할 수 없다. 기존 파일을 보존하며 현재 workspace에서 작업하고, 검증 결과는 이 문서에 기록한다.


## 완료 기록

- Task 1: complete — 프로젝트/주석 저장·복원, 태그 변경, undo/redo, 경로 검증 테스트 통과.
- Task 2: complete — 비파괴 segmentation snapshot, 재현 분할, 배경/미지정 구분 테스트 통과.
- Task 3: complete — 원본 좌표 캔버스, 왼손 키보드 흐름, 입력 포커스, 전체 창 전환과 UI screenshot 확인.
- Task 4: complete — 최신 프레임 슬롯, freshness/간격/저장 실패 정책과 multipart/REQ 재설정 테스트 통과. 소켓 전송 계층은 pyzmq 부재로 대체함.
- Task 5: complete — 학습/내보내기 프로세스, PT/ONNX 어댑터, CLI와 문서 연결. 실제 ML 모델 실행은 의존성 부재로 미검증.
- Final review: 독립 검토자가 P1 2건, P2 2건을 재현. 네 건 모두 RED→GREEN 회귀 테스트와 함께 수정. 미해결 검토 항목 없음.
- UI inspection: 압축되어 높이가 0이 되는 설정 입력과 주요 버튼 대비를 수정하고 offscreen 회귀 테스트 및 screenshot으로 확인.
- Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m pytest -q` → 43 passed.
- Verification: `/usr/bin/python3 -m compileall -q app.py studio scripts tests` → 성공.
- Desktop launch: `/usr/bin/python3 app.py` → 실행 환경에서 `could not connect to display :0`. 소켓과 인증 파일 존재는 확인했으나 데스크톱 접속은 불가. offscreen GUI 검증과 구분한다.
- Ruling: git 저장소가 아니므로 worktree/commit/merge 메뉴는 적용할 수 없으며 현재 workspace에 변경을 보존함.
- Ruling: 현재 환경의 패키지 다운로드가 불가능하므로 PySide6 우선/PyQt5 대체를 지원함. 실제 확인한 GUI는 시스템 PyQt5이며, PySide6·실제 ZMQ·GPU 학습·실제 PT/ONNX 결과는 별도 환경 검증이 필요함.
