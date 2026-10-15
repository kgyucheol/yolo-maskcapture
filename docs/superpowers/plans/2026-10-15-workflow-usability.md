# Workflow usability implementation plan

> Execute inline using superpowers:executing-plans. Review independently before applying.

**Goal:** Improve training progress readability, file capture, canvas navigation, dataset comparisons and file dialog locations without interrupting the user's running training.

**Architecture:** Work in a source-only temporary copy because the workspace is not a Git repository. Keep existing ML jobs lazy and preserve original data/models. Apply verified source changes only after the original app and its job processes have exited. Train progress uses structured epoch timing; text decoding retains partial UTF-8 sequences and normalizes terminal output. File media shares the capture/prediction frame interface. File dialog roles remember directories per project.

**Tech stack:** Python 3.10, Qt, OpenCV, existing Ultralytics worker process.

## Constraints and review focus

- No termination/restart of the current app or training; no real dataset training during tests.
- Unicode split at every byte boundary, carriage-return progress lines and process completion flushing must display readable logs.
- ETA cannot claim accuracy before the first completed epoch and must stop updating at success/failure/cancellation; early stopping is an upper-bound estimate.
- Source switches/EOF/pause must prevent old frames being automatically saved or used for stream prediction. File paths may contain Korean.
- Zoom percentages represent actual image pixels (scale 1 = 100%); reset centers the image without modifying annotations.
- Dataset comparisons must use the same preview frames already used by dataset generation; larger viewing must not resample saved files.
- Every picker should have an existing absolute start directory, clear project-relative defaults, separate remembered roles and harmless cancellation.

## Tasks

1. [x] Training: tests for split UTF-8/ANSI/carriage returns and ETA lifecycle; implement decoder/log parser + epoch times + live ETA timer; narrow settings and expand log area.
2. [x] Media capture: tests with real PNG/MP4 and failed opens, EOF, pause/resume/seek/source switch; implement file source and playback controls with existing capture hotkeys.
3. [x] Canvas: tests for wheel toast and exact centered fit; implement transient zoom readout and reset navigation state.
4. [x] Dataset: test comparison has a larger dedicated area and full-window comparison; move generation report behind its own tab, preserve frames and resize behavior.
5. [x] File dialogs: tests for role defaults, remembered selections and cancellation; implement helper and integrate every picker.
6. [x] Full suite, compile, screenshots and independent review. Wait for original app shutdown, verify baseline hashes, then apply only changed source/tests/docs. No app auto-restart during training.

## Execution notes

- Baseline is 125 passing tests. Original app PID 54146.
- Ruling: A source-only copy substitutes for a worktree because this workspace has no Git metadata; user data and weights remain exclusively in the original workspace.
- Verification: 139 tests passed; compileall and offscreen screenshots passed. Independent review found seek-keyboard handling and repeated-media identity bugs; both reproduced in failing tests then fixed. Re-review found no remaining material issue.
- Ruling: Replace the default two-small-images layout with a full-width draggable before/after comparison, retain optional adaptive two-image mode and larger dialog.
- Applied after the original app and worker exited. Baseline hashes were checked before copying; verification in the original workspace passed all 139 tests.
