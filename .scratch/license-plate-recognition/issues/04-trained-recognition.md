# 04: Recognize plates with the fine-tuned detector

**What to build:** Let a user run the command-line program on a still photograph using the fine-tuned detector and pretrained OCR, then inspect recognized text and an annotated image.

**Blocked by:** 03 Prepare data and fine-tune the detector.

**Status:** resolved

- [x] Load the fine-tuned detector and pretrained OCR locally using documented model assets.
- [x] Detect and read multiple reasonably visible plates, print uppercase text without spaces or separators, and save an image showing plate locations and recognized text.
- [x] Handle no detections, invalid images, and unavailable weights with useful behavior or messages.
- [x] Verify observable command-line behavior with deterministic checks and perform a documented real-model smoke test when assets are available.
- [x] Explain supported formats and known limitations without claiming unverified country coverage.

## Comments

The CLI accepts the fine-tuned detector through `--weights` and uses pretrained EasyOCR. A CPU smoke test on the user-provided real image produced an annotated image and detected a plate; OCR returned one extra character compared with the previous pretrained-detector run. Automated CLI checks cover multiple crops, normalization, missing and invalid inputs, missing weights, and no detections. See README for model paths, supported inputs, and limitations.
