# 04: Recognize plates with the fine-tuned detector

**What to build:** Let a user run the command-line program on a still photograph using the fine-tuned detector and pretrained OCR, then inspect recognized text and an annotated image.

**Blocked by:** 03 Prepare data and fine-tune the detector.

**Status:** ready-for-agent

- [ ] Load the fine-tuned detector and pretrained OCR locally using documented model assets.
- [ ] Detect and read multiple reasonably visible plates, print uppercase text without spaces or separators, and save an image showing plate locations and recognized text.
- [ ] Handle no detections, invalid images, and unavailable weights with useful behavior or messages.
- [ ] Verify observable command-line behavior with deterministic checks and perform a documented real-model smoke test when assets are available.
- [ ] Explain supported formats and known limitations without claiming unverified country coverage.
