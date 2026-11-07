# 02: Run a first recognition demonstration

**What to build:** Give a teammate a simple local command-line demonstration that reads a permitted sample photograph, detects a license plate, prints its text, and saves an annotated image using available pretrained components.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Provide a reproducible Python environment with compatible pinned dependencies and initial setup guidance for macOS, Windows, and Linux.
- [x] Run recognition locally on an image with a verified source and permitted use; describe how to obtain any required model assets and sample image.
- [x] Print normalized uppercase text without spaces or separators and save an annotated image; handle multiple plates and no detections.
- [x] Provide useful errors for invalid input and missing assets, and automated checks of observable command-line behavior without requiring network access or expensive training.
- [x] Keep the implementation readable and explain the detector and OCR pipeline for a university audience.

## Comments

The CLI and setup are implemented; a real-model CPU run on a CC BY 4.0 synthetic image detected two plates and returned one text result. A public-domain German plate close-up produced no detections. At that point a successful real-image run had not yet been verified; see the subsequent result below and README for setup and dataset provenance. Windows and Linux installation remain untested.

The user subsequently ran the CLI on a real image and reported one detected plate with text `PO35802` and a saved annotated image. This is a demonstration, not an accuracy evaluation.
