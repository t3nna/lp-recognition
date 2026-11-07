# 03: Prepare data and fine-tune the detector

**What to build:** Let a teammate with dataset access prepare a reproducible training split and fine-tune a small pretrained plate detector locally, targeting an M4 Pro laptop with 24 GB of memory.

**Blocked by:** 01 Find and verify a usable dataset; 02 Run a first recognition demonstration.

**Status:** resolved

- [x] Document and run data preparation and splitting, keeping test images separate and grouping repeated views of the same vehicle where identifiable.
- [x] Provide a local fine-tuning workflow and a smaller quick run, without requiring hosted GPUs or committing dataset images.
- [x] Measure and report training time on the target Mac when available; target under one hour, with a reduced sample or fewer epochs if necessary and an explanation of the accuracy trade-off.
- [x] Produce usable detector weights and document how teammates can obtain or reproduce them under applicable terms.
- [x] Verify preparation and training behavior with appropriate lightweight checks and a real training smoke test when data and resources are available.

## Comments

Prepared 1,601 training, 343 validation and 343 test images with seed 42; excluded 42 images without boxes. The M4 Pro MPS run completed three epochs in 218.8 seconds, excluding Python/library startup, and produced `output/training/full/weights/best.pt`. A 5% one-epoch CPU smoke run took 64.9 seconds. The new weights loaded and detected a plate in the user-supplied real-image example; OCR returned an extra character compared with the previous run. Held-out evaluation and near-duplicate/vehicle grouping remain future work; the source provides no vehicle identifiers. See README for commands and limitations.
