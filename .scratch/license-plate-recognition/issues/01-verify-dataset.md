# 01: Find and verify a usable dataset

**What to build:** Identify and document an appropriate dataset of ordinary single-line Polish or German license plates that can support the student project's detection training and evaluation.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Verify the dataset's source, actual access method, permitted academic use, country and plate-format coverage, annotation types, and relevant limitations; distinguish verified facts from unknowns.
- [x] Document steps for teammates to obtain the data, including any required agreement or author contact, without committing images or redistributing them contrary to their terms.
- [x] Identify how plate text evaluation will be supported, including manual transcription of a small held-out set if necessary.
- [x] If access remains pending or no candidate meets the requirements, document the blocker and viable next steps without claiming the dataset is available.

## Comments

Earlier investigation: ELP 1.0 and THI remain candidates: access, applicable permissions, and adequate Polish or German coverage require confirmation by the owners and inspection of the data. An authorized institutional representative may need to sign the ELP agreement. Those alternatives still require confirmation if used.

The team supplied the Kaggle ELPD; its creator identifies it as synthetic and primarily German-style, and the Kaggle data card states CC BY 4.0. Dataset findings, attribution, preparation issues, access instructions, and remaining evaluation limitations are documented in `docs/dataset-research.md`.
