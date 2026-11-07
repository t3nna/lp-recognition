# 03: Prepare data and fine-tune the detector

**What to build:** Let a teammate with dataset access prepare a reproducible training split and fine-tune a small pretrained plate detector locally, targeting an M4 Pro laptop with 24 GB of memory.

**Blocked by:** 01 Find and verify a usable dataset; 02 Run a first recognition demonstration.

**Status:** ready-for-agent

- [ ] Document and run data preparation and splitting, keeping test images separate and grouping repeated views of the same vehicle where identifiable.
- [ ] Provide a local fine-tuning workflow and a smaller quick run, without requiring hosted GPUs or committing dataset images.
- [ ] Measure and report training time on the target Mac when available; target under one hour, with a reduced sample or fewer epochs if necessary and an explanation of the accuracy trade-off.
- [ ] Produce usable detector weights and document how teammates can obtain or reproduce them under applicable terms.
- [ ] Verify preparation and training behavior with appropriate lightweight checks and a real training smoke test when data and resources are available.
