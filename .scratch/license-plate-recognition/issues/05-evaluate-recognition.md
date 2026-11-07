# 05: Evaluate detection and complete plate text

**What to build:** Let the team evaluate the complete recognition pipeline on held-out images and report results that can be explained in a university project.

**Blocked by:** 01 Find and verify a usable dataset; 04 Recognize plates with the fine-tuned detector.

**Status:** ready-for-agent

- [ ] Evaluate detection quality and exact normalized full-plate text accuracy using images excluded from training.
- [ ] Document the detection-to-ground-truth matching rule, metric denominators, test-set size, country coverage, split method, and limitations.
- [ ] Provide a practical way to transcribe a small held-out set if the dataset lacks text labels, documenting provenance and labeling.
- [ ] Check evaluation behavior against small labeled examples, including incorrect text and missed or extra detections.
- [ ] Report measured results only when the required data and trained assets are available; clearly identify missing evidence otherwise.
