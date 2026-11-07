# 06: Make the project reproducible for the team

**What to build:** Give teammates on macOS, Windows, and Linux clear instructions to set up, obtain permitted data and model assets, train, run recognition, and evaluate the project for their university report.

**Blocked by:** 03 Prepare data and fine-tune the detector; 05 Evaluate detection and complete plate text.

**Status:** ready-for-agent

- [ ] Document compatible Python versions, pinned dependencies, virtual-environment setup, and platform-specific differences, prioritizing the target Mac.
- [ ] Provide a coherent walkthrough covering access and licensing, data preparation, quick and full training, inference, and held-out evaluation.
- [ ] Document measured training time and evaluation results when available, together with dataset coverage, limitations, and reproducibility caveats.
- [ ] Verify the documented workflow as far as available platforms, data permissions, and compute allow, and identify any unverified steps.
- [ ] Keep generated data, model weights, and output artifacts manageable for repository sharing; leave remote publication to the owner.
