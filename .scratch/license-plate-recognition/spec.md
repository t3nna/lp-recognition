# License Plate Recognition

Status: ready-for-agent

## Problem Statement

A university team needs a readable, explainable machine-learning project that detects license plates in vehicle photographs and reads their registration characters. Team members use macOS, Windows, and Linux and need a reproducible setup and suitable legally accessible training and evaluation data. They cannot collect their own photographs or use hosted GPU training.

## Solution

Provide a simple Python command-line application, an accessible data preparation and local fine-tuning workflow, and an evaluation workflow. The program detects plates, reads each plate's text, prints normalized results, and saves an annotated image. Explain the choices, setup, data provenance, training, evaluation, and limitations for a university audience.

## User Stories

1. As a student, I want to set up the project on macOS, Windows, or Linux, so that my team can collaborate.
2. As a teammate, I want documented compatible Python and dependency versions, so that our environments are consistent.
3. As a teammate, I want straightforward virtual-environment and installation instructions, so that I can get started.
4. As a teammate, I want to obtain data using documented steps, so that the repository remains small and respects distribution terms.
5. As a student, I want each dataset's provenance and access conditions documented, so that I can assess its suitability.
6. As a student, I want the supported countries and plate formats identified, so that I understand the scope of the results.
7. As a student, I want an academic-use data request documented if needed, so that I can seek permission.
8. As a student, I want a path to run a demonstration while restricted dataset access is pending, so that work can continue.
9. As a student, I want to prepare images and annotations consistently, so that model training can use them.
10. As a student, I want to transcribe a small evaluation set when text labels are missing, so that OCR can be assessed.
11. As a student, I want to fine-tune a small pretrained detector locally, so that I can demonstrate model training.
12. As a Mac user, I want training designed around an M4 Pro with 24 GB memory, so that it is feasible on my laptop.
13. As a student, I want a small quick run, so that I can check the training workflow.
14. As a student, I want training to target under an hour on the target Mac, so that it fits my available time.
15. As a student, I want the time and accuracy trade-off documented, so that I understand the consequences of reducing training.
16. As a student, I want pretrained OCR to read detected crops, so that the complete project remains manageable.
17. As a user, I want to supply a still image through the command line, so that I can run recognition locally.
18. As a user, I want multiple visible plates detected in one image, so that each vehicle can be handled.
19. As a user, I want plate text normalized to uppercase without spaces or separators, so that output is consistent.
20. As a user, I want an annotated image saved, so that I can inspect and demonstrate the results.
21. As a user, I want plate text printed, so that I can see the results without interpreting an image.
22. As a user, I want useful messages for invalid input or unavailable model files, so that I can resolve problems.
23. As a student, I want training and test data separated, so that evaluation is meaningful.
24. As a student, I want repeated views of the same vehicle kept in one split where identifiable, so that evaluation leakage is reduced.
25. As a student, I want plate detection quality measured, so that I can assess localization.
26. As a student, I want exact full-plate recognition accuracy measured on held-out images, so that I can assess end-to-end results.
27. As a student, I want test-set size and country coverage reported, so that I can interpret metrics.
28. As a student, I want readable code and explanations of the pipeline, so that I can present and maintain the project.
29. As a teammate, I want generated datasets, model weights, and outputs handled appropriately, so that sharing the repository is practical.
30. As the project owner, I want to publish the repository myself, so that I control its hosting.

## Implementation Decisions

- The first version targets ordinary single-line passenger-vehicle plates resembling those used in Poland or Germany, whether or not an EU band is visible. Data may cover either country or both; do not claim coverage that has not been verified.
- Inputs are still photographs with plates reasonably readable by a person, potentially containing multiple vehicles.
- Use a small pretrained detector fine-tuned locally and pretrained OCR on detected plate crops. Select tools and models for understandable code, laptop feasibility, and suitable usage terms; specific frameworks are not yet decided.
- Normalize recognized text to uppercase and omit spaces and separators. Display detections and recognized text in a saved annotated image, and print text to the console.
- Provide pinned compatible dependencies and documented environment setup for macOS, Windows, and Linux. Prioritize local training on the specified Mac; do not require hosted GPUs.
- Target a training run below one hour on an M4 Pro with 24 GB RAM. Measure actual runtime and provide a reduced-data or reduced-epoch quick run if needed; accuracy and timing are not guaranteed.
- Investigate and verify public datasets before selection, including actual download access, annotation content, country coverage, and license. Images should be downloaded by teammates rather than committed, subject to terms. Academic agreements or author contact are acceptable. Team members cannot take new photos.
- If detection data lacks text annotations, provide a practical workflow to transcribe a small held-out evaluation set. Document provenance and labeling.
- Keep test images out of training; group repeated vehicle views in the same split when identifiable. Report dataset size, split method, coverage, and relevant caveats.
- Evaluate detection and end-to-end exact normalized plate-text accuracy on the held-out set; explain the matching rule and denominator so results are reproducible.
- Provide a demonstrable path while restricted training data access is pending, using only material whose access and reuse are verified; do not imply an unavailable dataset is bundled.
- Keep implementation simple, readable, and explainable. Document model selection, setup, data, training, inference, evaluation, and limitations.
- Repository publication is the owner's responsibility. Do not execute Git commands or access the Git metadata directory.

## Testing Decisions

- Favor tests of externally observable behavior over tests tied to internal implementation. The primary seam is the command-line workflow: a supplied image and controlled detector/OCR outputs produce normalized console results and an annotated image, including multiple-plate and no-plate cases.
- Exercise invalid inputs and unavailable model assets through the same interface. Small deterministic fixtures or substitutes may isolate the workflow from model downloads and expensive training.
- Verify evaluation behavior with small labeled examples, including correct and incorrect full text and detection matching; avoid leakage between training and held-out data.
- Separately perform a documented real-model smoke test and measure training time when usable model weights and data are available. Do not make network access or a lengthy training run a prerequisite for ordinary automated checks.
- There is no existing application or test suite to use as prior art.

## Out of Scope

- Production deployment, production-grade operations, hosted training, video processing, real-time guarantees, and training large models.
- Guaranteed support for all EU countries or special, temporary, motorcycle, and two-line plates.
- Reliable recognition of unreadable, very small, severely blurred, or night-time plates.
- Collecting new photographs, distributing datasets without permission, or publishing/configuring the remote repository.

## Further Notes

Dataset selection and actual achievable accuracy remain open pending verification. PGuard ELP describes European images with plate locations and text but requires an academic-use access process, and coverage for the desired countries has not been established. A German-heavy cropped-plate collection is not by itself a full-scene detection dataset; its terms also need verification. These leads are candidates, not approved dependencies.
