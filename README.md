# License plate recognition

A small university project: given an image, locate license plates and read their characters. The first demonstration uses a pretrained YOLOv8 detector and EasyOCR. The team can fine-tune the detector on the supplied synthetic data; the OCR model remains pretrained.

## Setup

Install Python 3.12 (64-bit), then open a terminal in this directory. The following commands use a virtual environment; activate it in each new terminal.

**macOS / Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python directly: `.venv\Scripts\python.exe -m pip install -r requirements.txt`. Use the same executable for the commands below. Windows and Linux setup are documented but have not yet been tested by this project.

Download the pretrained detector and EasyOCR models once (internet required):

```sh
python download_models.py
```

Weights are saved under `models/` and are not included in the repository. The detector is [yasirfaizahmed/license-plate-object-detection](https://huggingface.co/yasirfaizahmed/license-plate-object-detection), an independently pretrained model marked Apache 2.0. The [Ultralytics](https://github.com/ultralytics/ultralytics) runtime uses AGPL-3.0; [EasyOCR](https://github.com/JaidedAI/EasyOCR) provides pretrained text detection and recognition models. Review the respective terms when sharing the project. Only download model weights from sources you trust: PyTorch weights are loaded as serialized objects.

## Run the demonstration

The dataset is [European License Plate Dataset (ELPD)](https://www.kaggle.com/datasets/tielemarvin/european-license-plate-dataset-elpd) by Marvin Tiele, distributed on Kaggle under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Download and unpack it under `train-data/plates` to use this example. The images are synthetic, primarily German-style, and are not a measure of accuracy on real photographs. Please retain the attribution and indicate any modifications when sharing dataset material.

```sh
python plate_recognition.py train-data/plates/YOLO/images/Train/filename_prefix_00003_.png --output output/demo.png
```

The command accepts still images readable by Pillow, including JPEG and PNG, prints the recognized text for each detected plate, and saves an image with boxes and text. Use a `.png` or `.jpg` extension for the output. Empty or unreadable results are possible. Try your own permitted image by replacing the input path. Use `--weights path/to/weights.pt` for a compatible detector. The program does not overwrite its input image.

The initial target is ordinary single-line passenger-vehicle plates similar to German and Polish plates. Fine-tuning uses synthetic, predominantly German-style data; country coverage and performance on real photos have not been measured. Motorcycle, two-line, temporary and special plates are outside the initial scope.

The detector finds rectangles likely to contain plates; each rectangle is cropped and given to EasyOCR. The program joins the text fragments, converts them to uppercase, and removes spaces and punctuation. Both models run on CPU. A local smoke test on an M4 Pro detected two plates in the example image, read one as `BBXC1533`, and marked the other unreadable; a separate public-domain [close-up photograph](https://commons.wikimedia.org/wiki/File:DE_car_registration_plate.jpg) produced no detections. The user later reported a successful run on a real image, returning `PO35802` and saving an annotated image. These examples are not an accuracy measurement. Detection and OCR errors are expected, especially on blurry, small, unusual, or real-world images. No accuracy has been measured yet.

See [dataset research](docs/dataset-research.md) for data provenance, preparation problems, and evaluation limitations. The local dataset has plate boxes but no plate-text labels; a separate evaluation task will prepare text labels and measure recognition accuracy.

## Prepare data and fine-tune the detector

After downloading the data and pretrained models as above, prepare the splits:

```sh
python prepare_data.py
```

This reads the COCO export under `train-data/plates` and writes a separate `train-data/prepared` directory; original images and annotations are unchanged. It copies images, so allow roughly another 4 GB of disk space. The output must not already exist: for another split, use `--output` with a new path. The generated configuration contains an absolute path, so regenerate it on each teammate's machine rather than copying it to another location. A manifest records counts and the seed. The default seed 42 assigns approximately 70%/15%/15% of exact-image groups to train/validation/test; only the first two are used while fine-tuning. The test set is reserved for later evaluation.

The COCO export repairs one box missing from YOLO. Images without annotations are excluded because the creator says distant or obscured plates can be unlabelled. One unannotated source image was also unreadable during inspection and is among those excluded. Each distinct image is assigned using a SHA-256 hash so byte-identical images stay together. No vehicle or capture-sequence identifiers were supplied, so similar scenes or the same vehicle can still appear in different splits. A visual review and additional grouping of related images would be needed before drawing strong conclusions from validation or test metrics. See the dataset note for details.

Run a short check (one epoch and 5% of the training images):

```sh
python train_detector.py --quick
```

Run the default three-epoch fine-tuning on CPU, or use Apple Silicon's GPU where PyTorch MPS is available:

```sh
python train_detector.py
python train_detector.py --device mps
```

Run **one** of the last two commands, depending on your hardware. Training uses the downloaded pretrained plate detector, 416-pixel images and batch size 8. The quick run is useful for verifying the setup but its small sample and single epoch can reduce accuracy; neither setting guarantees good real-world performance. On the M4 Pro with 24 GB, the three-epoch MPS run reported 218.8 seconds (about 3 minutes 39 seconds, excluding Python/library startup). The one-epoch 5% CPU smoke run reported 64.9 seconds. Actual time varies. MPS may include nondeterministic operations even with a fixed seed, so identical weights are not guaranteed between runs.

Each run prints elapsed time and its output location. Best weights are saved under `output/training/quick*/weights/best.pt` or `output/training/full*/weights/best.pt`; new runs receive a new directory. To try a trained detector:

```sh
python plate_recognition.py train-data/plates/YOLO/images/Train/filename_prefix_00003_.png --weights output/training/full/weights/best.pt --output output/trained-demo.png
```

The CLI uses the original pretrained detector by default; pass `--weights` as shown to use your fine-tuned detector. Replace the weight path if your run printed a different directory. Weights, images, and output are not committed. Teammates can reproduce weights using the documented data and commands; review the dataset attribution and the pretrained model/runtime licenses before distributing derived weights. Validation metrics in the training log are for synthetic validation data and are not a held-out real-world accuracy result. In a smoke test using the fine-tuned weights on the user-provided `train-data/car.png`, the detector found a plate but OCR returned one extra character compared with the earlier pretrained-detector run. The smoke test used `python plate_recognition.py train-data/car.png --weights output/training/full/weights/best.pt --output output/ticket-04-demo.png`.

## Run checks

```sh
python -m unittest discover -s tests -v
```

These checks simulate recognition outputs and do not download weights or measure model accuracy.
