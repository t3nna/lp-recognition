# License plate recognition

A small university project: given an image, locate license plates and read their characters. The first demonstration uses a pretrained YOLOv8 detector and EasyOCR; **the team has not fine-tuned either model yet**. Later work will train and evaluate a detector using the team's synthetic data.

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

The command prints the recognized text for each detected plate and saves an image with boxes and text. Empty or unreadable results are possible. Try your own permitted image by replacing the input path. Use `--weights path/to/weights.pt` for a compatible detector. The program does not overwrite its input image.

The detector finds rectangles likely to contain plates; each rectangle is cropped and given to EasyOCR. The program joins the text fragments, converts them to uppercase, and removes spaces and punctuation. Both models run on CPU. A local smoke test on an M4 Pro detected two plates in the example image, read one as `BBXC1533`, and marked the other unreadable; a separate public-domain [close-up photograph](https://commons.wikimedia.org/wiki/File:DE_car_registration_plate.jpg) produced no detections. The user later reported a successful run on a real image, returning `PO35802` and saving an annotated image. These examples are not an accuracy measurement. Detection and OCR errors are expected, especially on blurry, small, unusual, or real-world images. No accuracy has been measured yet.

See [dataset research](docs/dataset-research.md) for data provenance, preparation problems, and evaluation limitations. The local dataset exports do not yet have usable training/validation/test splits or plate-text labels; training is a later ticket.

## Run checks

```sh
python -m unittest discover -s tests -v
```

These checks simulate recognition outputs and do not download weights or measure model accuracy.
