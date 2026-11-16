"""Detect license plates in one image, then read each detected crop."""
import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


class PlateRecognizer:
    """A small pretrained detector followed by pretrained optical character recognition."""

    def __init__(self, weights, ocr_weights=None):
        from ultralytics import YOLO

        self.detector = YOLO(str(weights))
        self.crnn = None
        self.reader = None
        if ocr_weights is not None:
            from ocr_model import load_model
            self.crnn = load_model(ocr_weights)
        else:
            import easyocr
            # CPU works on macOS, Windows and Linux, without configuring a GPU.
            self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='models/easyocr', download_enabled=False)

    def recognize(self, image):
        import numpy as np

        result = self.detector.predict(image, conf=0.25, device='cpu', verbose=False)[0]
        plates = []
        for box in result.boxes.xyxy.tolist():
            left, top, right, bottom = (int(value) for value in box)
            left, top = max(0, left), max(0, top)
            right, bottom = min(image.width, right), min(image.height, bottom)
            if right <= left or bottom <= top:
                continue
            if self.crnn is not None:
                from ocr_model import read_plate
                margin_x = int((right - left) * 0.05)
                margin_y = int((bottom - top) * 0.1)
                crop = image.crop((max(0, left - margin_x), max(0, top - margin_y),
                                   min(image.width, right + margin_x), min(image.height, bottom + margin_y)))
                plates.append(((left, top, right, bottom), read_plate(self.crnn, crop)))
            else:
                crop = image.crop((left, top, right, bottom))
                fragments = self.reader.readtext(np.asarray(crop), detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                plates.append(((left, top, right, bottom), ''.join(fragments)))
        return plates


def main(argv=None, recognizer_factory=PlateRecognizer):
    parser = argparse.ArgumentParser(description='Find and read license plates in an image.')
    parser.add_argument('image', type=Path, help='Input image')
    parser.add_argument('--output', type=Path, default=Path('output/annotated.png'), help='Annotated image path')
    parser.add_argument('--weights', type=Path, default=Path('models/plate-detector.pt'), help='Plate detector weights')
    parser.add_argument('--ocr-weights', type=Path, help='Our CRNN weights; without this EasyOCR reads the plates')
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise ValueError(f'Input image not found: {args.image}')
        if args.image.resolve() == args.output.resolve() or (args.output.exists() and args.image.samefile(args.output)):
            raise ValueError('Output must be different from the input image.')
        with Image.open(args.image) as input_image:
            image = ImageOps.exif_transpose(input_image).convert('RGB')
        if recognizer_factory is PlateRecognizer and not args.weights.is_file():
            raise ValueError(
                f'Detector weights not found: {args.weights}. Check the --weights path. '
                'For the original pretrained model, run python download_models.py; '
                'for fine-tuned weights, run python train_detector.py.'
            )
        if args.ocr_weights is not None and not args.ocr_weights.is_file():
            raise ValueError(f'CRNN weights not found: {args.ocr_weights}. Run python train_ocr.py first.')
        if args.ocr_weights is None:
            recognizer = recognizer_factory(args.weights)
        else:
            recognizer = recognizer_factory(args.weights, args.ocr_weights)
        detections = recognizer.recognize(image)
        draw = ImageDraw.Draw(image)
        for number, (box, text) in enumerate(detections, start=1):
            normalized = re.sub(r'[^A-Z0-9]', '', text.upper())
            print(f'Plate {number}: {normalized or "[unreadable]"}')
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], max(0, box[1] - 15)), normalized or '[unreadable]', fill='red')
        if not detections:
            print('No license plates detected.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(f'Annotated image saved to {args.output}')
        return 0
    except FileNotFoundError as error:
        print(f'Error: {error}. If OCR models are missing, run python download_models.py.', file=sys.stderr)
        return 1
    except (OSError, ValueError, UnidentifiedImageError, ImportError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
