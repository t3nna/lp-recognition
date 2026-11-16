import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i]
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[len(b)]


def clean_text(text):
    result = ''
    for char in text.upper():
        if char.isalnum():
            result += char
    return result


def make_easyocr_reader():
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='models/easyocr', download_enabled=False)

    def read(image):
        fragments = reader.readtext(np.asarray(image), detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return ''.join(fragments)

    return read


def make_crnn_reader(weights):
    from ocr_model import load_model, read_plate
    model = load_model(weights)

    def read(image):
        return read_plate(model, image)

    return read


def main():
    parser = argparse.ArgumentParser(description='Measure how well plate crops are read.')
    parser.add_argument('--data', type=Path, default=Path('train-data/katowice-crops/test'))
    parser.add_argument('--model', choices=['easyocr', 'crnn'], default='crnn')
    parser.add_argument('--weights', type=Path, default=Path('output/ocr/crnn.pt'), help='CRNN weights')
    parser.add_argument('--show-errors', action='store_true')
    args = parser.parse_args()

    if args.model == 'easyocr':
        read = make_easyocr_reader()
    else:
        read = make_crnn_reader(args.weights)

    with open(args.data / 'labels.csv', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    correct = 0
    wrong_chars = 0
    all_chars = 0
    for row in rows:
        image = Image.open(args.data / row['file']).convert('RGB')
        prediction = clean_text(read(image))
        if prediction == row['text']:
            correct += 1
        elif args.show_errors:
            print(f'{row["file"]}: expected {row["text"]}, got {prediction}')
        wrong_chars += edit_distance(prediction, row['text'])
        all_chars += len(row['text'])

    print(f'Plates: {len(rows)}')
    print(f'Whole plate correct: {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%')
    print(f'Character error rate (CER): {100 * wrong_chars / all_chars:.1f}%')


if __name__ == '__main__':
    main()
