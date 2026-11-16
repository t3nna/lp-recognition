import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps


def clean_text(text):
    result = ''
    for char in text.upper():
        if char.isalnum():
            result += char
    return result


def main():
    parser = argparse.ArgumentParser(description='Cut license plates out of the Katowice photos.')
    parser.add_argument('--source', type=Path, default=Path('train-data/katowice'))
    parser.add_argument('--output', type=Path, default=Path('train-data/katowice-crops'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tree = ET.parse(args.source / 'annotations.xml')
    plates = []
    for image in tree.getroot().findall('image'):
        for box in image.findall('box'):
            text = clean_text(box.find('attribute').text or '')
            if text == '':
                print('Skipping plate without text in', image.get('name'))
                continue
            left = float(box.get('xtl'))
            top = float(box.get('ytl'))
            right = float(box.get('xbr'))
            bottom = float(box.get('ybr'))
            plates.append([image.get('name'), left, top, right, bottom, text])

    texts = sorted(set(plate[5] for plate in plates))
    random.Random(args.seed).shuffle(texts)
    test_count = round(len(texts) * 0.4)
    val_count = round(len(texts) * 0.2)
    test_texts = set(texts[:test_count])
    val_texts = set(texts[test_count:test_count + val_count])

    counts = {'train': 0, 'val': 0, 'test': 0}
    rows = {'train': [], 'val': [], 'test': []}
    for split in rows:
        (args.output / split).mkdir(parents=True, exist_ok=True)

    for number, plate in enumerate(plates):
        name, left, top, right, bottom, text = plate
        if text in test_texts:
            split = 'test'
        elif text in val_texts:
            split = 'val'
        else:
            split = 'train'
        image = ImageOps.exif_transpose(Image.open(args.source / 'photos' / 'photos' / name)).convert('RGB')
        margin_x = (right - left) * 0.05
        margin_y = (bottom - top) * 0.1
        box = (max(0, left - margin_x), max(0, top - margin_y),
               min(image.width, right + margin_x), min(image.height, bottom + margin_y))
        crop = image.crop(box)
        file_name = f'{number:04d}_{text}.png'
        crop.save(args.output / split / file_name)
        rows[split].append([file_name, text])
        counts[split] += 1

    for split in rows:
        with open(args.output / split / 'labels.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['file', 'text'])
            writer.writerows(rows[split])

    print('Plates:', counts, 'different plate numbers:', len(texts))


if __name__ == '__main__':
    main()
