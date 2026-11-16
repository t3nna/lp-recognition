import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FIRST_LETTERS = 'BCDEFGKLNOPRSTWZ'
CODE_LETTERS = 'ABCDEFGHIJKLMNOPRSTUWYZ'
LETTERS_AFTER_CODE = 'ACEFGHJKLMNPRSTUVWXY'
DIGITS = '0123456789'
PATTERNS_FOR_TWO_LETTERS = ['99999', '9999A', '999AA', '9A999', '9AA99']
PATTERNS_FOR_THREE_LETTERS = ['99999', '9999A', '999AA', '9A999', '9AA99', 'A999', '99AA', '9A99', '99A9', '9AA9', 'AA99']
FRAME_WORDS = ['AUTO', 'MOTORS', 'SERWIS', 'KATOWICE', 'TOYOTA', 'GARAGE', 'CENTRUM', 'GLIWICE', 'SKODA', 'TEL 32 123 45 67', 'WWW.AUTO.PL', 'DEALER']


def random_plate_text():
    code = random.choice(FIRST_LETTERS)
    if random.random() < 0.5:
        code += random.choice(CODE_LETTERS)
        pattern = random.choice(PATTERNS_FOR_TWO_LETTERS)
    else:
        code += random.choice(CODE_LETTERS) + random.choice(CODE_LETTERS)
        pattern = random.choice(PATTERNS_FOR_THREE_LETTERS)
    rest = ''
    for symbol in pattern:
        if symbol == '9':
            rest += random.choice(DIGITS)
        else:
            rest += random.choice(LETTERS_AFTER_CODE)
    return code, rest


def draw_eu_band(draw):
    draw.rectangle([0, 0, 50, 113], fill=(0, 51, 153))
    for i in range(12):
        angle = 2 * math.pi * i / 12
        x = 25 + 13 * math.cos(angle)
        y = 30 + 13 * math.sin(angle)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 204, 0))
    small_font = ImageFont.load_default(size=30)
    draw.text((8, 62), 'PL', fill='white', font=small_font)


def draw_plate(code, rest, font_path):
    background = random.randint(200, 255)
    plate = Image.new('RGB', (520, 114), (background, background, background))
    draw = ImageDraw.Draw(plate)
    draw_eu_band(draw)
    draw.rectangle([0, 0, 519, 113], outline=(20, 20, 20), width=4)

    ink = random.randint(0, 60)
    thickness = random.randint(0, 2)
    gap = random.randint(25, 45)
    size = 100
    font = ImageFont.truetype(str(font_path), size)
    code_width = draw.textlength(code, font=font)
    rest_width = draw.textlength(rest, font=font)
    while code_width + gap + rest_width > 440 or font.getbbox(code + rest)[3] - font.getbbox(code + rest)[1] > 85:
        size -= 2
        font = ImageFont.truetype(str(font_path), size)
        code_width = draw.textlength(code, font=font)
        rest_width = draw.textlength(rest, font=font)

    box = font.getbbox(code + rest)
    text_height = box[3] - box[1]
    y = (114 - text_height) / 2 - box[1]
    x = 60 + (450 - code_width - gap - rest_width) / 2
    draw.text((x, y), code, fill=(ink, ink, ink), font=font, stroke_width=thickness, stroke_fill=(ink, ink, ink))
    if random.random() < 0.7:
        center = x + code_width + gap / 2
        draw.ellipse([center - 8, 50, center + 8, 66], fill=(random.randint(150, 230), 180, 180))
    draw.text((x + code_width + gap, y), rest, fill=(ink, ink, ink), font=font, stroke_width=thickness, stroke_fill=(ink, ink, ink))
    return plate


def put_in_frame(plate):
    frame_color = random.randint(0, 255)
    width = 520 + random.randint(20, 60)
    height = 114 + random.randint(20, 40)
    image = Image.new('RGB', (width, height), (frame_color, frame_color, frame_color))
    left = (width - 520) // 2 + random.randint(-5, 5)
    top = (height - 114) // 2 + random.randint(-5, 5)
    image.paste(plate, (left, top))
    if random.random() < 0.5:
        draw = ImageDraw.Draw(image)
        words = random.choice(FRAME_WORDS) + '  ' + random.choice(FRAME_WORDS)
        if frame_color > 128:
            word_color = (0, 0, 0)
        else:
            word_color = (255, 255, 255)
        small_font = ImageFont.load_default(size=random.randint(10, 16))
        draw.text((left + random.randint(30, 200), top + 114 + 1), words, fill=word_color, font=small_font)
    return image


def make_worse(image):
    angle = random.uniform(-4, 4)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(100, 100, 100))
    small_width = random.randint(50, 200)
    image = image.resize((small_width, small_width * image.height // image.width))
    image = image.convert('L').resize((128, 32))
    image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.0)))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.4))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.4, 1.3))
    array = np.asarray(image, dtype=np.float32)
    array = array + np.random.normal(0, random.uniform(0, 12), array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main():
    parser = argparse.ArgumentParser(description='Generate fake Polish license plates for training.')
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--output', type=Path, default=Path('train-data/synthetic'))
    parser.add_argument('--fonts', type=Path, default=Path('models/fonts'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    fonts = sorted(args.fonts.glob('*.ttf'))
    if len(fonts) == 0:
        print('No fonts found. Run python download_fonts.py first.')
        return
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for number in range(args.count):
        code, rest = random_plate_text()
        plate = draw_plate(code, rest, random.choice(fonts))
        image = make_worse(put_in_frame(plate))
        file_name = f'{number:06d}.png'
        image.save(args.output / file_name)
        rows.append([file_name, code + rest])
        if (number + 1) % 5000 == 0:
            print(number + 1, 'plates generated')

    with open(args.output / 'labels.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file', 'text'])
        writer.writerows(rows)
    print('Saved', args.count, 'plates to', args.output)


if __name__ == '__main__':
    main()
