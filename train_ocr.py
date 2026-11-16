import argparse
import csv
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ocr_model import CRNN, IMAGE_HEIGHT, IMAGE_WIDTH, image_to_tensor, numbers_to_text, text_to_numbers


class PlateDataset(Dataset):
    def __init__(self, folder, augment):
        self.images = []
        self.texts = []
        self.augment = augment
        with open(folder / 'labels.csv', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                image = Image.open(folder / row['file']).convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                self.images.append(image)
                self.texts.append(row['text'])
        print(f'Loaded {len(self.images)} plates from {folder}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            image = image.rotate(random.uniform(-3, 3), resample=Image.BILINEAR, fillcolor=100)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        return image_to_tensor(image), self.texts[index]


def collate(batch):
    images = []
    targets = []
    lengths = []
    texts = []
    for image, text in batch:
        numbers = text_to_numbers(text)
        images.append(image)
        targets += numbers
        lengths.append(len(numbers))
        texts.append(text)
    return torch.stack(images), torch.tensor(targets), torch.tensor(lengths), texts


def check_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, lengths, texts in loader:
            output = model(images)
            predictions = output.argmax(2)
            for i in range(len(texts)):
                if numbers_to_text(predictions[i].tolist()) == texts[i]:
                    correct += 1
                total += 1
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Train the CRNN plate reader.')
    parser.add_argument('--train', type=Path, nargs='+', default=[Path('train-data/synthetic')])
    parser.add_argument('--val', type=Path, default=Path('train-data/katowice-crops/val'))
    parser.add_argument('--weights', type=Path, help='Start from these weights instead of from scratch')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--augment', action='store_true', help='Randomly change brightness and rotation while training')
    parser.add_argument('--output', type=Path, default=Path('output/ocr/crnn.pt'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_sets = []
    for folder in args.train:
        train_sets.append(PlateDataset(folder, args.augment))
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PlateDataset(args.val, False), batch_size=args.batch_size, collate_fn=collate)

    model = CRNN()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for images, targets, lengths, texts in train_loader:
            output = model(images)
            log_probs = output.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
            loss = loss_function(log_probs, targets, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_accuracy = check_accuracy(model, val_loader)
        print(f'Epoch {epoch}: loss {train_loss:.3f}, val accuracy {100 * val_accuracy:.1f}%, time {time.time() - start:.0f} s')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output)
            print('  saved best model to', args.output)

    print(f'Best val accuracy: {100 * best_accuracy:.1f}%')


if __name__ == '__main__':
    main()
