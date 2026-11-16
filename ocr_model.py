import numpy as np
import torch
from torch import nn

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),
        )
        self.rnn = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, len(ALPHABET) + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def image_to_tensor(image):
    image = image.convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.tensor(array).unsqueeze(0)


def text_to_numbers(text):
    numbers = []
    for char in text:
        numbers.append(ALPHABET.index(char) + 1)
    return numbers


def numbers_to_text(numbers):
    text = ''
    previous = 0
    for number in numbers:
        if number != 0 and number != previous:
            text += ALPHABET[number - 1]
        previous = number
    return text


def load_model(path):
    model = CRNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def read_plate(model, image):
    x = image_to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
    numbers = output[0].argmax(1).tolist()
    return numbers_to_text(numbers)
