import shutil
import urllib.request
from pathlib import Path

FONTS = [
    'robotocondensed/RobotoCondensed%5Bwght%5D.ttf',
    'barlowcondensed/BarlowCondensed-Medium.ttf',
    'oswald/Oswald%5Bwght%5D.ttf',
    'archivonarrow/ArchivoNarrow%5Bwght%5D.ttf',
    'sairacondensed/SairaCondensed-Medium.ttf',
    'asapcondensed/AsapCondensed-Medium.ttf',
    'ptsansnarrow/PT_Sans-Narrow-Web-Regular.ttf',
    'sofiasanscondensed/SofiaSansCondensed%5Bwght%5D.ttf',
]


def main():
    folder = Path('models/fonts')
    folder.mkdir(parents=True, exist_ok=True)
    for font in FONTS:
        name = font.split('/')[1].replace('%5B', '-').replace('%5D', '')
        print('Downloading', name)
        urllib.request.urlretrieve('https://github.com/google/fonts/raw/main/ofl/' + font, folder / name)
    windows_font = Path('C:/Windows/Fonts/bahnschrift.ttf')
    if windows_font.exists():
        shutil.copyfile(windows_font, folder / 'bahnschrift.ttf')
        print('Copied Bahnschrift from Windows')
    print('Fonts saved to', folder)


if __name__ == '__main__':
    main()
