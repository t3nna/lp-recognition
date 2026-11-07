"""Download the pretrained plate detector and EasyOCR models once."""
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import easyocr


def main():
    target = Path('models/plate-detector.pt')
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = hf_hub_download(
        repo_id='yasirfaizahmed/license-plate-object-detection',
        filename='best.pt',
        revision='9d42447',
    )
    shutil.copyfile(weights, target)
    easyocr.Reader(['en'], gpu=False, model_storage_directory='models/easyocr', download_enabled=True)
    print(f'Detector saved to {target}; EasyOCR models are in models/easyocr.')


if __name__ == '__main__':
    main()
