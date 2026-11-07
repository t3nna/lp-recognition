"""Fine-tune the pretrained plate detector on prepared images."""
import argparse
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('train-data/prepared/data.yaml'))
    parser.add_argument('--weights', type=Path, default=Path('models/plate-detector.pt'))
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--quick', action='store_true', help='Run one epoch on 5%% of training images')
    args = parser.parse_args()
    if not args.data.is_file():
        print(f'Data configuration not found: {args.data}. Run python prepare_data.py first.', file=sys.stderr)
        return 1
    if not args.weights.is_file():
        print(f'Detector weights not found: {args.weights}. Run python download_models.py first.', file=sys.stderr)
        return 1
    try:
        import torch
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; choose --device cpu or run where the Mac GPU is accessible.')
        from ultralytics import YOLO
        start = time.monotonic()
        model = YOLO(str(args.weights))
        model.train(data=str(args.data.resolve()), epochs=1 if args.quick else 3,
                    fraction=0.05 if args.quick else 1.0, imgsz=416, batch=8,
                    device=args.device, workers=0, seed=42, amp=False,
                    project='output/training', name='quick' if args.quick else 'full',
                    exist_ok=False, plots=False, verbose=False)
        print(f'Training elapsed: {(time.monotonic() - start):.1f} seconds')
        print(f'Weights: {model.trainer.save_dir / "weights" / "best.pt"}')
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Training failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
