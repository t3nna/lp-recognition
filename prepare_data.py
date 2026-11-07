"""Prepare a reproducible YOLO dataset from the supplied COCO export."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile


def prepare(source, output, seed):
    annotations_path = source / 'COCO' / 'annotations' / 'instances_Train.json'
    images_dir = source / 'COCO' / 'images' / 'Train'
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new directory.')
    data = json.loads(annotations_path.read_text(encoding='utf-8'))
    by_image = defaultdict(list)
    for annotation in data['annotations']:
        if annotation['category_id'] != 1:
            raise ValueError('Unexpected annotation category; inspect the dataset first.')
        by_image[annotation['image_id']].append(annotation['bbox'])

    groups = defaultdict(list)
    image_ids, stems = set(), set()
    skipped = 0
    for image in data['images']:
        name = image['file_name']
        if Path(name).name != name:
            raise ValueError(f'Unexpected image filename: {name}')
        if image['id'] in image_ids or Path(name).stem in stems:
            raise ValueError(f'Duplicate image ID or filename stem: {name}')
        image_ids.add(image['id'])
        stems.add(Path(name).stem)
        boxes = by_image[image['id']]
        if not boxes:
            # Missing boxes do not prove that the image has no visible plates.
            skipped += 1
            continue
        width, height = image['width'], image['height']
        for x, y, w, h in boxes:
            if not all(math.isfinite(v) for v in (width, height, x, y, w, h)) or width <= 0 or height <= 0 or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f'Invalid plate bounding box in {name}')
        file = images_dir / name
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        groups[digest].append((image, boxes, file))
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 3:
        raise ValueError('At least three distinct annotated images are required.')
    n_test = max(1, round(len(ordered) * 0.15))
    n_val = max(1, round(len(ordered) * 0.15))
    allocations = {'test': ordered[:n_test], 'val': ordered[n_test:n_test+n_val], 'train': ordered[n_test+n_val:]}
    counts = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a private temporary sibling so an interrupted copy never looks complete.
    with tempfile.TemporaryDirectory(prefix='.preparing-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        for split, batches in allocations.items():
            image_out, label_out = staging / 'images' / split, staging / 'labels' / split
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            counts[split] = 0
            for batch in batches:
                for image, boxes, file in batch:
                    width, height = image['width'], image['height']
                    lines = [f'0 {(x+w/2)/width:.6f} {(y+h/2)/height:.6f} {w/width:.6f} {h/height:.6f}' for x, y, w, h in boxes]
                    shutil.copy2(file, image_out / file.name)
                    (label_out / f'{file.stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
                    counts[split] += 1
        (staging / 'data.yaml').write_text(
            f'path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license-plate\n', encoding='utf-8')
        manifest = {'seed': seed, 'source': str(source.resolve()), 'counts': counts, 'distinct_image_groups': len(groups), 'excluded_without_boxes': skipped,
                    'grouping': 'SHA-256 of exact image bytes; vehicle identity is not available'}
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if output.exists():
            raise ValueError(f'Output appeared during preparation: {output}')
        staging.rename(output)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('train-data/plates'))
    parser.add_argument('--output', type=Path, default=Path('train-data/prepared'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.seed)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f'Error preparing data: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
