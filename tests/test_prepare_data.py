"""Preparation is exercised through its command-line interface."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class PrepareDataTests(unittest.TestCase):
    def test_prepares_reproducible_splits_without_separating_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'source' / 'COCO' / 'images' / 'Train'
            images.mkdir(parents=True)
            annotations = root / 'source' / 'COCO' / 'annotations'
            annotations.mkdir(parents=True)
            records, boxes = [], []
            for i in range(12):
                name = f'{i}.png'
                # Two names contain the same image and annotation.
                color = (i if i != 11 else 0, 0, 0)
                Image.new('RGB', (100, 50), color).save(images / name)
                records.append({'id': i, 'file_name': name, 'width': 100, 'height': 50})
                boxes.append({'id': i, 'image_id': i, 'category_id': 1, 'bbox': [10, 5, 40, 20]})
            (annotations / 'instances_Train.json').write_text(json.dumps({'images': records, 'annotations': boxes}))
            out = root / 'prepared'
            result = subprocess.run([sys.executable, 'prepare_data.py', '--source', str(root / 'source'), '--output', str(out), '--seed', '42'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            groups = {split: {p.stem for p in (out / 'images' / split).glob('*.png')} for split in ('train', 'val', 'test')}
            self.assertEqual(sum(map(len, groups.values())), 12)
            self.assertTrue(all(groups.values()))
            self.assertTrue(any({'0', '11'} <= group for group in groups.values()))
            label = next((out / 'labels').glob('*/0.txt')).read_text().strip()
            self.assertEqual(label, '0 0.300000 0.300000 0.400000 0.400000')
            self.assertTrue((out / 'data.yaml').is_file())
            second = root / 'prepared-again'
            retry = subprocess.run([sys.executable, 'prepare_data.py', '--source', str(root / 'source'), '--output', str(second), '--seed', '42'], text=True, capture_output=True)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            second_groups = {split: {p.stem for p in (second / 'images' / split).glob('*.png')} for split in ('train', 'val', 'test')}
            self.assertEqual(groups, second_groups)
            existing = subprocess.run([sys.executable, 'prepare_data.py', '--source', str(root / 'source'), '--output', str(out)], text=True, capture_output=True)
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn('already exists', existing.stderr)

    def test_invalid_boxes_do_not_leave_a_partial_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source' / 'COCO'
            images = source / 'images' / 'Train'
            annotations = source / 'annotations'
            images.mkdir(parents=True)
            annotations.mkdir()
            records, boxes = [], []
            for i in range(3):
                name = f'{i}.png'
                Image.new('RGB', (100, 50), (i, 0, 0)).save(images / name)
                records.append({'id': i, 'file_name': name, 'width': 100, 'height': 50})
                boxes.append({'id': i, 'image_id': i, 'category_id': 1, 'bbox': [float('nan') if i == 2 else 10, 5, 40, 20]})
            (annotations / 'instances_Train.json').write_text(json.dumps({'images': records, 'annotations': boxes}))
            output = root / 'prepared'
            result = subprocess.run([sys.executable, 'prepare_data.py', '--source', str(root / 'source'), '--output', str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid plate bounding box', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
