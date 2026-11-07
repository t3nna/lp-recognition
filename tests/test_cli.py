"""Checks of the command-line behavior without downloading model weights."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from plate_recognition import main


class FakeRecognizer:
    def __init__(self, detections):
        self.detections = detections

    def recognize(self, image):
        return self.detections


class CommandLineTests(unittest.TestCase):
    def test_multiple_plates_are_normalized_and_image_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            output = Path(folder) / 'result.png'
            Image.new('RGB', (200, 100), 'white').save(source)
            stdout = io.StringIO()
            recognizer = FakeRecognizer([((10, 10, 100, 40), ' b-ab 123 '), ((20, 50, 120, 80), 'cd 456')])
            with contextlib.redirect_stdout(stdout):
                code = main([str(source), '--output', str(output)], recognizer_factory=lambda _: recognizer)
            self.assertEqual(code, 0)
            self.assertIn('BAB123', stdout.getvalue())
            self.assertIn('CD456', stdout.getvalue())
            self.assertTrue(output.is_file())
            self.assertNotEqual(Image.open(source).tobytes(), Image.open(output).tobytes())

    def test_no_detections_still_saves_the_image(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            output = Path(folder) / 'result.png'
            Image.new('RGB', (40, 30), 'white').save(source)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(source), '--output', str(output)], recognizer_factory=lambda _: FakeRecognizer([]))
            self.assertEqual(code, 0)
            self.assertIn('No license plates detected', stdout.getvalue())
            self.assertTrue(output.exists())

    def test_missing_input_is_reported(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(['does-not-exist.png'])
        self.assertEqual(code, 1)
        self.assertIn('Input image not found', stderr.getvalue())

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            Image.new('RGB', (40, 30)).save(source)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), '--weights', str(Path(folder) / 'missing.pt')])
            self.assertEqual(code, 1)
            self.assertIn('Detector weights not found', stderr.getvalue())

    def test_input_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            Image.new('RGB', (40, 30)).save(source)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), '--output', str(source)])
            self.assertEqual(code, 1)
            self.assertIn('different from the input', stderr.getvalue())

    def test_cli_reports_a_missing_image(self):
        result = subprocess.run([sys.executable, 'plate_recognition.py', 'missing-image.png'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Input image not found', result.stderr)

    def test_hard_link_does_not_overwrite_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            output = Path(folder) / 'linked.png'
            Image.new('RGB', (40, 30)).save(source)
            os.link(source, output)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), '--output', str(output)])
            self.assertEqual(code, 1)
            self.assertIn('different from the input', stderr.getvalue())

    def test_photo_orientation_is_applied_before_recognition(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.jpg'
            output = Path(folder) / 'result.png'
            image = Image.new('RGB', (40, 20), 'white')
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)
            class CheckOrientation:
                def recognize(self, image):
                    self.size = image.size
                    return []
            recognizer = CheckOrientation()
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(source), '--output', str(output)], recognizer_factory=lambda _: recognizer)
            self.assertEqual(code, 0)
            self.assertEqual(recognizer.size, (20, 40))

    def test_corrupt_image_is_reported_through_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'broken.png'
            source.write_text('not an image')
            result = subprocess.run([sys.executable, 'plate_recognition.py', str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Error:', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_missing_custom_weights_are_reported_through_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'car.png'
            Image.new('RGB', (40, 30)).save(source)
            result = subprocess.run([sys.executable, 'plate_recognition.py', str(source), '--weights', str(Path(folder) / 'missing.pt')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Detector weights not found', result.stderr)
            self.assertIn('Check the --weights path', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_cli_runs_detector_and_ocr_for_each_plate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, weights = root / 'car.png', root / 'out.png', root / 'model.pt'
            Image.new('RGB', (200, 100), 'white').save(source)
            weights.touch()
            (root / 'ultralytics.py').write_text("""
class Coordinates:
    def tolist(self): return [[10, 10, 100, 40], [20, 50, 120, 80]]
class Boxes:
    xyxy = Coordinates()
class Result:
    boxes = Boxes()
class YOLO:
    def __init__(self, path): pass
    def predict(self, image, **kwargs): return [Result()]
""")
            (root / 'easyocr.py').write_text("""
class Reader:
    def __init__(self, *args, **kwargs): pass
    def readtext(self, image, **kwargs):
        if image.shape[:2] == (30, 90): return ['ab 123']
        if image.shape[:2] == (30, 100): return ['cd 456']
        raise ValueError('Unexpected crop size')
""")
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
            result = subprocess.run([sys.executable, 'plate_recognition.py', str(source), '--weights', str(weights), '--output', str(output)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Plate 1: AB123', result.stdout)
            self.assertIn('Plate 2: CD456', result.stdout)
            self.assertTrue(output.is_file())


if __name__ == '__main__':
    unittest.main()
