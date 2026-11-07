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


if __name__ == '__main__':
    unittest.main()
