"""The training command reports missing prerequisites without starting a run."""
import subprocess
import sys
import unittest


class TrainingCliTests(unittest.TestCase):
    def test_missing_data_configuration_is_reported(self):
        result = subprocess.run([sys.executable, 'train_detector.py', '--data', 'missing-data.yaml', '--quick'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Data configuration not found', result.stderr)


if __name__ == '__main__':
    unittest.main()
