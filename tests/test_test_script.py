import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT_PATH = PROJECT_ROOT / "test.py"


def load_test_script():
    spec = importlib.util.spec_from_file_location("vae_test_script", TEST_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScriptTests(unittest.TestCase):
    def test_loads_dataset_from_file_or_directory(self):
        module = load_test_script()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data = np.arange(12, dtype=np.float64).reshape(3, 4)
            file_path = tmp_path / "custom.npy"
            dir_path = tmp_path / "dataset"
            dir_path.mkdir()
            np.save(file_path, data)
            np.save(dir_path / "test_data.npy", data)

            np.testing.assert_array_equal(module.load_dataset(file_path), data)
            np.testing.assert_array_equal(module.load_dataset(dir_path), data)


if __name__ == "__main__":
    unittest.main()
