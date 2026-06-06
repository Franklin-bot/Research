import tempfile
import unittest
from pathlib import Path

import numpy as np

import model_test


class FakeModel:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float64)
        self.inputs = []
        self.cleaned_up = False

    def reconstruct(self, data):
        self.inputs.append(np.asarray(data, dtype=np.float64))
        return self.output, np.zeros_like(self.output)

    def cleanup(self):
        self.cleaned_up = True


class ModelTestTests(unittest.TestCase):
    def test_evaluate_model_writes_train_style_outputs(self):
        dataset = np.arange(40, dtype=np.float64).reshape(2, 20)
        output = dataset + 1.0
        model = FakeModel(output)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = model_test.evaluate_model(
                model,
                dataset,
                tmpdir,
                eval_mode="masked-kinematics",
                kin_dim=4,
                plot_filename="plots.pdf",
            )

            output_csv = Path(tmpdir) / "output_data.csv"
            summary_txt = Path(tmpdir) / "run_summary.txt"
            metrics_csv = Path(tmpdir) / "metrics.csv"
            plots_pdf = Path(tmpdir) / "plots.pdf"

            self.assertEqual(result["output_csv"], output_csv)
            self.assertEqual(result["summary_txt"], summary_txt)
            self.assertEqual(result["metrics_csv"], metrics_csv)
            self.assertEqual(result["plots_pdf"], plots_pdf)

            np.testing.assert_allclose(np.loadtxt(output_csv, delimiter=","), output)
            self.assertTrue(summary_txt.exists())
            self.assertIn("Evaluation input: masked kinematics input", summary_txt.read_text())
            self.assertTrue(metrics_csv.exists())
            self.assertGreater(plots_pdf.stat().st_size, 0)
            self.assertTrue(np.all(model.inputs[0][:, :4] == model_test.MASK_VALUE))
            self.assertFalse(model.cleaned_up)

    def test_compute_metrics_splits_kinematics_and_imu_halves(self):
        target = np.zeros((2, 6), dtype=np.float64)
        output = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
                [1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
            ]
        )

        metrics = model_test.compute_metrics(target, output, kin_dim=2)

        self.assertEqual(metrics["mse_kin"], 1.0)
        self.assertEqual(metrics["mse_imu_t"], 4.0)
        self.assertEqual(metrics["mse_imu_tminus1"], 16.0)
        self.assertEqual(metrics["mse_imu"], 10.0)


if __name__ == "__main__":
    unittest.main()
