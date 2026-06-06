# Research

## Run Model Evaluation

Use `model_test.py` when you already have a trained VAE checkpoint and a formatted
test dataset, and you only want to run the testing/evaluation portion of
`src/ML/Train.py`.

```bash
.venv/bin/python model_test.py \
  --model path/to/model_1 \
  --dataset data/ml/syndata/test_data.npy \
  --output-dir model_test_outputs
```

`--model` can be any of:

- a checkpoint prefix, such as `data/outputs/syndata/run_id/models/model_1`
- a checkpoint index file, such as `data/outputs/syndata/run_id/models/model_1.index`
- a model directory containing `model_1.index`

`--dataset` can be either:

- a `.npy` file containing the test data
- a directory containing `test_data.npy`

By default, evaluation uses `--eval-mode masked-kinematics`, which matches the
default testing mode in `src/ML/Train.py`. Other modes are available:

```bash
.venv/bin/python model_test.py \
  --model data/outputs/syndata/run_id/models/model_1 \
  --dataset data/ml/syndata \
  --output-dir model_test_outputs/clean \
  --eval-mode clean
```

Supported evaluation modes:

- `clean`: reconstructs the full clean input
- `masked-kinematics`: masks the first `--kin-dim` kinematics columns
- `masked-imu-t`: masks the IMU(t) half of the IMU columns

Outputs written to `--output-dir`:

- `output_data.csv`: reconstructed model output
- `metrics.csv`: MSE and standard deviation metrics
- `run_summary.txt`: text summary of the run
- `model_test.pdf`: summary page plus per-feature original/reconstructed plots

Optional flags:

```bash
.venv/bin/python model_test.py \
  --model data/outputs/syndata/run_id/models \
  --dataset data/ml/syndata \
  --output-dir model_test_outputs/masked_imu_t \
  --plot-filename masked_imu_t.pdf \
  --eval-mode masked-imu-t \
  --kin-dim 13 \
  --seed 42
```
