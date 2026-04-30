import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_BASENAME = "model_1"
MASK_VALUE = -2.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot-filename",
    required=True,
    help="PDF filename written under outputs/ml/plots",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=int(os.environ.get("TRAIN_EPOCHS", "8000")),
    help="Number of training epochs.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Load and validate datasets, then exit before TensorFlow training.",
)
parser.add_argument(
    "--data-dir",
    default="syndata",
    help="Dataset name under data/ml and source name under data/motions, e.g. syndata or expdata.",
)
parser.add_argument(
    "--formatted-data-dir",
    default=None,
    help="Directory containing formatted train_data.npy and test_data.npy. Defaults to data/ml/<data-dir>.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=int(os.environ.get("TRAIN_SEED", "42")),
    help="Random seed for NumPy and TensorFlow.",
)
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = PROJECT_ROOT / "data" / "outputs" / args.data_dir / run_id
results_dir = outputs_dir / "results"
plots_dir = outputs_dir / "plots"
model_dir = outputs_dir / "models"
checkpoint_prefix = model_dir / MODEL_BASENAME
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
print(f"Run ID: {run_id}")
print(f"Run output directory: {outputs_dir}")

ML_DATA_ROOT = PROJECT_ROOT / "data" / "ml"
ML_DATA_DIR = (
    Path(args.formatted_data_dir)
    if args.formatted_data_dir is not None
    else ML_DATA_ROOT / args.data_dir
)
TRAIN_DATA_PATH = ML_DATA_DIR / "train_data.npy"
TEST_DATA_PATH = ML_DATA_DIR / "test_data.npy"

if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        "Formatted datasets not found under data/ml. "
        "Run src/ML/FormatDataset.py first."
    )

train_data = np.load(TRAIN_DATA_PATH)
test_data = np.load(TEST_DATA_PATH)

print(f"Train Data (augmented paired): {train_data.shape}")
print(f"Test Data: {test_data.shape}")

n_samples = train_data.shape[0]
input_dim = test_data.shape[1]
kin_dim = 13
imu_dim = input_dim - kin_dim
print(f"n_samples: {n_samples}")
print(f"n_input: {input_dim} (kin={kin_dim}, imu={imu_dim})")

if train_data.shape[1] != input_dim * 2:
    raise ValueError(
        f"Expected paired train dataset width {input_dim * 2}, got {train_data.shape[1]}."
    )

if test_data.shape[1] != input_dim:
    raise ValueError(
        f"Feature mismatch: input={input_dim}, test={test_data.shape[1]}. "
        "Test feature dimensions must match the model input."
    )


def network_param():
    network_architecture = {
        "n_input": input_dim,
        "n_z": 100,
        "size_slices": [kin_dim, imu_dim],
        "size_slices_shared": [48, 100],
        "mod0": [95, 48],
        "mod1": [200, 100],
        "mod0_2": [48, kin_dim],
        "mod1_2": [100, imu_dim],
        "enc_shared": [350],
        "dec_shared": [350, 148],
    }
    return network_architecture


def mask_imu_t_features(data, kin_dim, imu_dim, mask_value=MASK_VALUE):
    masked = np.array(data, copy=True)
    imu_t_end = kin_dim + (imu_dim // 2)
    masked[:, kin_dim:imu_t_end] = mask_value
    return masked


epochs = args.epochs
learning_rate = 0.0001
batch_size = 12000
batch_size = min(batch_size, n_samples)


def _format_seconds(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


train_shape = train_data.shape
test_shape = test_data.shape
print(f"Train dataset size (shape): {train_shape}")
print(f"Test dataset size (shape): {test_shape}")

train_steps_per_epoch = int(np.ceil(n_samples / batch_size))
total_train_steps = train_steps_per_epoch * epochs
num_features = test_shape[1]

# Heuristic estimate; actual runtime depends on hardware and TensorFlow performance.
train_time_low_s = total_train_steps * 0.25
train_time_high_s = total_train_steps * 2.0
plot_time_low_s = num_features * 0.02
plot_time_high_s = num_features * 0.10
print(
    "Estimated runtime: "
    f"{_format_seconds(train_time_low_s + plot_time_low_s)} to "
    f"{_format_seconds(train_time_high_s + plot_time_high_s)} "
    f"(train_steps={total_train_steps}, feature_plots={num_features})"
)

if args.dry_run:
    print("Dry run complete. Dataset loading and shape checks passed.")
    raise SystemExit(0)

os.environ["PYTHONHASHSEED"] = str(args.seed)

import VAE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(args.seed)
tf.set_random_seed(args.seed)
print(f"Seed: {args.seed}")

vae_mode = True
vae_mode_modalities = False

vae = VAE.VariationalAutoencoder(
    network_param(),
    learning_rate=learning_rate,
    batch_size=batch_size,
    vae_mode=vae_mode,
    vae_mode_modalities=vae_mode_modalities,
)

print("Training")
epoch_list, avg_cost_list, avg_recon_list, avg_latent_list = VAE.train_whole(
    vae,
    train_data,
    checkpoint_prefix=str(checkpoint_prefix),
    training_epochs=epochs,
    batch_size=batch_size,
)
vae.cleanup()

network_architecture = network_param()
print(network_architecture)

model = VAE.VariationalAutoencoder(
    network_architecture,
    batch_size=test_data.shape[0],
    learning_rate=0.00001,
    vae_mode=False,
    vae_mode_modalities=False,
)
model.load_checkpoint(str(checkpoint_prefix))
print("Model restored.")

print("Test 1")
masked_test_data = mask_imu_t_features(test_data, kin_dim, imu_dim)
output_data, x_reconstruct_log_sigma_sq_1 = model.reconstruct(masked_test_data)
model.cleanup()

np.savetxt(
    results_dir / "output_data.csv",
    output_data,
    delimiter=",",
)
print("Output Data Saved!")

# Quick summary metrics and preview plot
imu_t_end = kin_dim + (imu_dim // 2)
mse_total = float(np.mean((test_data - output_data) ** 2))
mse_kin = float(np.mean((test_data[:, :kin_dim] - output_data[:, :kin_dim]) ** 2))
mse_imu = float(np.mean((test_data[:, kin_dim:] - output_data[:, kin_dim:]) ** 2))
mse_imu_t = float(
    np.mean((test_data[:, kin_dim:imu_t_end] - output_data[:, kin_dim:imu_t_end]) ** 2)
)
mse_imu_tminus1 = float(
    np.mean((test_data[:, imu_t_end:] - output_data[:, imu_t_end:]) ** 2)
)
print(f"MSE total: {mse_total:.8f}")
print(f"MSE kinematics: {mse_kin:.8f}")
print(f"MSE IMU: {mse_imu:.8f}")
print(f"MSE IMU(t): {mse_imu_t:.8f}")
print(f"MSE IMU(t-1): {mse_imu_tminus1:.8f}")

pdf_path = plots_dir / args.plot_filename

with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    summary_lines = [
        "Run Summary",
        "",
        f"Dataset: {args.data_dir}",
        f"Run ID: {run_id}",
        f"Train shape: {train_shape}",
        f"Test shape: {test_shape}",
        "Evaluation input: masked imu_t, clean target",
        "",
        f"MSE total: {mse_total:.8f}",
        f"MSE kinematics: {mse_kin:.8f}",
        f"MSE IMU: {mse_imu:.8f}",
        f"MSE IMU(t): {mse_imu_t:.8f}",
        f"MSE IMU(t-1): {mse_imu_tminus1:.8f}",
        "",
        f"Results CSV: {results_dir / 'output_data.csv'}",
        f"Checkpoint prefix: {checkpoint_prefix}",
    ]
    plt.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    pdf.savefig()
    plt.close()

    for feature in range(num_features):
        plt.figure()
        plt.plot(output_data[:, feature], label="reconstructed")
        plt.plot(test_data[:, feature], label="original")
        plt.plot(masked_test_data[:, feature], label="masked input", linestyle="--", alpha=0.7)
        plt.title(f"Feature {feature}")
        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Feature Value")
        pdf.savefig()
        plt.close()

print(f"Plots saved: {pdf_path}")
print(f"Results saved: {results_dir / 'output_data.csv'}")
print(f"Model checkpoint prefix: {checkpoint_prefix}")
