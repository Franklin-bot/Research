import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
args = parser.parse_args()

filepath = str(PROJECT_ROOT)
outputs_dir = PROJECT_ROOT / "outputs" / "ml"
results_dir = outputs_dir / "results"
plots_dir = outputs_dir / "plots"
model_dir = outputs_dir / "models" / "Models1"
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

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

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import VAE

VAE.n_samples = n_samples
VAE.filepath = filepath

# Train network
sess = tf.InteractiveSession()

vae_mode = True
vae_mode_modalities = False

vae = VAE.VariationalAutoencoder(
    sess,
    network_param(),
    learning_rate=learning_rate,
    batch_size=batch_size,
    vae_mode=vae_mode,
    vae_mode_modalities=vae_mode_modalities,
)

print("Training")
epoch_list, avg_cost_list, avg_recon_list, avg_latent_list = VAE.train_whole(
    sess,
    vae,
    train_data,
    training_epochs=epochs,
    batch_size=batch_size,
)
vae.cleanup()
sess.close()

with tf.Graph().as_default():
    with tf.Session() as sess:
        network_architecture = network_param()
        print(network_architecture)

        model = VAE.VariationalAutoencoder(
            sess,
            network_architecture,
            batch_size=test_data.shape[0],
            learning_rate=0.00001,
            vae_mode=False,
            vae_mode_modalities=False,
        )

        new_saver = tf.train.Saver()
        new_saver.restore(sess, str(model_dir / "model_1.ckpt"))
        print("Model restored.")

        print("Test 1")
        output_data, x_reconstruct_log_sigma_sq_1 = model.reconstruct(sess, test_data)
        model.cleanup()

np.savetxt(
    results_dir / "output_data.csv",
    output_data,
    delimiter=",",
)
print("Output Data Saved!")

# Quick summary metrics and preview plot
mse_total = float(np.mean((test_data - output_data) ** 2))
mse_kin = float(np.mean((test_data[:, :13] - output_data[:, :13]) ** 2))
mse_imu = float(np.mean((test_data[:, 13:] - output_data[:, 13:]) ** 2))
print(f"MSE total: {mse_total:.8f}")
print(f"MSE kinematics: {mse_kin:.8f}")
print(f"MSE IMU: {mse_imu:.8f}")

pdf_path = plots_dir / args.plot_filename

with PdfPages(pdf_path) as pdf:
    for feature in range(num_features):
        plt.figure()
        plt.plot(output_data[:, feature], label="reconstructed")
        plt.plot(test_data[:, feature], label="original")
        plt.title(f"Feature {feature}")
        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Feature Value")
        pdf.savefig()
        plt.close()

print(f"Plots saved: {pdf_path}")
