import os
import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from DatasetLogging import build_eval_layout_summary, build_training_layout_summary

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
    "--run-id",
    default=None,
    help=(
        "Optional run identifier used for the output directory name under "
        "data/outputs/<data-dir>. Defaults to a timestamp."
    ),
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
parser.add_argument(
    "--batch-size",
    type=int,
    default=int(os.environ.get("TRAIN_BATCH_SIZE", "12000")),
    help="Global training batch size.",
)
parser.add_argument(
    "--distribution",
    choices=["auto", "none", "mirrored"],
    default=os.environ.get("TRAIN_DISTRIBUTION", "auto"),
    help=(
        "Training distribution strategy. 'auto' uses MirroredStrategy when more than "
        "one GPU is visible; 'none' always uses a single device."
    ),
)
parser.add_argument(
    "--eval-mode",
    choices=["clean", "masked-kinematics", "masked-imu-t"],
    default="masked-kinematics",
    help=(
        "Evaluation input mode. 'masked-kinematics' matches the current full-dataset "
        "evaluation scheme; 'clean' reconstructs the full input; 'masked-imu-t' "
        "evaluates the IMU(t) imputation task."
    ),
)
args = parser.parse_args()

run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
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
TEST_COLUMNS_PATH = ML_DATA_DIR / "test_columns.txt"

if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        "Formatted datasets not found under data/ml. "
        "Run src/ML/FormatDataset.py first."
    )

train_data = np.load(TRAIN_DATA_PATH)
test_data = np.load(TEST_DATA_PATH)
test_columns = None
if TEST_COLUMNS_PATH.exists():
    test_columns = [
        column.strip()
        for column in TEST_COLUMNS_PATH.read_text().splitlines()
        if column.strip()
    ]

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

test_csv_header = ""
if test_columns is not None:
    if len(test_columns) != test_data.shape[1]:
        print(
            "Warning: test_columns.txt column count does not match test_data; "
            "writing test_data.csv without a header."
        )
    else:
        test_csv_header = ",".join(test_columns)

np.savetxt(
    results_dir / "test_data.csv",
    test_data,
    delimiter=",",
    header=test_csv_header,
    comments="",
)
print(f"Test data CSV saved: {results_dir / 'test_data.csv'}")

training_layout_summary = build_training_layout_summary(
    train_data,
    input_dim=input_dim,
    kin_dim=kin_dim,
    mask_value=MASK_VALUE,
)
print(training_layout_summary)


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


def mask_kinematics_features(data, kin_dim, mask_value=MASK_VALUE):
    masked = np.array(data, copy=True)
    masked[:, :kin_dim] = mask_value
    return masked


epochs = args.epochs
learning_rate = 0.0001
batch_size = args.batch_size
batch_size = min(batch_size, n_samples)


def _format_seconds(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def render_summary_page(summary_lines):
    figure = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    return figure


def render_training_curve_page(epoch_list, curve_values, curve_label):
    figure = plt.figure(figsize=(10, 5))
    if epoch_list:
        plt.plot(epoch_list, curve_values, label=curve_label)
        plt.legend()
    else:
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "No training curve points were recorded.",
            ha="center",
            va="center",
            fontsize=14,
        )
    return figure


def render_feature_page(feature, output_data, test_data):
    figure = plt.figure()
    plt.plot(output_data[:, feature], label="reconstructed")
    plt.plot(test_data[:, feature], label="original")
    plt.title(f"Feature {feature}")
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    return figure


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
import tensorflow as tf


def build_strategy(distribution_mode):
    visible_gpus = tf.config.list_physical_devices("GPU")
    for gpu in visible_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if distribution_mode == "none":
        return None, visible_gpus

    if distribution_mode == "mirrored":
        if not visible_gpus:
            print("No GPUs visible. Falling back to single-device execution.")
            return None, visible_gpus
        return tf.distribute.MirroredStrategy(), visible_gpus

    if len(visible_gpus) > 1:
        return tf.distribute.MirroredStrategy(), visible_gpus

    return None, visible_gpus


np.random.seed(args.seed)
tf.random.set_seed(args.seed)
print(f"Seed: {args.seed}")

strategy, visible_gpus = build_strategy(args.distribution)
print(f"Visible GPUs: {len(visible_gpus)}")
if strategy is None:
    print("Distribution strategy: single-device")
else:
    print(
        "Distribution strategy: MirroredStrategy "
        f"({strategy.num_replicas_in_sync} replicas)"
    )
    print(
        "Per-replica batch size: "
        f"{batch_size // strategy.num_replicas_in_sync}"
    )

vae_mode = True
vae_mode_modalities = False

train_scope = strategy.scope() if strategy is not None else nullcontext()
with train_scope:
    vae = VAE.VariationalAutoencoder(
        network_param(),
        learning_rate=learning_rate,
        batch_size=batch_size,
        vae_mode=vae_mode,
        vae_mode_modalities=vae_mode_modalities,
        seed=args.seed,
        strategy=strategy,
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

training_curves = np.column_stack(
    [
        np.asarray(epoch_list, dtype=np.float64),
        np.asarray(avg_cost_list, dtype=np.float64),
        np.asarray(avg_recon_list, dtype=np.float64),
        np.asarray(avg_latent_list, dtype=np.float64),
    ]
)
np.savetxt(
    results_dir / "training_curves.csv",
    training_curves,
    delimiter=",",
    header="epoch,cost,recon,latent",
    comments="",
)

network_architecture = network_param()
print(network_architecture)

infer_scope = strategy.scope() if strategy is not None else nullcontext()
with infer_scope:
    model = VAE.VariationalAutoencoder(
        network_architecture,
        batch_size=test_data.shape[0],
        learning_rate=0.00001,
        vae_mode=False,
        vae_mode_modalities=False,
        seed=args.seed,
        strategy=strategy,
    )
model.load_checkpoint(str(checkpoint_prefix))
print("Model restored.")

print("Test 1")
masked_test_data = mask_imu_t_features(test_data, kin_dim, imu_dim)
masked_kinematics_test_data = mask_kinematics_features(test_data, kin_dim)
if args.eval_mode == "clean":
    eval_input = test_data
    eval_label = "clean reconstruction input"
elif args.eval_mode == "masked-kinematics":
    eval_input = masked_kinematics_test_data
    eval_label = "masked kinematics input"
else:
    eval_input = masked_test_data
    eval_label = "masked imu_t input"
eval_layout_summary = build_eval_layout_summary(
    args.eval_mode,
    input_dim=input_dim,
    kin_dim=kin_dim,
    imu_dim=imu_dim,
    mask_value=MASK_VALUE,
)
print(eval_layout_summary)
output_data, x_reconstruct_log_sigma_sq_1 = model.reconstruct(eval_input)
model.cleanup()

np.savetxt(
    results_dir / "output_data.csv",
    output_data,
    delimiter=",",
)
print("Output Data Saved!")

# Quick summary metrics and preview plot
imu_t_end = kin_dim + (imu_dim // 2)
sqerr_total = (test_data - output_data) ** 2
sqerr_kin = (test_data[:, :kin_dim] - output_data[:, :kin_dim]) ** 2
sqerr_imu = (test_data[:, kin_dim:] - output_data[:, kin_dim:]) ** 2
sqerr_imu_t = (test_data[:, kin_dim:imu_t_end] - output_data[:, kin_dim:imu_t_end]) ** 2
sqerr_imu_tminus1 = (test_data[:, imu_t_end:] - output_data[:, imu_t_end:]) ** 2

mse_total = float(np.mean(sqerr_total))
mse_kin = float(np.mean(sqerr_kin))
mse_imu = float(np.mean(sqerr_imu))
mse_imu_t = float(np.mean(sqerr_imu_t))
mse_imu_tminus1 = float(np.mean(sqerr_imu_tminus1))

std_total = float(np.std(sqerr_total))
std_kin = float(np.std(sqerr_kin))
std_imu = float(np.std(sqerr_imu))
std_imu_t = float(np.std(sqerr_imu_t))
std_imu_tminus1 = float(np.std(sqerr_imu_tminus1))

print(f"MSE total: {mse_total:.8f}")
print(f"MSE kinematics: {mse_kin:.8f}")
print(f"MSE IMU: {mse_imu:.8f}")
print(f"MSE IMU(t): {mse_imu_t:.8f}")
print(f"MSE IMU(t-1): {mse_imu_tminus1:.8f}")
print(f"STD total: {std_total:.8f}")
print(f"STD kinematics: {std_kin:.8f}")
print(f"STD IMU: {std_imu:.8f}")
print(f"STD IMU(t): {std_imu_t:.8f}")
print(f"STD IMU(t-1): {std_imu_tminus1:.8f}")

pdf_path = plots_dir / args.plot_filename

summary_lines = [
    "Run Summary",
    "",
    f"Dataset: {args.data_dir}",
    f"Run ID: {run_id}",
    f"Train shape: {train_shape}",
    f"Test shape: {test_shape}",
    "",
    training_layout_summary,
    "",
    eval_layout_summary,
    "",
    f"Evaluation input: {eval_label}, clean target",
    "",
    f"MSE total: {mse_total:.8f}",
    f"MSE kinematics: {mse_kin:.8f}",
    f"MSE IMU: {mse_imu:.8f}",
    f"MSE IMU(t): {mse_imu_t:.8f}",
    f"MSE IMU(t-1): {mse_imu_tminus1:.8f}",
    f"STD total: {std_total:.8f}",
    f"STD kinematics: {std_kin:.8f}",
    f"STD IMU: {std_imu:.8f}",
    f"STD IMU(t): {std_imu_t:.8f}",
    f"STD IMU(t-1): {std_imu_tminus1:.8f}",
    "",
    f"Results CSV: {results_dir / 'output_data.csv'}",
    f"Checkpoint prefix: {checkpoint_prefix}",
]

with PdfPages(pdf_path) as pdf:
    summary_figure = render_summary_page(summary_lines)
    pdf.savefig(summary_figure)
    plt.close(summary_figure)

    cost_figure = render_training_curve_page(
        epoch_list,
        avg_cost_list,
        "avg_cost",
    )
    pdf.savefig(cost_figure)
    plt.close(cost_figure)

    recon_figure = render_training_curve_page(
        epoch_list,
        avg_recon_list,
        "avg_recon",
    )
    pdf.savefig(recon_figure)
    plt.close(recon_figure)

    latent_figure = render_training_curve_page(
        epoch_list,
        avg_latent_list,
        "avg_latent",
    )
    pdf.savefig(latent_figure)
    plt.close(latent_figure)

    for feature in range(num_features):
        feature_figure = render_feature_page(feature, output_data, test_data)
        pdf.savefig(feature_figure)
        plt.close(feature_figure)

(results_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n")

print(f"Plots saved: {pdf_path}")
print(f"Results saved: {results_dir / 'output_data.csv'}")
print(f"Training curves saved: {results_dir / 'training_curves.csv'}")
print(f"Model checkpoint prefix: {checkpoint_prefix}")
