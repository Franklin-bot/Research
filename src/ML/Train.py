import os
import argparse
import re
from pathlib import Path
import numpy as np

import polars as pl
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
args = parser.parse_args()

filepath = str(PROJECT_ROOT)
outputs_dir = PROJECT_ROOT / "outputs" / "ml"
results_dir = outputs_dir / "results"
plots_dir = outputs_dir / "plots"
model_dir = outputs_dir / "models" / "Models1"
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

SYN_IMU_DIR = PROJECT_ROOT / "data" / "motions" / "syndata"
EXP_IMU_DIR = PROJECT_ROOT / "data" / "motions" / "expdata"
SYN_IMU_PREFIX = "syndata"
EXP_IMU_PREFIX = "expdata"
SYN_KINEMATICS_PATH = PROJECT_ROOT / "data" / "motions" / "syndata" / "syndata.mot"
EXP_KINEMATICS_PATH = PROJECT_ROOT / "data" / "motions" / "expdata" / "expdata.mot"


def split_many_cols_named(df, n_values, suffixes):
    splits = n_values - 1
    out = []

    for c in df.columns:
        s = pl.col(c).str.split_exact(",", splits)
        for i, suf in enumerate(suffixes):
            out.append(
                s.struct.field(f"field_{i}")
                .cast(pl.Float64)
                .alias(f"{c}_{suf}")
            )

    return df.select(out)


def select_row_columns(df, angle_suffix, row_position, rotations_per_vertical=5, imu_start=2):
    """
    Pick one vertical IMU row when IMUs are indexed with 5 rotational placements per vertical height.
    row_position is the 0-based vertical index (e.g., 0..4 for a 5-row setup).
    """
    angle_cols = [c for c in df.columns if c.endswith(angle_suffix)]
    selected_cols = []
    matched_imu_cols = 0
    for c in angle_cols:
        match = re.match(r"^(.*_imu)(\d+)" + re.escape(angle_suffix) + r"$", c)
        if match is None:
            continue

        imu_index = int(match.group(2))
        vertical_index = (imu_index - imu_start) // rotations_per_vertical
        if vertical_index != row_position:
            continue

        selected_cols.append(c)
        matched_imu_cols += 1

    if matched_imu_cols == 0:
        raise ValueError(
            f"No IMU columns matched row position {row_position} "
            f"for suffix {angle_suffix}."
        )

    # Normalize names across row positions so vertical concat works.
    # Different rows pick different source columns (imu2/imu3/...), but
    # they must map into the same feature schema.
    return df.select(
        [pl.col(c).alias(f"imu_slot_{i:03d}") for i, c in enumerate(selected_cols)]
    )


def read_imu_tables(imu_dir, imu_prefix):
    quaternion_data = pl.read_csv(
        imu_dir / f"{imu_prefix}_orientations.sto",
        separator="\t",
        skip_rows=4,
    ).with_columns(pl.col("time").cast(pl.Float64))

    velocity_data = pl.read_csv(
        imu_dir / f"{imu_prefix}_angular_velocity.sto",
        separator="\t",
        skip_rows=4,
    ).with_columns(pl.col("time").cast(pl.Float64))

    acceleration_data = pl.read_csv(
        imu_dir / f"{imu_prefix}_linear_accelerations.sto",
        separator="\t",
        skip_rows=4,
    ).with_columns(pl.col("time").cast(pl.Float64))

    return quaternion_data, velocity_data, acceleration_data


def read_kinematics(kinematics_path):
    return pl.read_csv(
        kinematics_path,
        separator="\t",
        skip_rows=7,
        ignore_errors=True,
    ).with_columns(pl.col("time").cast(pl.Float64))


def create_kinematics_instance(kinematics_data):
    kinematics_cols = [
        "lumbar_extension",
        "lumbar_bending",
        "lumbar_rotation",
        "arm_flex_r",
        "arm_add_r",
        "arm_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "arm_flex_l",
        "arm_add_l",
        "arm_rot_l",
        "elbow_flex_l",
        "pro_sup_l",
    ]
    return kinematics_data.select([pl.col(c) for c in kinematics_cols])


def create_imu_orientation_instance(quaternion_data, angle_suffix, row_position):
    row_filtered = select_row_columns(
        quaternion_data,
        angle_suffix,
        row_position,
    )
    return split_many_cols_named(row_filtered, 4, ["qw", "qx", "qy", "qz"])


def build_dataset_for_rows(
    imu_dir,
    imu_prefix,
    kinematics_path,
    row_positions,
    angle_suffix="_0deg",
):
    quaternion_data, _, _ = read_imu_tables(imu_dir, imu_prefix)
    kinematics_data = read_kinematics(kinematics_path)
    kin_df = create_kinematics_instance(kinematics_data)

    dataset_frames = []
    for row_pos in row_positions:
        imu_df = create_imu_orientation_instance(quaternion_data, angle_suffix, row_pos)
        if imu_df.height != kin_df.height:
            min_rows = min(kin_df.height, imu_df.height)
            print(
                f"Row mismatch for IMU row {row_pos}: "
                f"kin={kin_df.height}, imu={imu_df.height}. "
                f"Using first {min_rows} rows."
            )
            kin_use = kin_df.slice(0, min_rows)
            imu_use = imu_df.slice(0, min_rows)
        else:
            kin_use = kin_df
            imu_use = imu_df
        dataset_frames.append(pl.concat([kin_use, imu_use], how="horizontal"))

    dataset = pl.concat(dataset_frames, how="vertical")
    return dataset.to_numpy()


# Train/test split from synthetic data by vertical IMU row:
# Row positions are 0..4 within each 5-row group.
# Train uses outer rows 0,1,3,4; test uses middle row 2.
train_data = build_dataset_for_rows(
    SYN_IMU_DIR,
    SYN_IMU_PREFIX,
    SYN_KINEMATICS_PATH,
    [0, 1, 3, 4],
    angle_suffix="_0deg",
)

test_data = build_dataset_for_rows(
    SYN_IMU_DIR,
    SYN_IMU_PREFIX,
    SYN_KINEMATICS_PATH,
    [2],
    angle_suffix="_0deg",
)

print(f"Train Data: {train_data.shape}")
print(f"Test Data: {test_data.shape}")

n_samples = train_data.shape[0]
input_dim = train_data.shape[1]
imu_dim = input_dim - 13
print(f"n_samples: {n_samples}")
print(f"n_input: {input_dim} (kin=13, imu={imu_dim})")

if test_data.shape[1] != input_dim:
    raise ValueError(
        f"Feature mismatch: train={input_dim}, test={test_data.shape[1]}. "
        "Train/test feature dimensions must match."
    )


def network_param():
    network_architecture = {
        "n_input": input_dim,
        "n_z": 100,
        "size_slices": [13, imu_dim],
        "size_slices_shared": [48, 100],
        "mod0": [95, 48],
        "mod1": [200, 100],
        "mod0_2": [48, 13],
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
