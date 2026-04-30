import argparse
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTIONS_DIR = PROJECT_ROOT / "data" / "motions"
ML_DATA_ROOT = PROJECT_ROOT / "data" / "ml"
REFERENCE_IMUS = [
    "humerus_r_imu",
    "humerus_l_imu",
    "ulna_r_imu",
    "ulna_l_imu",
]
MASK_VALUE = -2.0
TRAIN_SPLIT = 0.8


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


def select_reference_imu_columns(df):
    missing = [c for c in REFERENCE_IMUS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing reference IMU columns: {missing}")
    return df.select([pl.col(c) for c in REFERENCE_IMUS])


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


def create_reference_imu_instance(quaternion_data, velocity_data, acceleration_data):
    q = split_many_cols_named(
        select_reference_imu_columns(quaternion_data),
        4,
        ["qw", "qx", "qy", "qz"],
    )
    v = split_many_cols_named(
        select_reference_imu_columns(velocity_data),
        3,
        ["vx", "vy", "vz"],
    )
    a = split_many_cols_named(
        select_reference_imu_columns(acceleration_data),
        3,
        ["ax", "ay", "az"],
    )
    return pl.concat([q, v, a], how="horizontal")


def align_frames(kin_df, imu_df):
    if imu_df.height != kin_df.height:
        min_rows = min(kin_df.height, imu_df.height)
        print(
            "Row mismatch between kinematics and IMU signals: "
            f"kin={kin_df.height}, imu={imu_df.height}. Using first {min_rows} rows."
        )
        kin_df = kin_df.slice(0, min_rows)
        imu_df = imu_df.slice(0, min_rows)
    return kin_df, imu_df


def build_reference_dataset_frame(imu_dir, imu_prefix, kinematics_path):
    quaternion_data, velocity_data, acceleration_data = read_imu_tables(imu_dir, imu_prefix)
    kinematics_data = read_kinematics(kinematics_path)
    kin_df = create_kinematics_instance(kinematics_data)
    imu_df = create_reference_imu_instance(quaternion_data, velocity_data, acceleration_data)
    kin_df, imu_df = align_frames(kin_df, imu_df)

    imu_current = imu_df.slice(1).select([pl.col(c).alias(f"{c}_t") for c in imu_df.columns])
    imu_prev = imu_df.slice(0, imu_df.height - 1).select(
        [pl.col(c).alias(f"{c}_tminus1") for c in imu_df.columns]
    )
    kin_current = kin_df.slice(1)

    return pl.concat([kin_current, imu_current, imu_prev], how="horizontal")


def build_masked_imu_t_dataset(base_df):
    imu_t_cols = [c for c in base_df.columns if c.endswith("_t")]
    masked_imu_t = pl.DataFrame(
        {c: np.full(base_df.height, MASK_VALUE, dtype=np.float64) for c in imu_t_cols}
    )
    masked_df = pl.concat(
        [
            base_df.select([pl.col(c) for c in base_df.columns if c not in imu_t_cols]),
            masked_imu_t,
        ],
        how="horizontal",
    ).select([pl.col(c) for c in base_df.columns])
    return masked_df


def build_training_and_test_sets(imu_dir, imu_prefix, kinematics_path):
    base_df = build_reference_dataset_frame(imu_dir, imu_prefix, kinematics_path)
    total_rows = base_df.height
    train_rows = int(total_rows * TRAIN_SPLIT)
    if train_rows <= 0 or train_rows >= total_rows:
        raise ValueError(
            f"Invalid split for dataset with {total_rows} rows and train ratio {TRAIN_SPLIT}."
        )

    train_original = base_df.slice(0, train_rows)
    test_original = base_df.slice(train_rows, total_rows - train_rows)
    train_masked_imu_t = build_masked_imu_t_dataset(train_original)

    train_inputs = pl.concat([train_original, train_masked_imu_t], how="vertical")
    train_targets = pl.concat([train_original, train_original], how="vertical").select(
        [pl.col(c).alias(f"{c}_target") for c in train_original.columns]
    )
    train_augmented = pl.concat([train_inputs, train_targets], how="horizontal")

    return train_augmented, test_original


def write_outputs(train_df, test_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_data.npy", train_df.to_numpy())
    np.save(output_dir / "test_data.npy", test_df.to_numpy())
    (output_dir / "train_columns.txt").write_text("\n".join(train_df.columns) + "\n")
    (output_dir / "test_columns.txt").write_text("\n".join(test_df.columns) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="syndata",
        help="Subdirectory name under data/motions to format, e.g. syndata or expdata.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where formatted ML datasets will be written. Defaults to data/ml/<data-dir>.",
    )
    args = parser.parse_args()

    motion_dir = MOTIONS_DIR / args.data_dir
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ML_DATA_ROOT / args.data_dir
    )
    train_df, test_df = build_training_and_test_sets(
        motion_dir,
        args.data_dir,
        motion_dir / f"{args.data_dir}.mot",
    )
    write_outputs(train_df, test_df, output_dir)
    print(f"Formatted dataset: {args.data_dir}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Saved train dataset: {output_dir / 'train_data.npy'}")
    print(f"Saved test dataset: {output_dir / 'test_data.npy'}")


if __name__ == "__main__":
    main()
