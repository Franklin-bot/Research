import numpy as np


def build_training_layout_summary(train_data, input_dim, kin_dim, mask_value):
    rows, cols = train_data.shape
    lines = [
        "Training dataset layout:",
        f"  shape: {rows} rows x {cols} columns",
        f"  columns 0:{input_dim} -> model input",
        f"  columns {input_dim}:{input_dim * 2} -> clean target",
    ]

    if cols != input_dim * 2:
        lines.append("  paired input/target diagnostics: skipped (unexpected width)")
        return "\n".join(lines)

    if rows % 2 != 0:
        lines.append("  two-case diagnostics: skipped (odd number of rows)")
        return "\n".join(lines)

    case_rows = rows // 2
    inputs = train_data[:, :input_dim]
    targets = train_data[:, input_dim:]
    case1_input = inputs[:case_rows]
    case2_input = inputs[case_rows:]
    case1_target = targets[:case_rows]
    case2_target = targets[case_rows:]

    case1_input_equals_target = np.allclose(case1_input, case1_target)
    case2_kinematics_masked = np.all(case2_input[:, :kin_dim] == mask_value)
    case2_non_kinematics_match = np.allclose(
        case2_input[:, kin_dim:],
        case1_input[:, kin_dim:],
    )
    case2_target_equals_case1_target = np.allclose(case2_target, case1_target)

    lines.extend(
        [
            f"  case 1 rows 0:{case_rows} -> complete input",
            f"  case 2 rows {case_rows}:{rows} -> kinematics-masked input",
            f"  case 1 input equals target: {case1_input_equals_target}",
            f"  case 2 kinematics columns all {mask_value}: {case2_kinematics_masked}",
            (
                "  case 2 non-kinematics columns match case 1: "
                f"{case2_non_kinematics_match}"
            ),
            f"  case 2 target equals case 1 target: {case2_target_equals_case1_target}",
        ]
    )
    return "\n".join(lines)


def build_eval_layout_summary(eval_mode, input_dim, kin_dim, imu_dim, mask_value):
    lines = [
        "Evaluation dataset layout:",
        f"  clean test target columns: 0:{input_dim}",
    ]

    if eval_mode == "clean":
        lines.append("  eval input: clean test_data")
    elif eval_mode == "masked-kinematics":
        lines.append(
            f"  eval input: test_data with kinematics columns 0:{kin_dim} set to {mask_value}"
        )
    elif eval_mode == "masked-imu-t":
        imu_t_end = kin_dim + (imu_dim // 2)
        lines.append(
            f"  eval input: test_data with IMU(t) columns {kin_dim}:{imu_t_end} "
            f"set to {mask_value}"
        )
    elif eval_mode == "masked-kinematics-and-imus-by-location":
        lines.extend(
            [
                (
                    "  eval input: five runs with kinematics columns "
                    f"0:{kin_dim} set to {mask_value}"
                ),
                (
                    "  each run masks all IMU columns except one retained "
                    "virtual IMU location set"
                ),
            ]
        )
    else:
        lines.append(f"  eval input: unknown mode {eval_mode}")

    return "\n".join(lines)
