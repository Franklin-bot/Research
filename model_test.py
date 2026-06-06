import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ML_SRC_DIR = PROJECT_ROOT / "src" / "ML"
MODEL_BASENAME = "model_1"
MASK_VALUE = -2.0

if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from DatasetLogging import build_eval_layout_summary  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the testing/evaluation portion of src/ML/Train.py."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model checkpoint prefix, checkpoint .index file, or directory "
            f"containing {MODEL_BASENAME}.index."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a test .npy file or a directory containing test_data.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_test_outputs",
        help="Directory where output CSVs, summary, metrics, and plots are written.",
    )
    parser.add_argument(
        "--plot-filename",
        default="model_test.pdf",
        help="PDF filename written under --output-dir.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["clean", "masked-kinematics", "masked-imu-t"],
        default="masked-kinematics",
        help="Evaluation input mode matching src/ML/Train.py.",
    )
    parser.add_argument(
        "--kin-dim",
        type=int,
        default=13,
        help="Number of kinematics features at the start of each sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when restoring a checkpoint path.",
    )
    return parser.parse_args()


def load_dataset(dataset):
    if isinstance(dataset, np.ndarray):
        return np.asarray(dataset, dtype=np.float64)

    dataset_path = Path(dataset)
    if dataset_path.is_dir():
        dataset_path = dataset_path / "test_data.npy"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return np.load(dataset_path)


def resolve_model_checkpoint(model_path):
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / MODEL_BASENAME
    elif model_path.suffix == ".index":
        model_path = model_path.with_suffix("")

    if not Path(f"{model_path}.index").exists():
        raise FileNotFoundError(
            f"Model checkpoint index not found: {model_path}.index"
        )

    return model_path


def network_param(input_dim, kin_dim):
    imu_dim = input_dim - kin_dim
    return {
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


def mask_imu_t_features(data, kin_dim, imu_dim, mask_value=MASK_VALUE):
    masked = np.array(data, copy=True)
    imu_t_end = kin_dim + (imu_dim // 2)
    masked[:, kin_dim:imu_t_end] = mask_value
    return masked


def mask_kinematics_features(data, kin_dim, mask_value=MASK_VALUE):
    masked = np.array(data, copy=True)
    masked[:, :kin_dim] = mask_value
    return masked


def build_eval_input(data, eval_mode, kin_dim):
    imu_dim = data.shape[1] - kin_dim

    if eval_mode == "clean":
        return data, "clean reconstruction input"
    if eval_mode == "masked-kinematics":
        return mask_kinematics_features(data, kin_dim), "masked kinematics input"
    if eval_mode == "masked-imu-t":
        return mask_imu_t_features(data, kin_dim, imu_dim), "masked imu_t input"

    raise ValueError(f"Unsupported eval mode: {eval_mode}")


def compute_metrics(target_data, output_data, kin_dim):
    imu_dim = target_data.shape[1] - kin_dim
    imu_t_end = kin_dim + (imu_dim // 2)
    sqerr_total = (target_data - output_data) ** 2
    sqerr_kin = (target_data[:, :kin_dim] - output_data[:, :kin_dim]) ** 2
    sqerr_imu = (target_data[:, kin_dim:] - output_data[:, kin_dim:]) ** 2
    sqerr_imu_t = (
        target_data[:, kin_dim:imu_t_end] - output_data[:, kin_dim:imu_t_end]
    ) ** 2
    sqerr_imu_tminus1 = (
        target_data[:, imu_t_end:] - output_data[:, imu_t_end:]
    ) ** 2

    return {
        "mse_total": float(np.mean(sqerr_total)),
        "mse_kin": float(np.mean(sqerr_kin)),
        "mse_imu": float(np.mean(sqerr_imu)),
        "mse_imu_t": float(np.mean(sqerr_imu_t)),
        "mse_imu_tminus1": float(np.mean(sqerr_imu_tminus1)),
        "std_total": float(np.std(sqerr_total)),
        "std_kin": float(np.std(sqerr_kin)),
        "std_imu": float(np.std(sqerr_imu)),
        "std_imu_t": float(np.std(sqerr_imu_t)),
        "std_imu_tminus1": float(np.std(sqerr_imu_tminus1)),
    }


def format_metrics(metrics):
    return [
        f"MSE total: {metrics['mse_total']:.8f}",
        f"MSE kinematics: {metrics['mse_kin']:.8f}",
        f"MSE IMU: {metrics['mse_imu']:.8f}",
        f"MSE IMU(t): {metrics['mse_imu_t']:.8f}",
        f"MSE IMU(t-1): {metrics['mse_imu_tminus1']:.8f}",
        f"STD total: {metrics['std_total']:.8f}",
        f"STD kinematics: {metrics['std_kin']:.8f}",
        f"STD IMU: {metrics['std_imu']:.8f}",
        f"STD IMU(t): {metrics['std_imu_t']:.8f}",
        f"STD IMU(t-1): {metrics['std_imu_tminus1']:.8f}",
    ]


def render_summary_page(summary_lines):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def render_feature_page(feature, output_data, target_data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.plot(output_data[:, feature], label="reconstructed")
    plt.plot(target_data[:, feature], label="original")
    plt.title(f"Feature {feature}")
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    return figure


def write_plots(pdf_path, summary_lines, output_data, target_data):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            summary_figure = render_summary_page(summary_lines)
            pdf.savefig(summary_figure)
            plt.close(summary_figure)

            for feature in range(target_data.shape[1]):
                feature_figure = render_feature_page(feature, output_data, target_data)
                pdf.savefig(feature_figure)
                plt.close(feature_figure)
    except Exception as exc:
        write_basic_pdf_plots(pdf_path, summary_lines, output_data, target_data, exc)


def _pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_lines(lines, x, y, size=10, leading=14):
    commands = ["BT", f"/F1 {size} Tf", f"{x} {y} Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append(f"0 -{leading} Td")
        commands.append(f"({_pdf_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands)


def _plot_polyline(
    values,
    x_min,
    x_max,
    y_min,
    y_max,
    color,
    value_min=None,
    value_max=None,
):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return ""

    value_min = float(np.min(values) if value_min is None else value_min)
    value_max = float(np.max(values) if value_max is None else value_max)
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0

    span_x = max(values.size - 1, 1)
    span_y = value_max - value_min
    points = []
    for index, value in enumerate(values):
        x = x_min + (index / span_x) * (x_max - x_min)
        y = y_min + ((float(value) - value_min) / span_y) * (y_max - y_min)
        points.append((x, y))

    commands = [color, "1.2 w", f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
    commands.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
    commands.append("S")
    return "\n".join(commands)


def _basic_plot_page(feature, output_data, target_data):
    output_values = output_data[:, feature]
    target_values = target_data[:, feature]
    all_values = np.concatenate([output_values, target_values])
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0

    x_min, x_max = 72, 540
    y_min, y_max = 120, 680
    header = _text_lines(
        [
            f"Feature {feature}",
            "Blue: reconstructed",
            "Black: original",
        ],
        72,
        740,
        size=12,
        leading=16,
    )
    axes = "\n".join(
        [
            "0 0 0 RG",
            "0.8 w",
            f"{x_min} {y_min} m {x_max} {y_min} l S",
            f"{x_min} {y_min} m {x_min} {y_max} l S",
            _text_lines([f"min {value_min:.4g}", f"max {value_max:.4g}"], 72, 92),
        ]
    )
    target_line = _plot_polyline(
        target_values,
        x_min,
        x_max,
        y_min,
        y_max,
        "0 0 0 RG",
        value_min,
        value_max,
    )
    output_line = _plot_polyline(
        output_values,
        x_min,
        x_max,
        y_min,
        y_max,
        "0 0.2 0.85 RG",
        value_min,
        value_max,
    )
    return "\n".join([header, axes, target_line, output_line])


def write_basic_pdf_plots(
    pdf_path,
    summary_lines,
    output_data,
    target_data,
    fallback_error,
):
    pages = [
        _text_lines(
            [
                *summary_lines,
                "",
                "Matplotlib rendering failed; this PDF was generated with the basic fallback renderer.",
                f"Fallback reason: {fallback_error}",
            ],
            36,
            756,
            size=9,
            leading=12,
        )
    ]
    pages.extend(
        _basic_plot_page(feature, output_data, target_data)
        for feature in range(target_data.shape[1])
    )
    _write_pdf(pdf_path, pages)


def _write_pdf(pdf_path, pages):
    objects = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + page_index * 2} 0 R" for page_index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")

    for page_index, page_content in enumerate(pages):
        page_obj = 3 + page_index * 2
        content_obj = page_obj + 1
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {3 + len(pages) * 2} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        encoded_content = page_content.encode("latin-1", errors="replace")
        objects.append(
            f"<< /Length {len(encoded_content)} >>\nstream\n"
            f"{encoded_content.decode('latin-1')}\nendstream"
        )

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("latin-1")
    )
    Path(pdf_path).write_bytes(output)


def write_metrics_csv(metrics_csv, metrics):
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def load_model_from_checkpoint(model_path, input_dim, kin_dim, batch_size, seed):
    import tensorflow as tf
    import VAE

    checkpoint_prefix = resolve_model_checkpoint(model_path)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = VAE.VariationalAutoencoder(
        network_param(input_dim, kin_dim),
        batch_size=batch_size,
        learning_rate=0.00001,
        vae_mode=False,
        vae_mode_modalities=False,
        seed=seed,
    )
    model.load_checkpoint(str(checkpoint_prefix))
    return model, checkpoint_prefix


def evaluate_model(
    model,
    dataset,
    output_dir,
    *,
    eval_mode="masked-kinematics",
    kin_dim=13,
    plot_filename="model_test.pdf",
    seed=42,
):
    test_data = load_dataset(dataset)
    if test_data.ndim != 2:
        raise ValueError(f"Expected a 2D test dataset, got shape {test_data.shape}.")

    input_dim = test_data.shape[1]
    imu_dim = input_dim - kin_dim
    if imu_dim <= 0:
        raise ValueError(
            f"kin_dim ({kin_dim}) must be smaller than input dim ({input_dim})."
        )

    created_model = False
    checkpoint_prefix = None
    if isinstance(model, (str, Path)):
        model, checkpoint_prefix = load_model_from_checkpoint(
            model,
            input_dim=input_dim,
            kin_dim=kin_dim,
            batch_size=test_data.shape[0],
            seed=seed,
        )
        created_model = True

    eval_input, eval_label = build_eval_input(test_data, eval_mode, kin_dim)
    eval_layout_summary = build_eval_layout_summary(
        eval_mode,
        input_dim=input_dim,
        kin_dim=kin_dim,
        imu_dim=imu_dim,
        mask_value=MASK_VALUE,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "output_data.csv"
    summary_txt = output_dir / "run_summary.txt"
    metrics_csv = output_dir / "metrics.csv"
    plots_pdf = output_dir / plot_filename

    output_data, _ = model.reconstruct(eval_input)
    np.savetxt(output_csv, output_data, delimiter=",")

    metrics = compute_metrics(test_data, output_data, kin_dim)
    summary_lines = [
        "Run Summary",
        "",
        f"Model checkpoint prefix: {checkpoint_prefix or '<provided model object>'}",
        f"Test shape: {test_data.shape}",
        "",
        eval_layout_summary,
        "",
        f"Evaluation input: {eval_label}, clean target",
        "",
        *format_metrics(metrics),
        "",
        f"Results CSV: {output_csv}",
        f"Metrics CSV: {metrics_csv}",
        f"Plots PDF: {plots_pdf}",
    ]

    summary_txt.write_text("\n".join(summary_lines) + "\n")
    write_metrics_csv(metrics_csv, metrics)
    write_plots(plots_pdf, summary_lines, output_data, test_data)

    if created_model and hasattr(model, "cleanup"):
        model.cleanup()

    return {
        "output_csv": output_csv,
        "summary_txt": summary_txt,
        "metrics_csv": metrics_csv,
        "plots_pdf": plots_pdf,
        "metrics": metrics,
    }


def main():
    args = parse_args()
    result = evaluate_model(
        args.model,
        args.dataset,
        args.output_dir,
        eval_mode=args.eval_mode,
        kin_dim=args.kin_dim,
        plot_filename=args.plot_filename,
        seed=args.seed,
    )
    print(f"Results saved: {result['output_csv']}")
    print(f"Metrics saved: {result['metrics_csv']}")
    print(f"Summary saved: {result['summary_txt']}")
    print(f"Plots saved: {result['plots_pdf']}")


if __name__ == "__main__":
    main()
