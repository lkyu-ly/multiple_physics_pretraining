import argparse
from pathlib import Path

import numpy as np


def load_output_array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)

    payload = np.load(path)
    try:
        if "output" in payload.files:
            return payload["output"]
        if len(payload.files) == 1:
            return payload[payload.files[0]]
        raise ValueError(
            f"Ambiguous npz content in {path}: expected key 'output' or a single array, got {payload.files}"
        )
    finally:
        payload.close()


def compute_error_metrics(torch_output, paddle_output):
    torch_output = np.asarray(torch_output, dtype=np.float64)
    paddle_output = np.asarray(paddle_output, dtype=np.float64)

    if torch_output.shape != paddle_output.shape:
        raise ValueError(
            f"Shape mismatch: torch {torch_output.shape} vs paddle {paddle_output.shape}"
        )

    abs_error = np.abs(torch_output - paddle_output)
    denom = np.maximum(np.abs(torch_output), 1e-12)
    rel_error = abs_error / denom

    return {
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((torch_output - paddle_output) ** 2))),
        "max_rel_error": float(np.max(rel_error)),
        "mean_rel_error": float(np.mean(rel_error)),
    }


def format_metrics(metrics):
    ordered_keys = [
        "max_abs_error",
        "mean_abs_error",
        "rmse",
        "max_rel_error",
        "mean_rel_error",
    ]
    return "\n".join(f"{key}: {metrics[key]:.10e}" for key in ordered_keys)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_output", required=True, type=Path)
    parser.add_argument("--paddle_output", required=True, type=Path)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    torch_output = load_output_array(args.torch_output)
    paddle_output = load_output_array(args.paddle_output)
    metrics = compute_error_metrics(torch_output, paddle_output)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
