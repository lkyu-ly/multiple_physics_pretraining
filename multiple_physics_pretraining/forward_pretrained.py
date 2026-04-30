import argparse
from pathlib import Path

import numpy as np
import yaml


def load_yaml_config(yaml_path, config_name):
    with open(yaml_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if config_name not in payload:
        raise KeyError(f"Config '{config_name}' not found in {yaml_path}")
    config = payload[config_name]
    return argparse.Namespace(**config)


def extract_model_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    if "model_state" in payload:
        return payload["model_state"]
    return payload


def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def normalize_case_arrays(case_payload):
    x = np.asarray(case_payload["x"], dtype=np.float32)
    state_labels = np.asarray(case_payload["state_labels"], dtype=np.int64)
    bcs = np.asarray(case_payload["bcs"], dtype=np.int64)
    batch_size = x.shape[1]
    if state_labels.ndim == 1:
        state_labels = np.tile(state_labels, (batch_size, 1))
    if bcs.ndim == 1:
        bcs = np.tile(bcs, (batch_size, 1))
    return x, state_labels, bcs


def summarize_array(array):
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "finite": bool(np.isfinite(array).all()),
    }


def run_forward(yaml_config, config_name, weights_path, case_npz, output_path, device):
    import torch

    try:
        from models.avit import build_avit
    except ImportError:
        from .models.avit import build_avit

    params = load_yaml_config(yaml_config, config_name)
    model = build_avit(params).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model_state = strip_module_prefix(extract_model_state_dict(checkpoint))
    model.load_state_dict(model_state)
    model.eval()

    with np.load(case_npz) as case_payload:
        x, state_labels, bcs = normalize_case_arrays(case_payload)

    with torch.inference_mode():
        output = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(state_labels).to(device),
            torch.from_numpy(bcs).to(device),
        )

    output_array = output.detach().cpu().numpy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, output=output_array)
    return summarize_array(output_array)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True, type=Path)
    parser.add_argument("--config", default="basic_config", type=str)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--case_npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_forward(
        yaml_config=args.yaml_config,
        config_name=args.config,
        weights_path=args.weights,
        case_npz=args.case_npz,
        output_path=args.output,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
