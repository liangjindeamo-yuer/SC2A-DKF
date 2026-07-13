from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


class ExportWrapper(torch.nn.Module):
    def __init__(self, lit_model: torch.nn.Module, max_iter: int):
        super().__init__()
        self.lit_model = lit_model
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lit_model(
            x,
            adaptive_iter=False,
            max_iter=self.max_iter,
            alpha_schedule=None,
        ).clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final SCF-IDF model to ONNX.")
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--model-config",
        default="configs/models/sc2a_dkf.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sc2a_dkf_grad4_last.ckpt",
    )
    parser.add_argument(
        "--out",
        default="runs/sc2a_dkf_onnx/sc2a_dkf_k10_64x64.onnx",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic-batch", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from idf.utils.common import instantiate_from_config

    config_path = resolve(repo, args.model_config)
    ckpt_path = resolve(repo, args.checkpoint)
    out_path = resolve(repo, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)
    lit_model = instantiate_from_config(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = lit_model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")

    lit_model.eval().to(device)
    wrapper = ExportWrapper(lit_model, max_iter=int(config.params.misc_config.max_iteration or 10)).eval().to(device)
    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32, device=device).clamp(0.0, 1.0)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    with torch.inference_mode():
        y = wrapper(dummy)
        print(f"[check] input={tuple(dummy.shape)} output={tuple(y.shape)}", flush=True)
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    print(f"[done] exported ONNX: {out_path}", flush=True)


if __name__ == "__main__":
    main()

