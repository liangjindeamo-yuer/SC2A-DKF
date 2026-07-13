from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Params, FLOPs, peak memory, and runtime for denoisers.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out-dir", default=r"runs\model_efficiency_profile")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-flops", action="store_true")

    parser.add_argument("--ours-config", default="configs/models/sc2a_dkf.yaml")
    parser.add_argument("--ours-ckpt", default="checkpoints/sc2a_dkf_grad4_last.ckpt")
    parser.add_argument("--idf-config", default=r"configs\models\idf_base.yaml")
    parser.add_argument("--idf-ckpt", default=r"runs\idf_base\CSVLogger\idf_base\checkpoints\last.ckpt")
    parser.add_argument("--restormer-root", default="external/Restormer")
    parser.add_argument(
        "--restormer-ckpt",
        default="external/Restormer/Denoising/pretrained_models/gaussian_color_denoising_sigma15.pth",
    )
    parser.add_argument("--swinir-root", default="external/SwinIR")
    parser.add_argument(
        "--swinir-ckpt",
        default="external/SwinIR/model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    )
    parser.add_argument("--dncnn-root", default="external/DnCNN")
    parser.add_argument(
        "--dncnn-ckpt",
        default="external/DnCNN/logs/DnCNN-S-RGB-IDF-sigma15/best.pth",
    )
    return parser.parse_args()


def resolve(repo: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def load_idf_like(repo: Path, config_path: Path, ckpt_path: Path, device: torch.device):
    from omegaconf import OmegaConf
    from idf.utils.common import instantiate_from_config, load_state_dict

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    try:
        load_state_dict(model, state, strict=True)
    except RuntimeError as exc:
        state_dict = state.get("state_dict", state)
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace(
                "model.block.diag_kernel_predictor.",
                "model.block.rgb3d_head.diag_kernel_predictor.",
            )
            new_key = new_key.replace(
                "model.block.offdiag_kernel_predictor.",
                "model.block.rgb3d_head.offdiag_kernel_predictor.",
            )
            new_key = new_key.replace(
                "model.block.offdiag_gate_head.",
                "model.block.rgb3d_head.offdiag_gate_head.",
            )
            remapped[new_key] = value
        if remapped == state_dict:
            raise exc
        load_state_dict(model, {"state_dict": remapped}, strict=True)
    return model.to(device).eval()


class IDFWrapper(nn.Module):
    def __init__(self, lit_model: nn.Module, max_iter: int = 10):
        super().__init__()
        self.lit_model = lit_model
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lit_model(x, adaptive_iter=False, max_iter=self.max_iter, alpha_schedule=None).clamp(0.0, 1.0)


def load_restormer(root: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(root))
    from basicsr.models.archs.restormer_arch import Restormer

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        dual_pixel_task=False,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("params", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class RestormerWrapper(nn.Module):
    def __init__(self, model: nn.Module, factor: int = 8):
        super().__init__()
        self.model = model
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.factor - h % self.factor) % self.factor
        pad_w = (self.factor - w % self.factor) % self.factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return self.model(x)[..., :h, :w].clamp(0.0, 1.0)


def load_swinir(root: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(root))
    from models.network_swinir import SwinIR

    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class SwinIRWrapper(nn.Module):
    def __init__(self, model: nn.Module, window_size: int = 8):
        super().__init__()
        self.model = model
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        h_pad = (h // self.window_size + 1) * self.window_size - h
        w_pad = (w // self.window_size + 1) * self.window_size - w
        x_pad = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : h + h_pad, :]
        x_pad = torch.cat([x_pad, torch.flip(x_pad, [3])], 3)[:, :, :, : w + w_pad]
        return self.model(x_pad)[..., :h, :w].clamp(0.0, 1.0)


def normalized_state_dict(ckpt) -> dict[str, torch.Tensor]:
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt.get("net", ckpt))) if isinstance(ckpt, dict) else ckpt
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def load_dncnn(root: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    spec = importlib.util.spec_from_file_location("dncnn_models_local", root / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import DnCNN models.py from {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DnCNN = module.DnCNN

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = normalized_state_dict(ckpt)
    first = next(v for v in state_dict.values() if torch.is_tensor(v) and v.ndim == 4)
    channels = int(first.shape[1])
    model = DnCNN(channels=channels, num_of_layers=17)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class DnCNNWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.model(x)).clamp(0.0, 1.0)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_flops(model: nn.Module, x: torch.Tensor, device: torch.device) -> int | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.inference_mode():
                _ = model(x)
                synchronize(device)
        return int(sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", 0)))
    except Exception as exc:
        print(f"[warn] FLOPs profiler failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def profile_runtime_and_memory(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[float, int | None]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                _ = model(x)
            end.record()
            synchronize(device)
            runtime_ms = start.elapsed_time(end) / float(repeat)
            peak_mem = int(torch.cuda.max_memory_allocated(device))
            return float(runtime_ms), peak_mem

        start_time = time.perf_counter()
        for _ in range(repeat):
            _ = model(x)
        runtime_ms = (time.perf_counter() - start_time) * 1000.0 / float(repeat)
        return float(runtime_ms), None


def free_model(model: nn.Module | None, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fmt_millions(x: int | None) -> float | None:
    return None if x is None else float(x) / 1e6


def fmt_giga(x: int | None) -> float | None:
    return None if x is None else float(x) / 1e9


def fmt_mb(x: int | None) -> float | None:
    return None if x is None else float(x) / (1024.0 ** 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    headers = ["Model", "Params(M)", "FLOPs(G)", "Peak Mem(MB)", "Runtime(ms)"]
    lines = [
        "# Model Efficiency Profile",
        "",
        f"Input: `{args.batch_size}x3x{args.height}x{args.width}`",
        f"Device: `{args.device}`",
        f"Warmup/repeat: `{args.warmup}/{args.repeat}`",
        "",
        "FLOPs are reported with PyTorch profiler `with_flops=True`; custom ops may be partially counted.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {params_m:.4f} | {flops_g} | {peak_mem_mb} | {runtime_ms:.4f} |".format(
                model=row["model"],
                params_m=row["params_m"],
                flops_g="" if row["flops_g"] is None else f"{row['flops_g']:.4f}",
                peak_mem_mb="" if row["peak_mem_mb"] is None else f"{row['peak_mem_mb']:.2f}",
                runtime_ms=row["runtime_ms"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = resolve(repo, args.out_dir)
    x = torch.rand(args.batch_size, 3, args.height, args.width, device=device)

    factories: list[tuple[str, Callable[[], nn.Module]]] = [
        (
            "SC2A-DKF (ours)",
            lambda: IDFWrapper(
                load_idf_like(repo, resolve(repo, args.ours_config), resolve(repo, args.ours_ckpt), device),
                max_iter=10,
            ),
        )
    ]

    idf_ckpt = resolve(repo, args.idf_ckpt)
    if idf_ckpt.exists():
        factories.append(
            (
                "IDF",
                lambda: IDFWrapper(
                    load_idf_like(repo, resolve(repo, args.idf_config), idf_ckpt, device),
                    max_iter=10,
                ),
            )
        )

    restormer_root, restormer_ckpt = Path(args.restormer_root), Path(args.restormer_ckpt)
    if restormer_root.is_dir() and restormer_ckpt.is_file():
        factories.append(("Restormer", lambda: RestormerWrapper(load_restormer(restormer_root, restormer_ckpt, device))))

    swinir_root, swinir_ckpt = Path(args.swinir_root), Path(args.swinir_ckpt)
    if swinir_root.is_dir() and swinir_ckpt.is_file():
        factories.append(("SwinIR", lambda: SwinIRWrapper(load_swinir(swinir_root, swinir_ckpt, device))))

    dncnn_root, dncnn_ckpt = Path(args.dncnn_root), Path(args.dncnn_ckpt)
    if dncnn_root.is_dir() and dncnn_ckpt.is_file():
        factories.append(("DnCNN", lambda: DnCNNWrapper(load_dncnn(dncnn_root, dncnn_ckpt, device))))

    rows: list[dict] = []
    for model_name, factory in factories:
        print(f"[profile] loading {model_name}", flush=True)
        model = factory().to(device).eval()
        total_params, trainable_params = count_params(model)

        flops = None if args.skip_flops else profile_flops(model, x, device)
        runtime_ms, peak_mem = profile_runtime_and_memory(model, x, device, args.warmup, args.repeat)

        row = {
            "model": model_name,
            "input_shape": f"{args.batch_size}x3x{args.height}x{args.width}",
            "params": total_params,
            "params_m": fmt_millions(total_params),
            "trainable_params": trainable_params,
            "flops": flops,
            "flops_g": fmt_giga(flops),
            "peak_mem_bytes": peak_mem,
            "peak_mem_mb": fmt_mb(peak_mem),
            "runtime_ms": runtime_ms,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "device": str(device),
            "flops_method": "torch.profiler.with_flops",
        }
        rows.append(row)
        flops_text = "" if row["flops_g"] is None else f"{row['flops_g']:.4f}G"
        peak_text = "" if row["peak_mem_mb"] is None else f"{row['peak_mem_mb']:.2f}MB"
        print(
            f"[profile] {model_name}: params={row['params_m']:.4f}M "
            f"flops={flops_text} "
            f"peak={peak_text} "
            f"runtime={runtime_ms:.4f}ms",
            flush=True,
        )
        free_model(model, device)

    write_csv(out_dir / "model_efficiency_profile.csv", rows)
    write_markdown(out_dir / "model_efficiency_profile.md", rows, args)
    print(f"[done] CSV: {out_dir / 'model_efficiency_profile.csv'}", flush=True)
    print(f"[done] MD:  {out_dir / 'model_efficiency_profile.md'}", flush=True)


if __name__ == "__main__":
    main()




