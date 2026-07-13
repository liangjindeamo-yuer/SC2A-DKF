from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SCF-IDF iteration count on CycleISP DND/SIDD and profile cost.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dnd-root", default="data/cycleisp_cbsd68_rgb_dnd")
    parser.add_argument("--sidd-root", default="data/cycleisp_cbsd68_rgb_sidd")
    parser.add_argument("--model-config", default="configs/models/sc2a_dkf.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sc2a_dkf_grad4_last.ckpt")
    parser.add_argument("--out-dir", default=r"profile_results\scf_t_sweep_cycleisp_dnd_sidd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-t", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--profile-height", type=int, default=256)
    parser.add_argument("--profile-width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def tensor_from_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def paired_paths(dataset_root: Path, max_images: int = 0) -> list[tuple[Path, Path]]:
    noisy_dir = dataset_root / "noisy"
    clean_dir = dataset_root / "clean"
    noisy_paths = sorted(
        [p for p in noisy_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if max_images > 0:
        noisy_paths = noisy_paths[:max_images]
    pairs: list[tuple[Path, Path]] = []
    for noisy_path in noisy_paths:
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean image for {noisy_path}: expected {clean_path}")
        pairs.append((noisy_path, clean_path))
    if not pairs:
        raise FileNotFoundError(f"No image pairs found under {dataset_root}")
    return pairs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_model(repo: Path, model_config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    from idf.utils.common import instantiate_from_config, load_state_dict

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    load_state_dict(model, state, strict=True)
    model.to(device).eval()
    return model


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    return (
        int(sum(p.numel() for p in model.parameters())),
        int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_flops(model, x: torch.Tensor, t: int, device: torch.device) -> int | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.inference_mode():
                _ = model(x, adaptive_iter=False, max_iter=t, alpha_schedule=None).clamp(0.0, 1.0)
                synchronize(device)
        return int(sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", 0)))
    except Exception as exc:
        print(f"[warn] FLOPs failed for T={t}: {type(exc).__name__}: {exc}", flush=True)
        return None


def profile_runtime_memory(
    model,
    x: torch.Tensor,
    t: int,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[float, int | None]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x, adaptive_iter=False, max_iter=t, alpha_schedule=None).clamp(0.0, 1.0)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                _ = model(x, adaptive_iter=False, max_iter=t, alpha_schedule=None).clamp(0.0, 1.0)
            end.record()
            synchronize(device)
            runtime_ms = start.elapsed_time(end) / float(repeat)
            peak_mem = int(torch.cuda.max_memory_allocated(device))
            return float(runtime_ms), peak_mem

        import time

        start_time = time.perf_counter()
        for _ in range(repeat):
            _ = model(x, adaptive_iter=False, max_iter=t, alpha_schedule=None).clamp(0.0, 1.0)
        runtime_ms = (time.perf_counter() - start_time) * 1000.0 / float(repeat)
        return float(runtime_ms), None


def evaluate_pairs(dataset_name: str, pairs: list[tuple[Path, Path]], model, device: torch.device, t: int) -> dict:
    from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

    psnrs: list[float] = []
    ssims: list[float] = []
    for noisy_path, clean_path in tqdm(pairs, desc=f"{dataset_name} T={t}", leave=False):
        noisy = tensor_from_image(load_rgb(noisy_path), device)
        clean = tensor_from_image(load_rgb(clean_path), device)
        with torch.inference_mode():
            pred = model(noisy, adaptive_iter=False, max_iter=t, alpha_schedule=None).clamp(0.0, 1.0)
            psnr = calculate_psnr_pt(clean, pred, 0, test_y_channel=False).mean().item()
            ssim = calculate_ssim_pt(clean, pred, 0, test_y_channel=False).mean().item()
        psnrs.append(psnr)
        ssims.append(ssim)
    return {
        "dataset": dataset_name,
        "num_images": len(pairs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def make_markdown(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# SCF-IDF Iteration Sweep on CycleISP DND/SIDD",
        "",
        f"Model: `{args.checkpoint}`",
        f"Efficiency input: `1x3x{args.profile_height}x{args.profile_width}`",
        f"Warmup/repeat: `{args.warmup}/{args.repeat}`",
        "",
        "FLOPs use PyTorch profiler `with_flops=True`.",
        "",
        "| T | DND PSNR/SSIM | SIDD PSNR/SSIM | Avg PSNR/SSIM | Params(M) | FLOPs(G) | Peak Mem(MB) | Runtime(ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {T} | {dnd_psnr:.4f} / {dnd_ssim:.4f} | {sidd_psnr:.4f} / {sidd_ssim:.4f} | "
            "{avg_psnr:.4f} / {avg_ssim:.4f} | {params_m:.4f} | {flops_g:.4f} | "
            "{peak_mem_mb:.2f} | {runtime_ms:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out_dir = resolve(repo, args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(repo, resolve(repo, args.model_config), resolve(repo, args.checkpoint), device)
    total_params, trainable_params = count_params(model)

    dnd_pairs = paired_paths(Path(args.dnd_root), args.max_images)
    sidd_pairs = paired_paths(Path(args.sidd_root), args.max_images)
    profile_x = torch.rand(1, 3, args.profile_height, args.profile_width, device=device)

    rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    for t in range(args.min_t, args.max_t + 1):
        print(f"[T={t}] evaluating DND/SIDD", flush=True)
        dnd = evaluate_pairs("cycleisp_dnd", dnd_pairs, model, device, t)
        sidd = evaluate_pairs("cycleisp_sidd", sidd_pairs, model, device, t)
        per_dataset_rows.extend([{"T": t, **dnd}, {"T": t, **sidd}])

        print(f"[T={t}] profiling", flush=True)
        flops = None if args.skip_flops else profile_flops(model, profile_x, t, device)
        runtime_ms, peak_mem = profile_runtime_memory(model, profile_x, t, device, args.warmup, args.repeat)

        row = {
            "T": t,
            "dnd_psnr": dnd["psnr"],
            "dnd_ssim": dnd["ssim"],
            "sidd_psnr": sidd["psnr"],
            "sidd_ssim": sidd["ssim"],
            "avg_psnr": float((dnd["psnr"] + sidd["psnr"]) / 2.0),
            "avg_ssim": float((dnd["ssim"] + sidd["ssim"]) / 2.0),
            "params": total_params,
            "params_m": total_params / 1e6,
            "trainable_params": trainable_params,
            "flops": flops,
            "flops_g": None if flops is None else flops / 1e9,
            "peak_mem_bytes": peak_mem,
            "peak_mem_mb": None if peak_mem is None else peak_mem / (1024.0**2),
            "runtime_ms": runtime_ms,
            "profile_input": f"1x3x{args.profile_height}x{args.profile_width}",
            "warmup": args.warmup,
            "repeat": args.repeat,
        }
        rows.append(row)
        print(
            f"[T={t}] DND={dnd['psnr']:.4f}/{dnd['ssim']:.4f} "
            f"SIDD={sidd['psnr']:.4f}/{sidd['ssim']:.4f} "
            f"FLOPs={row['flops_g']:.4f}G Runtime={runtime_ms:.4f}ms",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "t_sweep_cycleisp_metrics_efficiency.csv", rows)
    write_csv(out_dir / "t_sweep_cycleisp_per_dataset.csv", per_dataset_rows)
    make_markdown(out_dir / "t_sweep_cycleisp_metrics_efficiency.md", rows, args)
    print(f"[done] {out_dir / 't_sweep_cycleisp_metrics_efficiency.csv'}", flush=True)
    print(f"[done] {out_dir / 't_sweep_cycleisp_metrics_efficiency.md'}", flush=True)


if __name__ == "__main__":
    main()

