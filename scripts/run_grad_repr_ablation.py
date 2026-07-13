from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


NOISE_ORDER = [
    ("gaussian_sigma50", "G50"),
    ("spatial_gaussian_sigma55", "SpatialG"),
    ("poisson_alpha3p5", "Poisson"),
    ("salt_pepper_d0p02", "S&P"),
    ("speckle_var0p04", "Speckle"),
    ("mixture_level4", "Mixture"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate SC2A-DKF gradient-representation ablations.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=["grad8", "grad4", "sobel"], choices=["grad8", "grad4", "sobel"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--val-interval", type=int, default=10000)
    parser.add_argument("--run-prefix", default="sc2a_dkf_gradrepr")
    parser.add_argument("--results-dir", default=r"results\grad_repr_ablation")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--profile-size", type=int, default=256)
    parser.add_argument("--profile-repeat", type=int, default=20)
    parser.add_argument("--profile-warmup", type=int, default=5)
    return parser.parse_args()


def setup_env(run_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_CONSOLE"] = "off"
    tmp = run_root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    env["WANDB_DIR"] = str(tmp)
    return env


def run_logged(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        print("[cmd] " + " ".join(cmd), file=log, flush=True)
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def variant_run_name(prefix: str, variant: str, seed: int) -> str:
    return f"{prefix}_{variant}_seed{seed}"


def build_model_config(repo: Path, variant: str, num_iter: int) -> Any:
    cfg = OmegaConf.load(repo / "configs/models/idf_grad8_rgb3d_spatialmod_k4.yaml")
    denoiser = cfg.params.denoiser_config.params
    denoiser.num_iter = num_iter
    denoiser.lcm_type = "grad8"
    denoiser.grad_repr = variant
    denoiser.kernel_mode = "rgb3d_spatial_mod"
    denoiser.spatial_mod_per_output = True
    denoiser.use_spatial_grad_bias = True
    denoiser.spatial_grad_bias_beta = 0.1
    denoiser.spatial_delta_max = 0.2
    denoiser.spatial_mod_kl_weight = 0.02
    denoiser.spatial_mod_tv_weight = 0.001
    cfg.params.misc_config.adaptive_iteration = False
    cfg.params.misc_config.max_iteration = num_iter
    cfg.params.misc_config.warmup = 5000
    return cfg


def build_train_config(repo: Path, run_name: str, model_config: Path, run_root: Path, seed: int, max_steps: int, val_interval: int) -> Any:
    cfg = OmegaConf.load(repo / "configs/train/train_sc2a_dkf_30k.yaml")
    cfg.model.config = str(model_config)
    cfg.model.resume = None
    cfg.lightning.seed = seed
    cfg.lightning.trainer.default_root_dir = str(run_root)
    cfg.lightning.trainer.max_steps = max_steps
    cfg.lightning.trainer.val_check_interval = val_interval
    cfg.lightning.trainer.log_every_n_steps = 100
    cfg.lightning.trainer.enable_progress_bar = False
    cfg.lightning.callbacks[0].params.every_n_train_steps = val_interval
    cfg.lightning.loggers[0].params.save_dir = str(run_root)
    cfg.lightning.loggers[0].params.version = run_name
    cfg.lightning.loggers[1].params.save_dir = str(run_root)
    cfg.lightning.loggers[1].params.version = run_name
    return cfg


def save_yaml(cfg: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def find_last_checkpoint(run_root: Path) -> Path | None:
    matches = sorted(run_root.glob("**/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    matches = sorted(run_root.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_metrics(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["noise"]] = (float(row["psnr"]), float(row["ssim"]))
    return out


def load_idf_like(repo: Path, model_config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from idf.utils.common import instantiate_from_config, load_state_dict

    cfg = OmegaConf.load(model_config)
    model = instantiate_from_config(cfg)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    load_state_dict(model, state, strict=True)
    return model.to(device).eval()


class FixedIterWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_iter: int):
        super().__init__()
        self.model = model
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, adaptive_iter=False, max_iter=self.max_iter, alpha_schedule=None).clamp(0.0, 1.0)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def profile_model(repo: Path, model_config: Path, checkpoint: Path, device_name: str, size: int, repeat: int, warmup: int, max_iter: int) -> dict[str, float]:
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    lit_model = load_idf_like(repo, model_config, checkpoint, device)
    model = FixedIterWrapper(lit_model, max_iter=max_iter).to(device).eval()
    total, trainable = count_params(model)
    x = torch.rand(1, 3, size, size, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - t0) * 1000.0 / max(1, repeat)
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0) if device.type == "cuda" else 0.0

    flops = float("nan")
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(x)
        flops = float(sum(evt.flops for evt in prof.key_averages() if evt.flops is not None))
    except Exception:
        pass
    del model, lit_model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "params": float(total),
        "trainable_params": float(trainable),
        "flops": flops,
        "runtime_ms": float(runtime_ms),
        "peak_mem_mb": float(peak_mem_mb),
    }


def fmt_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def write_summary(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "status", "params", "trainable_params", "flops", "peak_mem_mb", "runtime_ms"]
    for _, label in NOISE_ORDER:
        fields.extend([f"{label}_psnr", f"{label}_ssim"])
    fields.extend(["avg_psnr", "avg_ssim"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Grad Representation Ablation",
        "",
        "| Variant | Status | G50 | SpatialG | Poisson | S&P | Speckle | Mixture | Avg | Params | FLOPs | Peak Mem MB | Runtime ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        noise_cells = []
        for _, label in NOISE_ORDER:
            noise_cells.append(f"{fmt_metric(float(row[f'{label}_psnr']))}/{fmt_metric(float(row[f'{label}_ssim']))}")
        lines.append(
            "| {variant} | {status} | {g50} | {spg} | {poi} | {sp} | {speckle} | {mix} | {avg} | {params:.0f} | {flops} | {mem} | {rt} |".format(
                variant=row["variant"],
                status=row["status"],
                g50=noise_cells[0],
                spg=noise_cells[1],
                poi=noise_cells[2],
                sp=noise_cells[3],
                speckle=noise_cells[4],
                mix=noise_cells[5],
                avg=f"{fmt_metric(float(row['avg_psnr']))}/{fmt_metric(float(row['avg_ssim']))}",
                params=float(row.get("params", float("nan"))),
                flops=fmt_metric(float(row.get("flops", float("nan")))),
                mem=fmt_metric(float(row.get("peak_mem_mb", float("nan")))),
                rt=fmt_metric(float(row.get("runtime_ms", float("nan")))),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    py = str(Path(args.python).resolve()) if Path(args.python).exists() else args.python
    results_root = repo / args.results_dir
    runs_root = results_root / "runs"
    cfg_root = repo / "configs" / "grad_repr_ablation"
    logs_root = results_root / "logs"
    rows: list[dict[str, Any]] = []

    for variant in args.variants:
        run_name = variant_run_name(args.run_prefix, variant, args.seed)
        run_root = runs_root / run_name
        model_cfg_path = cfg_root / run_name / "model.yaml"
        train_cfg_path = cfg_root / run_name / "train.yaml"
        save_yaml(build_model_config(repo, variant, args.num_iter), model_cfg_path)
        save_yaml(build_train_config(repo, run_name, model_cfg_path, run_root, args.seed, args.max_steps, args.val_interval), train_cfg_path)
        env = setup_env(run_root)
        status = "ok"
        if not args.skip_train:
            if find_last_checkpoint(run_root) and not args.force_train:
                print(f"[skip train] {run_name}", flush=True)
            else:
                print(f"[train] {run_name}", flush=True)
                try:
                    run_logged([py, "scripts/run_training.py", "--repo", str(repo), "--config", str(train_cfg_path), "--run-root", str(run_root)], logs_root / f"{run_name}_train.log", repo, env)
                except Exception as exc:
                    status = f"train_failed: {exc}"
        ckpt = find_last_checkpoint(run_root)
        eval_out = run_root / "ood_eval" / "step_last" / "six_noise_metrics.csv"
        if ckpt is None:
            status = status if status != "ok" else "missing_checkpoint"
        elif not args.skip_eval:
            if eval_out.exists() and not args.force_eval:
                print(f"[skip eval] {run_name}", flush=True)
            else:
                print(f"[eval] {run_name}", flush=True)
                try:
                    run_logged(
                        [
                            py,
                            "scripts/evaluate_cbsd68_six_noises.py",
                            "--repo",
                            str(repo),
                            "--model-config",
                            str(model_cfg_path),
                            "--checkpoint",
                            str(ckpt),
                            "--out",
                            str(eval_out),
                            "--device",
                            args.device,
                            "--seed",
                            str(args.seed),
                            "--modes",
                            "fixed",
                            "--max-iter",
                            str(args.num_iter),
                        ],
                        logs_root / f"{run_name}_eval.log",
                        repo,
                        env,
                    )
                except Exception as exc:
                    status = f"eval_failed: {exc}"

        metrics = read_metrics(eval_out)
        row: dict[str, Any] = {"variant": variant, "status": status}
        psnrs, ssims = [], []
        for noise_key, label in NOISE_ORDER:
            psnr, ssim = metrics.get(noise_key, (float("nan"), float("nan")))
            row[f"{label}_psnr"] = psnr
            row[f"{label}_ssim"] = ssim
            psnrs.append(psnr)
            ssims.append(ssim)
        row["avg_psnr"] = float(np.nanmean(psnrs)) if any(np.isfinite(v) for v in psnrs) else float("nan")
        row["avg_ssim"] = float(np.nanmean(ssims)) if any(np.isfinite(v) for v in ssims) else float("nan")
        row.update({"params": float("nan"), "trainable_params": float("nan"), "flops": float("nan"), "runtime_ms": float("nan"), "peak_mem_mb": float("nan")})
        if ckpt is not None and not args.skip_profile:
            print(f"[profile] {run_name}", flush=True)
            try:
                row.update(profile_model(repo, model_cfg_path, ckpt, args.device, args.profile_size, args.profile_repeat, args.profile_warmup, args.num_iter))
            except Exception as exc:
                status = row["status"]
                row["status"] = f"{status}; profile_failed: {exc}" if status != "ok" else f"profile_failed: {exc}"
        rows.append(row)
        write_summary(rows, results_root / "grad_repr_ablation_summary.csv", results_root / "grad_repr_ablation_summary.md")

    print(f"[done] {results_root / 'grad_repr_ablation_summary.csv'}", flush=True)
    print(f"[done] {results_root / 'grad_repr_ablation_summary.md'}", flush=True)


if __name__ == "__main__":
    main()



