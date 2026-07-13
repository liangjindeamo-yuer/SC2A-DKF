from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


DIRECTIONS: list[tuple[str, int, int]] = [
    ("up", -1, 0),
    ("down", 1, 0),
    ("left", 0, -1),
    ("right", 0, 1),
    ("left_up", -1, -1),
    ("right_up", -1, 1),
    ("left_down", 1, -1),
    ("right_down", 1, 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Problem / pre-theory analysis for SCF-IDF."
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataroot", default="data/CBSD68")
    parser.add_argument("--out-dir", default=r"runs\scf_problem_analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["gaussian50", "spatial_gaussian55", "poisson", "salt_pepper", "speckle", "mixture"],
    )
    parser.add_argument("--vis-count", type=int, default=3)
    parser.add_argument("--edge-percentile", type=float, default=80.0)
    parser.add_argument("--model-specs", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images; useful for smoke tests.")
    return parser.parse_args()


def resolve_path(repo: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def gaussian_noise(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return clip01(img + rng.normal(0.0, sigma / 255.0, img.shape).astype(np.float32))

    return add


def spatial_gaussian_noise(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0.0, sigma / 255.0, img.shape).astype(np.float32)
        noise_t = torch.from_numpy(noise.transpose(2, 0, 1)).unsqueeze(0)
        weight = torch.full((3, 1, 3, 3), 1.0 / 9.0, dtype=noise_t.dtype)
        noise_t = F.conv2d(F.pad(noise_t, (1, 1, 1, 1), mode="reflect"), weight, groups=3)
        noise = noise_t.squeeze(0).numpy().transpose(1, 2, 0)
        return clip01(img + noise)

    return add


def poisson_noise(alpha: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        quantized = np.round(img * 255.0) / 255.0
        poisson_sample = rng.poisson(quantized * 255.0).astype(np.float32) / 255.0
        noise = poisson_sample - quantized
        return clip01(img + alpha * noise)

    return add


def salt_pepper_noise(density: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = img.copy()
        mask = rng.random(img.shape)
        noisy[mask < density / 2.0] = 0.0
        noisy[(mask >= density / 2.0) & (mask < density)] = 1.0
        return noisy.astype(np.float32)

    return add


def speckle_noise(variance: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        width = np.sqrt(3.0 * variance)
        n = rng.uniform(-width, width, img.shape).astype(np.float32)
        return clip01(img + n * img)

    return add


def mixture_noise(level: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    levels = {
        4: (0.008, 0.008, 1.0, 0.004, 0.008),
    }
    var_g, var_s1, alpha, density, var_s2 = levels[level]

    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = clip01(img + rng.normal(0.0, np.sqrt(var_g), img.shape).astype(np.float32))
        out = speckle_noise(var_s1)(out, rng)
        out = poisson_noise(alpha)(out, rng)
        out = salt_pepper_noise(density)(out, rng)
        out = speckle_noise(var_s2)(out, rng)
        return out

    return add


def all_noise_specs() -> dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]]:
    return {
        "gaussian50": ("Gaussian sigma=50", gaussian_noise(50.0)),
        "spatial_gaussian55": ("Spatial Gaussian sigma=55", spatial_gaussian_noise(55.0)),
        "poisson": ("Poisson alpha=3.5", poisson_noise(3.5)),
        "salt_pepper": ("Salt & Pepper d=0.02", salt_pepper_noise(0.02)),
        "speckle": ("Speckle sigma^2=0.04", speckle_noise(0.04)),
        "mixture": ("Mixture level 4", mixture_noise(4)),
    }


def load_rgb_images(dataroot: Path, max_images: int = 0) -> list[tuple[str, torch.Tensor, np.ndarray]]:
    paths = sorted(
        [p for p in dataroot.iterdir() if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if max_images > 0:
        paths = paths[:max_images]
    images = []
    for path in paths:
        arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        images.append((path.name, tensor, arr))
    return images


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    coeff = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * coeff).sum(dim=1, keepdim=True)


def directional_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    _, _, h, w = x.shape
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    y0 = 1 + dy
    x0 = 1 + dx
    return padded[:, :, y0 : y0 + h, x0 : x0 + w]


def compute_grad8(x: torch.Tensor, reduce: str = "luminance") -> torch.Tensor:
    if reduce == "luminance":
        base = rgb_to_y(x) if x.shape[1] == 3 else x.mean(dim=1, keepdim=True)
    elif reduce == "mean_rgb":
        base = x.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported reduce mode: {reduce}")
    grads = [(directional_shift(base, dy, dx) - base).abs() for _, dy, dx in DIRECTIONS]
    return torch.cat(grads, dim=1)


def gradient_magnitude_from_grad8(grad8: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "max":
        return grad8.max(dim=1, keepdim=True).values
    return grad8.mean(dim=1, keepdim=True)


def edge_map_from_gradient(grad_mag: torch.Tensor, percentile: float = 80.0) -> torch.Tensor:
    b = grad_mag.shape[0]
    flat = grad_mag.flatten(1)
    thresh = torch.quantile(flat.float(), percentile / 100.0, dim=1).view(b, 1, 1, 1)
    return grad_mag >= thresh


def iou_binary(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool()
    b = b.bool()
    inter = (a & b).float().sum()
    union = (a | b).float().sum()
    if union.item() == 0:
        return 1.0
    return float((inter / union).item())


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = af.std(unbiased=False) * bf.std(unbiased=False)
    if denom.item() <= 1e-12:
        return 0.0
    return float((af * bf).mean().div(denom).clamp(-1, 1).item())


def cosine_similarity_map(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-8)
    return float(sim.mean().item())


def direction_min_agreement(grad8_a: torch.Tensor, grad8_b: torch.Tensor) -> float:
    return float((grad8_a.argmin(dim=1) == grad8_b.argmin(dim=1)).float().mean().item())


def direction_corr_mean(grad8_a: torch.Tensor, grad8_b: torch.Tensor) -> float:
    vals = [pearson_corr(grad8_a[:, i : i + 1], grad8_b[:, i : i + 1]) for i in range(grad8_a.shape[1])]
    return float(np.mean(vals))


def avg_blur3(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    weight = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=x.dtype, device=x.device)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), weight, groups=c)


def save_gray(path: Path, x: torch.Tensor) -> None:
    arr = x.detach().float().squeeze().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)


def save_rgb(path: Path, x: torch.Tensor) -> None:
    arr = x.detach().float().squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in group_keys)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        summary = {k: v for k, v in zip(group_keys, key)}
        for metric in metric_keys:
            vals = []
            for item in items:
                try:
                    val = float(item.get(metric, np.nan))
                except (TypeError, ValueError):
                    continue
                if not np.isnan(val):
                    vals.append(val)
            summary[f"{metric}_mean"] = float(np.mean(vals)) if vals else np.nan
            summary[f"{metric}_std"] = float(np.std(vals)) if vals else np.nan
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def experiment_directional(
    images: list[tuple[str, torch.Tensor, np.ndarray]],
    noises: dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]],
    out_dir: Path,
    edge_percentile: float,
    vis_count: int,
    seed: int,
    model_runners: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    vis_dir = out_dir / "vis_directional"
    for noise_idx, (noise_name, (_, add_noise)) in enumerate(noises.items()):
        for image_idx, (image_name, clean_t_cpu, clean_np) in enumerate(tqdm(images, desc=f"Directional {noise_name}")):
            rng = np.random.default_rng(seed + noise_idx * 100_000 + image_idx)
            noisy_np = add_noise(clean_np, rng)
            clean = clean_t_cpu.float()
            noisy = torch.from_numpy(noisy_np.transpose(2, 0, 1)).unsqueeze(0).float()
            blur = avg_blur3(noisy)

            clean_g8 = compute_grad8(clean)
            clean_edge = edge_map_from_gradient(gradient_magnitude_from_grad8(clean_g8), edge_percentile)
            sources: dict[str, torch.Tensor] = {
                "clean": clean,
                "raw_noisy": noisy,
                "blur_noisy": blur,
            }

            if model_runners:
                for model_name, runner in model_runners.items():
                    for max_iter in (1, 3, 5, 10):
                        try:
                            with torch.inference_mode():
                                pred = runner(noisy, max_iter).detach().cpu().clamp(0, 1)
                            sources[f"{model_name}_t{max_iter}"] = pred
                        except Exception as exc:
                            print(f"[warn] model directional failed {model_name} t{max_iter} {image_name}: {exc}", flush=True)
                            break

            for source_type, source in sources.items():
                g8 = compute_grad8(source)
                edge = edge_map_from_gradient(gradient_magnitude_from_grad8(g8), edge_percentile)
                rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_name,
                        "source_type": source_type,
                        "edge_iou": iou_binary(edge, clean_edge),
                        "grad8_cosine": cosine_similarity_map(g8, clean_g8),
                        "direction_min_agreement": direction_min_agreement(g8, clean_g8),
                        "direction_corr_mean": direction_corr_mean(g8, clean_g8),
                    }
                )

            if image_idx < vis_count:
                stem = f"{noise_name}_{Path(image_name).stem}"
                dst = vis_dir / stem
                dst.mkdir(parents=True, exist_ok=True)
                save_rgb(dst / "clean.png", clean)
                save_rgb(dst / "noisy.png", noisy)
                save_gray(dst / "clean_edge.png", clean_edge.float())
                save_gray(dst / "raw_noisy_edge.png", edge_map_from_gradient(gradient_magnitude_from_grad8(compute_grad8(noisy)), edge_percentile).float())
                save_gray(dst / "blur_noisy_edge.png", edge_map_from_gradient(gradient_magnitude_from_grad8(compute_grad8(blur)), edge_percentile).float())
                if model_runners:
                    for source_type, source in sources.items():
                        if source_type.endswith("_t10"):
                            save_gray(dst / f"{source_type}_edge.png", edge_map_from_gradient(gradient_magnitude_from_grad8(compute_grad8(source)), edge_percentile).float())
    return rows


def experiment_cross_channel(
    images: list[tuple[str, torch.Tensor, np.ndarray]],
    noises: dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]],
    out_dir: Path,
    edge_percentile: float,
    vis_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    structure_rows = []
    noise_rows = []
    vis_dir = out_dir / "vis_cross_channel"

    for image_idx, (image_name, clean_t_cpu, clean_np) in enumerate(tqdm(images, desc="Cross-channel structure")):
        clean = clean_t_cpu.float()
        channel_grads = []
        channel_edges = []
        for ch in range(3):
            g8 = compute_grad8(clean[:, ch : ch + 1].repeat(1, 3, 1, 1), reduce="mean_rgb")
            mag = gradient_magnitude_from_grad8(g8)
            channel_grads.append(mag)
            channel_edges.append(edge_map_from_gradient(mag, edge_percentile))

        rg = pearson_corr(channel_grads[0], channel_grads[1])
        rb = pearson_corr(channel_grads[0], channel_grads[2])
        gb = pearson_corr(channel_grads[1], channel_grads[2])
        edge_r_gb = edge_map_from_gradient((channel_grads[1] + channel_grads[2]) * 0.5, edge_percentile)
        edge_g_rb = edge_map_from_gradient((channel_grads[0] + channel_grads[2]) * 0.5, edge_percentile)
        edge_b_rg = edge_map_from_gradient((channel_grads[0] + channel_grads[1]) * 0.5, edge_percentile)
        structure_rows.append(
            {
                "image_name": image_name,
                "grad_corr_RG": rg,
                "grad_corr_RB": rb,
                "grad_corr_GB": gb,
                "edge_iou_R_vs_GB": iou_binary(channel_edges[0], edge_r_gb),
                "edge_iou_G_vs_RB": iou_binary(channel_edges[1], edge_g_rb),
                "edge_iou_B_vs_RG": iou_binary(channel_edges[2], edge_b_rg),
            }
        )

        if image_idx < vis_count:
            stem = Path(image_name).stem
            dst = vis_dir / stem
            dst.mkdir(parents=True, exist_ok=True)
            save_rgb(dst / "clean.png", clean)
            for ch_name, mag, edge in zip(("R", "G", "B"), channel_grads, channel_edges):
                save_gray(dst / f"grad_{ch_name}.png", mag)
                save_gray(dst / f"edge_{ch_name}.png", edge.float())

    for noise_idx, (noise_name, (_, add_noise)) in enumerate(noises.items()):
        for image_idx, (image_name, clean_t_cpu, clean_np) in enumerate(tqdm(images, desc=f"Cross-channel noise {noise_name}")):
            rng = np.random.default_rng(seed + noise_idx * 100_000 + image_idx)
            noisy_np = add_noise(clean_np, rng)
            residual = torch.from_numpy((noisy_np - clean_np).transpose(2, 0, 1)).unsqueeze(0).float()
            clean = clean_t_cpu.float()
            stds = residual.flatten(2).std(dim=2, unbiased=False).squeeze(0)
            denom = float(stds.min().clamp_min(1e-12).item())
            abs_noise = residual.abs().sum(dim=1, keepdim=True)
            noise_rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_name,
                    "std_R": float(stds[0].item()),
                    "std_G": float(stds[1].item()),
                    "std_B": float(stds[2].item()),
                    "std_max_over_min": float(stds.max().item() / denom),
                    "residual_corr_RG": pearson_corr(residual[:, 0:1], residual[:, 1:2]),
                    "residual_corr_RB": pearson_corr(residual[:, 0:1], residual[:, 2:3]),
                    "residual_corr_GB": pearson_corr(residual[:, 1:2], residual[:, 2:3]),
                    "abs_noise_luminance_corr": pearson_corr(abs_noise, rgb_to_y(clean)),
                }
            )

    return structure_rows, noise_rows


def load_model_specs(repo: Path, specs_path: Path, device: torch.device) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    from idf.utils.common import instantiate_from_config, load_state_dict

    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    models = {}
    for name, spec in specs.items():
        cfg_path = resolve_path(repo, spec["config"])
        cfg = OmegaConf.load(cfg_path)
        if "model" in cfg and "config" in cfg.model:
            model_cfg_path = resolve_path(repo, cfg.model.config)
            model_cfg = OmegaConf.load(model_cfg_path)
        else:
            model_cfg = cfg
        model = instantiate_from_config(model_cfg)
        ckpt = resolve_path(repo, spec["checkpoint"])
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        try:
            load_state_dict(model, state, strict=True)
        except Exception as exc:
            print(f"[warn] strict load failed for {name}: {exc}; retry with legacy RGB3D key remap", flush=True)
            state_dict = state.get("state_dict", state)
            state_dict = remap_legacy_rgb3d_keys_for_model(model, state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[warn] {name} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        model.to(device).eval()
        models[name] = model
    return models


def remap_legacy_rgb3d_keys_for_model(model: Any, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_keys = set(model.state_dict().keys())
    out = {}
    for key, value in state_dict.items():
        new_key = key
        replacements = [
            (
                "model.block.diag_kernel_predictor.",
                "model.block.rgb3d_head.diag_kernel_predictor.",
                "model.block.rgb3d_head.base_head.diag_kernel_predictor.",
            ),
            (
                "model.block.offdiag_kernel_predictor.",
                "model.block.rgb3d_head.offdiag_kernel_predictor.",
                "model.block.rgb3d_head.base_head.offdiag_kernel_predictor.",
            ),
            (
                "model.block.offdiag_gate_head.",
                "model.block.rgb3d_head.offdiag_gate_head.",
                "model.block.rgb3d_head.base_head.offdiag_gate_head.",
            ),
            (
                "block.diag_kernel_predictor.",
                "block.rgb3d_head.diag_kernel_predictor.",
                "block.rgb3d_head.base_head.diag_kernel_predictor.",
            ),
            (
                "block.offdiag_kernel_predictor.",
                "block.rgb3d_head.offdiag_kernel_predictor.",
                "block.rgb3d_head.base_head.offdiag_kernel_predictor.",
            ),
            (
                "block.offdiag_gate_head.",
                "block.rgb3d_head.offdiag_gate_head.",
                "block.rgb3d_head.base_head.offdiag_gate_head.",
            ),
        ]
        for old_prefix, rgb3d_prefix, spatial_prefix in replacements:
            if key.startswith(old_prefix):
                candidate = key.replace(old_prefix, rgb3d_prefix, 1)
                spatial_candidate = key.replace(old_prefix, spatial_prefix, 1)
                if candidate in target_keys:
                    new_key = candidate
                elif spatial_candidate in target_keys:
                    new_key = spatial_candidate
                break
        out[new_key] = value
    return out


def run_model(model: Any, x_cpu: torch.Tensor, device: torch.device, max_iter: int) -> torch.Tensor:
    x = x_cpu.to(device)
    with torch.inference_mode():
        try:
            pred = model(x, adaptive_iter=False, max_iter=max_iter, alpha_schedule=None)
        except TypeError:
            pred = model(x)
    return pred.detach()


def model_runner_factory(model: Any, device: torch.device) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def runner(x_cpu: torch.Tensor, max_iter: int) -> torch.Tensor:
        return run_model(model, x_cpu, device, max_iter)

    return runner


def get_inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def nan_row() -> dict[str, float]:
    return {
        "kernel_center_weight": np.nan,
        "kernel_entropy": np.nan,
        "diag_rgb_mass": np.nan,
        "offdiag_rgb_mass": np.nan,
        "color_identity_deviation": np.nan,
        "spatial_mod_center_weight": np.nan,
        "spatial_mod_entropy": np.nan,
        "spatial_mod_kl_to_uniform": np.nan,
        "spatial_mod_tv": np.nan,
        "low_grad_offset_weight": np.nan,
        "high_grad_offset_weight": np.nan,
        "spatial_bias_mean": np.nan,
        "spatial_bias_std": np.nan,
        "delta_abs_mean": np.nan,
        "delta_abs_p90": np.nan,
    }


def extract_kernel_stats(model: Any, x_for_grad: torch.Tensor) -> dict[str, float]:
    inner = get_inner_model(model)
    stats = nan_row()
    diag = inner.get_diagnostics() if hasattr(inner, "get_diagnostics") else {}
    for key, value in diag.items():
        if key in {"center_weight", "kernel_entropy", "offdiag_rgb_mass", "spatial_mod_center_weight", "spatial_mod_entropy", "spatial_bias_mean"}:
            out_key = "kernel_center_weight" if key == "center_weight" else key
            stats[out_key] = float(value.detach().cpu().item())

    block = getattr(inner, "block", None)
    kernels = getattr(block, "last_kernels", None)
    spatial = getattr(block, "last_spatial_mod", None)
    spatial_bias = getattr(block, "last_spatial_bias", None)

    if kernels is not None and kernels.ndim == 4:
        k = kernels.detach().float().cpu()
        b, cout, cin_kk, hw = k.shape
        c = cout
        kk = cin_kk // max(c, 1)
        kr = int(round(math.sqrt(kk)))
        if kr * kr == kk:
            kv = k.view(b, cout, c, kk, hw)
            same = torch.eye(c, dtype=torch.bool).view(1, c, c, 1, 1)
            diag_mass = kv.masked_fill(~same, 0).sum(dim=(2, 3)).mean()
            off_mass = kv.masked_fill(same, 0).sum(dim=(2, 3)).mean()
            stats["diag_rgb_mass"] = float(diag_mass.item())
            stats["offdiag_rgb_mass"] = float(off_mass.item())
            color_mass = kv.sum(dim=3).mean(dim=(0, 3))
            eye = torch.eye(c)
            stats["color_identity_deviation"] = float((color_mass - eye).abs().mean().item())
            stats["kernel_entropy"] = float((-(k.clamp_min(1e-12) * k.clamp_min(1e-12).log()).sum(dim=2).mean()).item())
            stats["kernel_center_weight"] = float(kv[:, :, :, kk // 2, :].sum(dim=2).mean().item())

            grad8 = compute_grad8(x_for_grad.cpu())
            if spatial is None and kk == 9:
                spatial_like = kv.sum(dim=2).view(b, cout, kk, x_for_grad.shape[-2], x_for_grad.shape[-1])
                low, high = low_high_offset_weights(spatial_like, grad8)
                stats["low_grad_offset_weight"] = low
                stats["high_grad_offset_weight"] = high

    if spatial is not None:
        s = spatial.detach().float().cpu()
        if s.ndim == 5:
            kk = s.shape[2]
            stats["spatial_mod_center_weight"] = float(s[:, :, kk // 2].mean().item())
            ent = -(s.clamp_min(1e-12) * s.clamp_min(1e-12).log()).sum(dim=2).mean()
            stats["spatial_mod_entropy"] = float(ent.item())
            uniform_log = -math.log(float(kk))
            kl = (s.clamp_min(1e-12) * (s.clamp_min(1e-12).log() - uniform_log)).sum(dim=2).mean()
            stats["spatial_mod_kl_to_uniform"] = float(kl.item())
            tv_h = (s[..., 1:, :] - s[..., :-1, :]).abs().mean()
            tv_w = (s[..., :, 1:] - s[..., :, :-1]).abs().mean()
            stats["spatial_mod_tv"] = float((tv_h + tv_w).item())
            grad8 = compute_grad8(x_for_grad.cpu())
            low, high = low_high_offset_weights(s, grad8)
            stats["low_grad_offset_weight"] = low
            stats["high_grad_offset_weight"] = high

    if spatial_bias is not None:
        b = spatial_bias.detach().float().cpu()
        stats["spatial_bias_mean"] = float(b.mean().item())
        stats["spatial_bias_std"] = float(b.std(unbiased=False).item())

    return stats


def low_high_offset_weights(spatial: torch.Tensor, grad8: torch.Tensor) -> tuple[float, float]:
    # spatial: [B, Cout, 9, H, W]. Non-center order matches row-major 3x3.
    # grad8 order is up/down/left/right/lu/ru/ld/rd, map to spatial indices.
    if spatial.shape[2] != 9:
        return np.nan, np.nan
    mapping = torch.tensor([1, 7, 3, 5, 0, 2, 6, 8], dtype=torch.long)
    weights8 = spatial[:, :, mapping].mean(dim=1)
    g = grad8.float()
    low_thresh = torch.quantile(g.flatten(2), 0.30, dim=2).view(g.shape[0], g.shape[1], 1, 1)
    high_thresh = torch.quantile(g.flatten(2), 0.70, dim=2).view(g.shape[0], g.shape[1], 1, 1)
    low_mask = g <= low_thresh
    high_mask = g >= high_thresh
    low = weights8.masked_select(low_mask).mean()
    high = weights8.masked_select(high_mask).mean()
    return float(low.item()), float(high.item())


def experiment_models(
    images: list[tuple[str, torch.Tensor, np.ndarray]],
    noises: dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]],
    models: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

    metric_rows = []
    kernel_rows = []
    weak_rows = []
    for model_name, model in models.items():
        for noise_idx, (noise_name, (_, add_noise)) in enumerate(noises.items()):
            psnrs = []
            ssims = []
            for image_idx, (image_name, clean_t_cpu, clean_np) in enumerate(tqdm(images, desc=f"Model {model_name} {noise_name}")):
                rng = np.random.default_rng(seed + noise_idx * 100_000 + image_idx)
                noisy_np = add_noise(clean_np, rng)
                noisy = torch.from_numpy(noisy_np.transpose(2, 0, 1)).unsqueeze(0).float()
                clean = clean_t_cpu.to(device).float()
                pred = run_model(model, noisy, device, max_iter=10).clamp(0, 1)
                psnr = calculate_psnr_pt(clean, pred, 0, test_y_channel=False).mean().item()
                ssim = calculate_ssim_pt(clean, pred, 0, test_y_channel=False).mean().item()
                psnrs.append(psnr)
                ssims.append(ssim)
                metric_rows.append(
                    {
                        "model_name": model_name,
                        "noise_type": noise_name,
                        "image_name": image_name,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )
                kstats = extract_kernel_stats(model, noisy)
                kernel_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_name,
                        "model_name": model_name,
                        "iteration": 10,
                        **kstats,
                    }
                )
                if "scf" in model_name.lower():
                    weak_rows.append(
                        {
                            "image_name": image_name,
                            "noise_type": noise_name,
                            "model_name": model_name,
                            **{k: kstats[k] for k in nan_row().keys()},
                        }
                    )
            metric_rows.append(
                {
                    "model_name": model_name,
                    "noise_type": noise_name,
                    "image_name": "__mean__",
                    "psnr": float(np.mean(psnrs)),
                    "ssim": float(np.mean(ssims)),
                }
            )

    return metric_rows, kernel_rows, weak_rows


def create_readme(out_dir: Path) -> None:
    direction_summary = read_summary_csv(out_dir / "directional_reliability_summary.csv")
    cross_summary = read_summary_csv(out_dir / "cross_channel_summary.csv")
    kernel_summary = read_summary_csv(out_dir / "kernel_behavior_summary.csv")
    weak_summary = read_summary_csv(out_dir / "weak_spatial_modulation_summary.csv")

    lines = [
        "# SCF-IDF Problem Analysis Summary",
        "",
        "This folder contains pre-theory analysis for SCF-IDF. The script does not train models.",
        "",
        "## Directional Structure",
        "",
        summarize_direction_text(direction_summary),
        "",
        "## Cross-Channel Redundancy",
        "",
        summarize_cross_text(cross_summary),
        "",
        "## Kernel Behavior",
        "",
        summarize_kernel_text(kernel_summary, weak_summary),
        "",
        "## Suggested Paper Statements",
        "",
        "1. Under OOD noise, raw gradients are often less reliable than gradients computed from denoised or iteratively refined estimates, motivating Grad8 context from the current IDF state.",
        "2. RGB channels share strong local edge and texture structures, while noise strength and residual statistics vary across channels, motivating RGB-aware dynamic filtering.",
        "3. RGB-aware kernels increase capacity, but a weak structure-aware spatial modulation constrains spatial offsets and preserves IDF's conservative sum-to-one filtering bias.",
        "",
        "## Outputs",
        "",
        "- `directional_reliability.csv` and `directional_reliability_summary.csv`",
        "- `cross_channel_structure_stats.csv`, `cross_channel_noise_stats.csv`, and `cross_channel_summary.csv`",
        "- `component_ablation_metrics.csv`, `kernel_behavior_stats.csv`, and `kernel_behavior_summary.csv` when `--model-specs` is provided",
        "- `weak_spatial_modulation_stats.csv` and `weak_spatial_modulation_summary.csv` for SCF-like models",
    ]
    (out_dir / "README_summary.md").write_text("\n".join(lines), encoding="utf-8")


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_direction_text(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Directional reliability summary is unavailable."
    by_source = defaultdict(list)
    for row in rows:
        try:
            by_source[row.get("source_type", "")].append(float(row.get("edge_iou_mean", "nan")))
        except ValueError:
            pass
    parts = []
    for source, vals in sorted(by_source.items()):
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            parts.append(f"- `{source}` mean edge IoU across noises: {np.mean(vals):.4f}")
    return "\n".join(parts) if parts else "No valid edge IoU values found."


def summarize_cross_text(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Cross-channel summary is unavailable."
    lines = []
    for row in rows[:8]:
        group = ", ".join(f"{k}={v}" for k, v in row.items() if not k.endswith("_mean") and not k.endswith("_std"))
        metrics = []
        for key in ("grad_corr_RG_mean", "grad_corr_RB_mean", "grad_corr_GB_mean", "std_max_over_min_mean"):
            if key in row:
                try:
                    metrics.append(f"{key}={float(row[key]):.4f}")
                except ValueError:
                    pass
        if metrics:
            lines.append(f"- {group}: " + ", ".join(metrics))
    return "\n".join(lines) if lines else "No compact cross-channel values found."


def summarize_kernel_text(kernel_rows: list[dict[str, str]], weak_rows: list[dict[str, str]]) -> str:
    lines = []
    if kernel_rows:
        for row in kernel_rows[:10]:
            model = row.get("model_name", "")
            noise = row.get("noise_type", "")
            vals = []
            for key in ("kernel_entropy_mean", "offdiag_rgb_mass_mean", "spatial_mod_entropy_mean"):
                if key in row:
                    try:
                        vals.append(f"{key}={float(row[key]):.4f}")
                    except ValueError:
                        pass
            if vals:
                lines.append(f"- {model} / {noise}: " + ", ".join(vals))
    if weak_rows:
        lines.append("- Weak SCF modulation statistics are available in `weak_spatial_modulation_summary.csv`.")
    return "\n".join(lines) if lines else "Kernel behavior summary is unavailable; provide `--model-specs` to enable it."


def run_sanity_checks() -> None:
    x = torch.ones(1, 3, 16, 16)
    g8 = compute_grad8(x)
    assert float(g8.abs().max().item()) < 1e-6, "grad8 constant image check failed"
    edge = edge_map_from_gradient(gradient_magnitude_from_grad8(compute_grad8(torch.rand(1, 3, 16, 16))))
    assert abs(iou_binary(edge, edge) - 1.0) < 1e-6, "edge IoU self check failed"
    assert not math.isnan(pearson_corr(torch.ones(8), torch.ones(8))), "Pearson constant check failed"


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    dataroot = Path(args.dataroot).resolve()
    out_dir = resolve_path(repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))

    run_sanity_checks()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    noise_specs_all = all_noise_specs()
    noises = {name: noise_specs_all[name] for name in args.noise_types}
    images = load_rgb_images(dataroot, args.max_images)
    print(f"[data] images={len(images)} noises={list(noises)} out={out_dir}", flush=True)

    models: dict[str, Any] = {}
    model_runners: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {}
    if args.model_specs:
        models = load_model_specs(repo, resolve_path(repo, args.model_specs), device)
        model_runners = {name: model_runner_factory(model, device) for name, model in models.items()}
        print(f"[models] loaded: {list(models)}", flush=True)

    directional_rows = experiment_directional(
        images,
        noises,
        out_dir,
        args.edge_percentile,
        args.vis_count,
        args.seed,
        model_runners=model_runners if model_runners else None,
    )
    write_csv(out_dir / "directional_reliability.csv", directional_rows)
    write_csv(
        out_dir / "directional_reliability_summary.csv",
        summarize_rows(
            directional_rows,
            ["noise_type", "source_type"],
            ["edge_iou", "grad8_cosine", "direction_min_agreement", "direction_corr_mean"],
        ),
    )

    structure_rows, noise_rows = experiment_cross_channel(
        images,
        noises,
        out_dir,
        args.edge_percentile,
        args.vis_count,
        args.seed,
    )
    write_csv(out_dir / "cross_channel_structure_stats.csv", structure_rows)
    write_csv(out_dir / "cross_channel_noise_stats.csv", noise_rows)
    cross_summary = summarize_rows(
        structure_rows,
        [],
        ["grad_corr_RG", "grad_corr_RB", "grad_corr_GB", "edge_iou_R_vs_GB", "edge_iou_G_vs_RB", "edge_iou_B_vs_RG"],
    )
    cross_summary += summarize_rows(
        noise_rows,
        ["noise_type"],
        ["std_max_over_min", "residual_corr_RG", "residual_corr_RB", "residual_corr_GB", "abs_noise_luminance_corr"],
    )
    write_csv(out_dir / "cross_channel_summary.csv", cross_summary)

    if models:
        metric_rows, kernel_rows, weak_rows = experiment_models(images, noises, models, out_dir, device, args.seed)
        write_csv(out_dir / "component_ablation_metrics.csv", metric_rows)
        write_csv(out_dir / "kernel_behavior_stats.csv", kernel_rows)
        write_csv(
            out_dir / "kernel_behavior_summary.csv",
            summarize_rows(
                kernel_rows,
                ["model_name", "noise_type"],
                [
                    "kernel_center_weight",
                    "kernel_entropy",
                    "diag_rgb_mass",
                    "offdiag_rgb_mass",
                    "color_identity_deviation",
                    "spatial_mod_center_weight",
                    "spatial_mod_entropy",
                    "spatial_mod_kl_to_uniform",
                    "spatial_mod_tv",
                    "low_grad_offset_weight",
                    "high_grad_offset_weight",
                ],
            ),
        )
        write_csv(out_dir / "weak_spatial_modulation_stats.csv", weak_rows)
        write_csv(
            out_dir / "weak_spatial_modulation_summary.csv",
            summarize_rows(
                weak_rows,
                ["model_name", "noise_type"],
                [
                    "spatial_bias_mean",
                    "spatial_bias_std",
                    "spatial_mod_center_weight",
                    "spatial_mod_entropy",
                    "spatial_mod_kl_to_uniform",
                    "spatial_mod_tv",
                    "low_grad_offset_weight",
                    "high_grad_offset_weight",
                    "delta_abs_mean",
                    "delta_abs_p90",
                ],
            ),
        )
    else:
        for name in (
            "component_ablation_metrics.csv",
            "kernel_behavior_stats.csv",
            "kernel_behavior_summary.csv",
            "weak_spatial_modulation_stats.csv",
            "weak_spatial_modulation_summary.csv",
        ):
            write_csv(out_dir / name, [])

    create_readme(out_dir)
    print(f"[done] wrote analysis to {out_dir}", flush=True)
    print("Example with models:", flush=True)
    print(
        r"python scripts/analyze_scf_problem_analysis.py --repo . "
        r"--dataroot data/CBSD68 "
        r"--out-dir runs\scf_problem_analysis --device cuda --vis-count 3 "
        r"--model-specs configs\analysis\scf_model_specs.json",
        flush=True,
    )


if __name__ == "__main__":
    main()


