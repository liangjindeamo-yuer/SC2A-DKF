from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
        description="Run SCF-IDF pre-theory experiments: direction rank, edge IoU, RGB redundancy, noise imbalance, and bilateral oracle."
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataroot", default="data/CBSD68")
    parser.add_argument("--out-dir", default=r"runs\scf_pretheory_experiments")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--edge-percentile", type=float, default=80.0)
    parser.add_argument(
        "--topk-edge-percentiles",
        default="80,90,95",
        help="Comma-separated clean-gradient percentiles for top-k direction agreement.",
    )
    parser.add_argument("--vis-count", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--bilateral-images", type=int, default=12, help="Number of images for bilateral oracle metrics; 0 means all.")
    parser.add_argument("--nlm-h", type=float, default=10.0)
    parser.add_argument("--nlm-h-color", type=float, default=10.0)
    parser.add_argument("--bilateral-d", type=int, default=7)
    parser.add_argument("--bilateral-sigma-color", type=float, default=45.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=5.0)
    parser.add_argument("--oracle-edge-dilate", type=int, default=3)
    return parser.parse_args()


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
    levels = {4: (0.008, 0.008, 1.0, 0.004, 0.008)}
    var_g, var_s1, alpha, density, var_s2 = levels[level]

    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = clip01(img + rng.normal(0.0, np.sqrt(var_g), img.shape).astype(np.float32))
        out = speckle_noise(var_s1)(out, rng)
        out = poisson_noise(alpha)(out, rng)
        out = salt_pepper_noise(density)(out, rng)
        out = speckle_noise(var_s2)(out, rng)
        return out

    return add


def noise_specs() -> dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]]:
    return {
        "gaussian50": ("Gaussian50", gaussian_noise(50.0)),
        "spatial_gaussian55": ("SpatialG55", spatial_gaussian_noise(55.0)),
        "poisson": ("Poisson", poisson_noise(3.5)),
        "salt_pepper": ("S&P", salt_pepper_noise(0.02)),
        "speckle": ("Speckle", speckle_noise(0.04)),
        "mixture": ("Mixture", mixture_noise(4)),
    }


def load_images(dataroot: Path, max_images: int = 0) -> list[tuple[str, np.ndarray]]:
    paths = sorted(
        [p for p in dataroot.iterdir() if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if max_images > 0:
        paths = paths[:max_images]
    return [(p.name, np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0) for p in paths]


def rgb_to_y_np(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def shift_reflect_2d(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = x.shape
    padded = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    y0 = 1 + dy
    x0 = 1 + dx
    return padded[y0 : y0 + h, x0 : x0 + w]


def grad8_luma(img: np.ndarray) -> np.ndarray:
    y = rgb_to_y_np(img) if img.ndim == 3 else img.astype(np.float32)
    return np.stack([np.abs(shift_reflect_2d(y, dy, dx) - y) for _, dy, dx in DIRECTIONS], axis=0)


def grad8_channel(channel: np.ndarray) -> np.ndarray:
    return np.stack([np.abs(shift_reflect_2d(channel, dy, dx) - channel) for _, dy, dx in DIRECTIONS], axis=0)


def edge_from_grad8(g8: np.ndarray, percentile: float) -> np.ndarray:
    mag = g8.max(axis=0)
    thresh = np.percentile(mag, percentile)
    return mag >= thresh


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def pearson_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    af -= af.mean()
    bf -= bf.mean()
    denom = af.std() * bf.std()
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.mean(af * bf) / denom, -1.0, 1.0))


def spearman_direction_rank(g8_a: np.ndarray, g8_b: np.ndarray) -> float:
    # Spearman per pixel over 8 directions, then average.
    # Fast ordinal ranks are used; exact tie-averaging is unnecessary for this
    # analysis and would be prohibitively slow over all BSD68 pixels.
    a = np.moveaxis(g8_a, 0, -1).reshape(-1, 8)
    b = np.moveaxis(g8_b, 0, -1).reshape(-1, 8)
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float32)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float32)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra * ra).mean(axis=1) * (rb * rb).mean(axis=1))
    corr = np.where(denom > 1e-12, (ra * rb).mean(axis=1) / denom, 0.0)
    return float(np.mean(np.clip(corr, -1.0, 1.0)))


def spearman_direction_rank_by_mask(g8_a: np.ndarray, g8_b: np.ndarray, mask: np.ndarray) -> float:
    a = np.moveaxis(g8_a, 0, -1).reshape(-1, 8)
    b = np.moveaxis(g8_b, 0, -1).reshape(-1, 8)
    m = mask.reshape(-1).astype(bool)
    if m.sum() == 0:
        return float("nan")
    a = a[m]
    b = b[m]
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float32)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float32)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra * ra).mean(axis=1) * (rb * rb).mean(axis=1))
    corr = np.where(denom > 1e-12, (ra * rb).mean(axis=1) / denom, 0.0)
    return float(np.mean(np.clip(corr, -1.0, 1.0)))


def topk_direction_agreement_by_mask(
    g8_source: np.ndarray,
    g8_clean: np.ndarray,
    mask: np.ndarray,
    k: int = 2,
) -> tuple[float, float]:
    """Compare strongest gradient directions on clean edge pixels only.

    Returns:
        topk_recall: mean fraction of clean top-k directions appearing in source top-k.
        topk_exact_match: fraction of pixels where the two top-k direction sets match.
    """
    source = np.moveaxis(g8_source, 0, -1).reshape(-1, 8)
    clean = np.moveaxis(g8_clean, 0, -1).reshape(-1, 8)
    m = mask.reshape(-1).astype(bool)
    if m.sum() == 0:
        return float("nan"), float("nan")

    source_top = np.argsort(source[m], axis=1)[:, -k:]
    clean_top = np.argsort(clean[m], axis=1)[:, -k:]
    source_sets = np.zeros((source_top.shape[0], 8), dtype=bool)
    clean_sets = np.zeros((clean_top.shape[0], 8), dtype=bool)
    rows = np.arange(source_top.shape[0])[:, None]
    source_sets[rows, source_top] = True
    clean_sets[rows, clean_top] = True
    hits = np.logical_and(source_sets, clean_sets).sum(axis=1)
    recall = hits.astype(np.float32) / float(k)
    exact = hits == k
    return float(recall.mean()), float(exact.mean())


def denoise_nlm(img: np.ndarray, h: float, h_color: float) -> np.ndarray:
    u8 = (clip01(img) * 255.0).round().astype(np.uint8)
    # OpenCV expects RGB/BGR only by convention; algorithm is channel-wise enough for this analysis.
    den = cv2.fastNlMeansDenoisingColored(u8, None, h=h, hColor=h_color, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32) / 255.0


def psnr(clean: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((clean.astype(np.float32) - pred.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def bilateral_per_channel(noisy: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    out = []
    for c in range(3):
        ch = (noisy[..., c] * 255.0).round().astype(np.uint8)
        filt = cv2.bilateralFilter(ch, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        out.append(filt.astype(np.float32) / 255.0)
    return np.stack(out, axis=-1)


def bilateral_cross_channel(noisy: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    y = (rgb_to_y_np(noisy) * 255.0).round().astype(np.uint8)
    y_f = cv2.bilateralFilter(y, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32) / 255.0
    y0 = rgb_to_y_np(noisy)
    ratio = y_f / np.maximum(y0, 1e-3)
    return clip01(noisy * ratio[..., None])


def oracle_edge_constrained_cross_channel(
    clean: np.ndarray,
    noisy: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
    edge_percentile: float,
    dilate: int,
) -> np.ndarray:
    cross = bilateral_cross_channel(noisy, d, sigma_color, sigma_space)
    edge = edge_from_grad8(grad8_luma(clean), edge_percentile).astype(np.uint8)
    if dilate > 1:
        kernel = np.ones((dilate, dilate), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
    # Oracle constraint: avoid cross-edge aggregation by falling back to noisy pixels on clean-edge bands.
    out = cross.copy()
    out[edge.astype(bool)] = noisy[edge.astype(bool)]
    return clip01(out)


def save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((clip01(img) * 255.0).round().astype(np.uint8)).save(path)


def save_gray(path: Path, img: np.ndarray) -> None:
    arr = img.astype(np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        row = {k: v for k, v in zip(group_keys, key)}
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items if metric in item and item[metric] != ""]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"{metric}_std"] = float(np.std(vals)) if vals else np.nan
        out.append(row)
    return out


def run_direction_edge_experiments(
    images: list[tuple[str, np.ndarray]],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    rank_rows = []
    edge_rows = []
    topk_rows = []
    specs = noise_specs()
    topk_edge_percentiles = parse_float_list(args.topk_edge_percentiles)
    for noise_idx, (noise_type, (label, add_noise)) in enumerate(specs.items()):
        for image_idx, (image_name, clean) in enumerate(tqdm(images, desc=f"Exp1/2 {noise_type}")):
            rng = np.random.default_rng(args.seed + noise_idx * 100_000 + image_idx)
            noisy = add_noise(clean, rng)
            denoised = denoise_nlm(noisy, args.nlm_h, args.nlm_h_color)

            clean_g8 = grad8_luma(clean)
            clean_edge = edge_from_grad8(clean_g8, args.edge_percentile)
            clean_smooth = ~clean_edge
            rank_rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_type,
                    "noise_label": label,
                    "source": "clean",
                    "region": "all",
                    "spearman_rank_corr": 1.0,
                }
            )
            rank_rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_type,
                    "noise_label": label,
                    "source": "clean",
                    "region": "edge",
                    "spearman_rank_corr": 1.0,
                }
            )
            rank_rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_type,
                    "noise_label": label,
                    "source": "clean",
                    "region": "smooth",
                    "spearman_rank_corr": 1.0,
                }
            )
            edge_rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_type,
                    "noise_label": label,
                    "source": "clean",
                    "edge_iou": 1.0,
                }
            )
            topk_masks = {}
            for percentile in topk_edge_percentiles:
                mask = edge_from_grad8(clean_g8, percentile)
                topk_masks[percentile] = mask
                topk_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_type,
                        "noise_label": label,
                        "source": "clean",
                        "region": "edge",
                        "edge_percentile": percentile,
                        "edge_top_fraction": 100.0 - percentile,
                        "topk": 2,
                        "top2_recall": 1.0,
                        "top2_exact_match": 1.0,
                    }
                )
            for source, img in (("raw", noisy), ("denoised_nlm", denoised)):
                g8 = grad8_luma(img)
                edge = edge_from_grad8(g8, args.edge_percentile)
                rank_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_type,
                        "noise_label": label,
                        "source": source,
                        "region": "all",
                        "spearman_rank_corr": spearman_direction_rank(g8, clean_g8),
                    }
                )
                rank_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_type,
                        "noise_label": label,
                        "source": source,
                        "region": "edge",
                        "spearman_rank_corr": spearman_direction_rank_by_mask(g8, clean_g8, clean_edge),
                    }
                )
                rank_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_type,
                        "noise_label": label,
                        "source": source,
                        "region": "smooth",
                        "spearman_rank_corr": spearman_direction_rank_by_mask(g8, clean_g8, clean_smooth),
                    }
                )
                edge_rows.append(
                    {
                        "image_name": image_name,
                        "noise_type": noise_type,
                        "noise_label": label,
                        "source": source,
                        "edge_iou": binary_iou(edge, clean_edge),
                    }
                )
                for percentile, mask in topk_masks.items():
                    top2_recall, top2_exact = topk_direction_agreement_by_mask(g8, clean_g8, mask, k=2)
                    topk_rows.append(
                        {
                            "image_name": image_name,
                            "noise_type": noise_type,
                            "noise_label": label,
                            "source": source,
                            "region": "edge",
                            "edge_percentile": percentile,
                            "edge_top_fraction": 100.0 - percentile,
                            "topk": 2,
                            "top2_recall": top2_recall,
                            "top2_exact_match": top2_exact,
                        }
                    )
    return rank_rows, edge_rows, topk_rows


def run_cross_channel_experiment(images: list[tuple[str, np.ndarray]]) -> list[dict]:
    rows = []
    for image_name, clean in tqdm(images, desc="Exp3 cross-channel"):
        mags = []
        for c in range(3):
            mags.append(grad8_channel(clean[..., c]).mean(axis=0))
        rows.append(
            {
                "image_name": image_name,
                "grad_corr_RG": pearson_corr_np(mags[0], mags[1]),
                "grad_corr_RB": pearson_corr_np(mags[0], mags[2]),
                "grad_corr_GB": pearson_corr_np(mags[1], mags[2]),
            }
        )
    return rows


def run_noise_imbalance_experiment(images: list[tuple[str, np.ndarray]], args: argparse.Namespace) -> list[dict]:
    rows = []
    specs = noise_specs()
    for noise_idx, (noise_type, (label, add_noise)) in enumerate(specs.items()):
        for image_idx, (image_name, clean) in enumerate(tqdm(images, desc=f"Exp4 imbalance {noise_type}")):
            rng = np.random.default_rng(args.seed + noise_idx * 100_000 + image_idx)
            noisy = add_noise(clean, rng)
            residual = noisy - clean
            stds = residual.reshape(-1, 3).std(axis=0)
            ratio = float(stds.max() / max(stds.min(), 1e-12))
            rows.append(
                {
                    "image_name": image_name,
                    "noise_type": noise_type,
                    "noise_label": label,
                    "std_R": float(stds[0]),
                    "std_G": float(stds[1]),
                    "std_B": float(stds[2]),
                    "std_max_over_min": ratio,
                }
            )
    return rows


def run_bilateral_oracle_experiment(images: list[tuple[str, np.ndarray]], out_dir: Path, args: argparse.Namespace) -> list[dict]:
    rows = []
    add_noise = salt_pepper_noise(0.02)
    limit = len(images) if args.bilateral_images == 0 else min(args.bilateral_images, len(images))
    vis_dir = out_dir / "vis_bilateral_oracle"
    for image_idx, (image_name, clean) in enumerate(tqdm(images[:limit], desc="Exp5 bilateral oracle")):
        rng = np.random.default_rng(args.seed + 3 * 100_000 + image_idx)
        noisy = add_noise(clean, rng)
        outputs = {
            "per_channel_bilateral": bilateral_per_channel(
                noisy, args.bilateral_d, args.bilateral_sigma_color, args.bilateral_sigma_space
            ),
            "cross_channel_bilateral": bilateral_cross_channel(
                noisy, args.bilateral_d, args.bilateral_sigma_color, args.bilateral_sigma_space
            ),
            "cross_channel_oracle_edge": oracle_edge_constrained_cross_channel(
                clean,
                noisy,
                args.bilateral_d,
                args.bilateral_sigma_color,
                args.bilateral_sigma_space,
                args.edge_percentile,
                args.oracle_edge_dilate,
            ),
        }
        rows.append({"image_name": image_name, "variant": "noisy", "psnr": psnr(clean, noisy)})
        for variant, pred in outputs.items():
            rows.append({"image_name": image_name, "variant": variant, "psnr": psnr(clean, pred)})
        if image_idx < args.vis_count:
            stem = Path(image_name).stem
            save_rgb(vis_dir / f"{stem}_clean.png", clean)
            save_rgb(vis_dir / f"{stem}_noisy_sp.png", noisy)
            for variant, pred in outputs.items():
                save_rgb(vis_dir / f"{stem}_{variant}.png", pred)
            save_gray(vis_dir / f"{stem}_clean_edge.png", edge_from_grad8(grad8_luma(clean), args.edge_percentile).astype(np.float32))
    return rows


def plot_rank_edge(rank_summary: list[dict], edge_summary: list[dict], out_dir: Path) -> None:
    labels = [noise_specs()[k][0] for k in noise_specs().keys()]
    keys = list(noise_specs().keys())
    sources = ["raw", "denoised_nlm"]
    x = np.arange(len(keys))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    for ax, summary, metric, title in (
        (axes[0], rank_summary, "spearman_rank_corr_mean", "Directional rank correlation"),
        (axes[1], edge_summary, "edge_iou_mean", "Edge detection IoU"),
    ):
        if metric == "spearman_rank_corr_mean":
            summary_for_plot = [r for r in summary if r.get("region", "all") == "all"]
        else:
            summary_for_plot = summary
        lookup = {(r["noise_type"], r["source"]): float(r[metric]) for r in summary_for_plot}
        for idx, source in enumerate(sources):
            vals = [lookup[(k, source)] for k in keys]
            ax.bar(x + (idx - 0.5) * width, vals, width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rank_corr_edge_iou.png")
    plt.close(fig)


def plot_topk_direction(topk_summary: list[dict], out_dir: Path) -> None:
    keys = list(noise_specs().keys())
    sources = ["raw", "denoised_nlm"]
    percentiles = sorted({float(r["edge_percentile"]) for r in topk_summary})
    top_fracs = [100.0 - p for p in percentiles]
    lookup = {
        (r["noise_type"], r["source"], float(r["edge_percentile"])): float(r["top2_recall_mean"])
        for r in topk_summary
        if r["source"] in sources
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharey=True)
    for ax, source in zip(axes, sources):
        for key in keys:
            vals = [lookup[(key, source, p)] for p in percentiles]
            ax.plot(top_fracs, vals, marker="o", label=noise_specs()[key][0])
        ax.set_xticks(top_fracs)
        ax.set_xlabel("clean edge top fraction (%)")
        ax.set_title(source)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("top-2 recall")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_top2_direction_agreement.png")
    plt.close(fig)


def plot_noise_imbalance(summary: list[dict], out_dir: Path) -> None:
    keys = list(noise_specs().keys())
    labels = [noise_specs()[k][0] for k in keys]
    lookup = {r["noise_type"]: float(r["std_max_over_min_mean"]) for r in summary}
    vals = [lookup[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(np.arange(len(keys)), vals)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("max channel noise std / min")
    ax.set_title("Channel noise imbalance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_channel_noise_imbalance.png")
    plt.close(fig)


def plot_bilateral_summary(summary: list[dict], out_dir: Path) -> None:
    variants = ["noisy", "per_channel_bilateral", "cross_channel_bilateral", "cross_channel_oracle_edge"]
    lookup = {r["variant"]: float(r["psnr_mean"]) for r in summary}
    vals = [lookup[v] for v in variants]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.bar(np.arange(len(variants)), vals)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(["Noisy", "Per-channel", "Cross-channel", "Cross-channel + oracle edge"], rotation=20, ha="right")
    ax.set_ylabel("PSNR")
    ax.set_title("Bilateral oracle on Salt & Pepper noise")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bilateral_oracle_psnr.png")
    plt.close(fig)


def write_readme(
    out_dir: Path,
    rank_summary: list[dict],
    topk_summary: list[dict],
    edge_summary: list[dict],
    cc_summary: list[dict],
    imbalance_summary: list[dict],
    bilateral_summary: list[dict],
) -> None:
    def table(rows: list[dict], cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows:
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    try:
                        vals.append(f"{float(v):.4f}")
                    except (ValueError, TypeError):
                        vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# SCF-IDF Pre-theory Experiments",
        "",
        "## Experiment 1: Directional Rank Correlation",
        "",
        table(rank_summary, ["noise_type", "source", "region", "spearman_rank_corr_mean", "spearman_rank_corr_std"]),
        "",
        "## Experiment 1b: Strongest Direction Top-2 Agreement",
        "",
        "This metric is computed only on clean edge pixels. It measures whether the two strongest clean gradient directions remain in the source top-2 directions.",
        "",
        table(
            topk_summary,
            [
                "noise_type",
                "source",
                "region",
                "edge_top_fraction",
                "edge_percentile",
                "top2_recall_mean",
                "top2_recall_std",
                "top2_exact_match_mean",
                "top2_exact_match_std",
            ],
        ),
        "",
        "## Experiment 2: Edge Detection IoU",
        "",
        table(edge_summary, ["noise_type", "source", "edge_iou_mean", "edge_iou_std"]),
        "",
        "## Experiment 3: Cross-channel Structure Correlation",
        "",
        table(cc_summary, ["grad_corr_RG_mean", "grad_corr_RB_mean", "grad_corr_GB_mean"]),
        "",
        "## Experiment 4: Channel Noise Imbalance",
        "",
        table(imbalance_summary, ["noise_type", "std_max_over_min_mean", "std_max_over_min_std"]),
        "",
        "## Experiment 5: Bilateral Filter Oracle on Salt & Pepper",
        "",
        table(bilateral_summary, ["variant", "psnr_mean", "psnr_std"]),
        "",
        "## Figures",
        "",
        "- `fig_rank_corr_edge_iou.png`",
        "- `fig_top2_direction_agreement.png`",
        "- `fig_channel_noise_imbalance.png`",
        "- `fig_bilateral_oracle_psnr.png`",
        "- `vis_bilateral_oracle/`",
    ]
    (out_dir / "README_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    out_dir = repo / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_images(Path(args.dataroot), args.max_images)
    print(f"[data] images={len(images)} out={out_dir}", flush=True)

    rank_rows, edge_rows, topk_rows = run_direction_edge_experiments(images, out_dir, args)
    cc_rows = run_cross_channel_experiment(images)
    imbalance_rows = run_noise_imbalance_experiment(images, args)
    bilateral_rows = run_bilateral_oracle_experiment(images, out_dir, args)

    rank_summary = grouped_mean(rank_rows, ["noise_type", "source", "region"], ["spearman_rank_corr"])
    topk_summary = grouped_mean(
        topk_rows,
        ["noise_type", "source", "region", "edge_top_fraction", "edge_percentile"],
        ["top2_recall", "top2_exact_match"],
    )
    edge_summary = grouped_mean(edge_rows, ["noise_type", "source"], ["edge_iou"])
    cc_summary = grouped_mean(cc_rows, [], ["grad_corr_RG", "grad_corr_RB", "grad_corr_GB"])
    imbalance_summary = grouped_mean(imbalance_rows, ["noise_type"], ["std_max_over_min"])
    bilateral_summary = grouped_mean(bilateral_rows, ["variant"], ["psnr"])

    write_csv(out_dir / "directional_rank_correlation.csv", rank_rows)
    write_csv(out_dir / "directional_rank_correlation_summary.csv", rank_summary)
    write_csv(out_dir / "top2_direction_agreement.csv", topk_rows)
    write_csv(out_dir / "top2_direction_agreement_summary.csv", topk_summary)
    write_csv(out_dir / "edge_iou.csv", edge_rows)
    write_csv(out_dir / "edge_iou_summary.csv", edge_summary)
    write_csv(out_dir / "cross_channel_structure_correlation.csv", cc_rows)
    write_csv(out_dir / "cross_channel_structure_correlation_summary.csv", cc_summary)
    write_csv(out_dir / "channel_noise_imbalance.csv", imbalance_rows)
    write_csv(out_dir / "channel_noise_imbalance_summary.csv", imbalance_summary)
    write_csv(out_dir / "bilateral_oracle_sp.csv", bilateral_rows)
    write_csv(out_dir / "bilateral_oracle_sp_summary.csv", bilateral_summary)

    plot_rank_edge(rank_summary, edge_summary, out_dir)
    plot_topk_direction(topk_summary, out_dir)
    plot_noise_imbalance(imbalance_summary, out_dir)
    plot_bilateral_summary(bilateral_summary, out_dir)
    write_readme(out_dir, rank_summary, topk_summary, edge_summary, cc_summary, imbalance_summary, bilateral_summary)
    print(f"[done] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

