from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_best_on_testsets import load_model, resolve, tensor_from_image, write_csv


IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


DATASET_DIRS = {
    "sidd": ("noisy", "clean"),
    "siddplus": ("noisy", "gt"),
    "polyU": ("noisy", "clean"),
    "polyU256": ("polyU", "noisy_256", "clean_256"),
    "nam": ("input_crops", "target_crops"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SCF-IDF on paired real-noise datasets with iter sweep.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--test-root", default="data/Test")
    parser.add_argument("--datasets", nargs="+", default=["sidd", "siddplus", "polyU", "nam"])
    parser.add_argument("--model-config", default="configs/models/sc2a_dkf.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sc2a_dkf_grad4_last.ckpt")
    parser.add_argument("--out-dir", default="runs/sc2a_dkf_realnoise_iter_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-iter", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def paired_paths(dataset_root: Path, dataset_name: str, max_images: int = 0) -> list[tuple[Path, Path]]:
    dataset_spec = DATASET_DIRS.get(dataset_name, ("noisy", "clean"))
    if len(dataset_spec) == 3:
        dataset_root = dataset_root.parent / dataset_spec[0]
        noisy_subdir, clean_subdir = dataset_spec[1], dataset_spec[2]
    else:
        noisy_subdir, clean_subdir = dataset_spec
    noisy_dir = dataset_root / noisy_subdir
    clean_dir = dataset_root / clean_subdir
    noisy_paths = sorted([p for p in noisy_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
    if max_images > 0:
        noisy_paths = noisy_paths[:max_images]
    pairs: list[tuple[Path, Path]] = []
    for noisy in noisy_paths:
        clean_name = noisy.name
        if dataset_name.lower() in {"polyu", "polyu256"}:
            clean_name = clean_name.replace("_real", "_mean")
        clean = clean_dir / clean_name
        if not clean.exists():
            raise FileNotFoundError(f"Missing GT for {noisy}: expected {clean}")
        pairs.append((noisy, clean))
    if not pairs:
        raise FileNotFoundError(f"No image pairs found in {noisy_dir} and {clean_dir}")
    return pairs


def evaluate_dataset_iter(
    dataset_name: str,
    pairs: list[tuple[Path, Path]],
    model,
    device: torch.device,
    num_iter: int,
    out_dir: Path,
) -> dict[str, str | float | int]:
    from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

    detail_rows: list[dict[str, str | float | int]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    noisy_psnr_values: list[float] = []
    noisy_ssim_values: list[float] = []

    for noisy_path, clean_path in tqdm(pairs, desc=f"{dataset_name} T{num_iter}", leave=False):
        noisy = load_rgb(noisy_path)
        clean = load_rgb(clean_path)
        if noisy.shape != clean.shape:
            raise ValueError(f"Shape mismatch: {noisy_path} {noisy.shape} vs {clean_path} {clean.shape}")
        x = tensor_from_image(noisy, device)
        y = tensor_from_image(clean, device)
        with torch.inference_mode():
            pred_t = model(x, adaptive_iter=False, max_iter=num_iter, alpha_schedule=None).clamp(0.0, 1.0)
            psnr = calculate_psnr_pt(y, pred_t, 0, test_y_channel=False).mean().item()
            ssim = calculate_ssim_pt(y, pred_t, 0, test_y_channel=False).mean().item()
            noisy_psnr = calculate_psnr_pt(y, x, 0, test_y_channel=False).mean().item()
            noisy_ssim = calculate_ssim_pt(y, x, 0, test_y_channel=False).mean().item()
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        noisy_psnr_values.append(noisy_psnr)
        noisy_ssim_values.append(noisy_ssim)
        detail_rows.append(
            {
                "dataset": dataset_name,
                "max_iter": num_iter,
                "image": noisy_path.name,
                "psnr": psnr,
                "ssim": ssim,
                "noisy_psnr": noisy_psnr,
                "noisy_ssim": noisy_ssim,
            }
        )

    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dataset_dir / f"per_image_iter_{num_iter:02d}.csv", detail_rows)
    return {
        "dataset": dataset_name,
        "max_iter": num_iter,
        "num_images": len(pairs),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "noisy_psnr": float(np.mean(noisy_psnr_values)),
        "noisy_ssim": float(np.mean(noisy_ssim_values)),
    }


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    test_root = Path(args.test_root).resolve()
    out_dir = resolve(repo, args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(repo))
    model = load_model(repo, resolve(repo, args.model_config), resolve(repo, args.checkpoint), device)

    dataset_pairs = {}
    for dataset_name in args.datasets:
        pairs = paired_paths(test_root / dataset_name, dataset_name, args.max_images)
        dataset_pairs[dataset_name] = pairs
        print(f"[dataset] {dataset_name}: {len(pairs)} pairs", flush=True)

    all_rows: list[dict[str, str | float | int]] = []
    for num_iter in range(args.min_iter, args.max_iter + 1):
        print(f"\n[iter] max_iter={num_iter}", flush=True)
        iter_rows = []
        for dataset_name, pairs in dataset_pairs.items():
            row = evaluate_dataset_iter(dataset_name, pairs, model, device, num_iter, out_dir)
            iter_rows.append(row)
            all_rows.append(row)
            print(
                f"{dataset_name:9s} T{num_iter:02d} PSNR={row['psnr']:.4f} SSIM={row['ssim']:.4f} "
                f"(noisy {row['noisy_psnr']:.4f}/{row['noisy_ssim']:.4f})",
                flush=True,
            )
        overall = {
            "dataset": "overall",
            "max_iter": num_iter,
            "num_images": sum(int(r["num_images"]) for r in iter_rows),
            "psnr": float(np.mean([float(r["psnr"]) for r in iter_rows])),
            "ssim": float(np.mean([float(r["ssim"]) for r in iter_rows])),
            "noisy_psnr": float(np.mean([float(r["noisy_psnr"]) for r in iter_rows])),
            "noisy_ssim": float(np.mean([float(r["noisy_ssim"]) for r in iter_rows])),
        }
        all_rows.append(overall)
        print(f"[iter done] T{num_iter:02d} overall PSNR={overall['psnr']:.4f} SSIM={overall['ssim']:.4f}", flush=True)

    write_csv(out_dir / "realnoise_iter_summary.csv", all_rows)
    best = max((r for r in all_rows if r["dataset"] == "overall"), key=lambda r: float(r["psnr"]))
    print(f"\n[done] summary: {out_dir / 'realnoise_iter_summary.csv'}", flush=True)
    print(f"[best overall PSNR] T{best['max_iter']} PSNR={best['psnr']:.4f} SSIM={best['ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()

