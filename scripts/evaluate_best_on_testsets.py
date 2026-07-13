from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best SCF-IDF model on multiple clean testsets.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--test-root", default="data/Test")
    parser.add_argument("--datasets", nargs="+", default=["CBSD68", "McMaster", "Kodak24", "Urban100"])
    parser.add_argument(
        "--model-config",
        default="configs/models/sc2a_dkf.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sc2a_dkf_grad4_last.ckpt",
    )
    parser.add_argument("--out-dir", default="runs/sc2a_dkf_testsets_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def resolve(repo: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def image_dir(dataset_root: Path) -> Path:
    clean = dataset_root / "clean"
    return clean if clean.exists() else dataset_root


def load_images(dataset_root: Path, max_images: int = 0) -> list[Path]:
    root = image_dir(dataset_root)
    paths = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {root}")
    return paths


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def tensor_from_image(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)


def load_model(repo: Path, model_config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    from idf.utils.common import instantiate_from_config, load_state_dict

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
    model.to(device).eval()
    return model


def evaluate_dataset(
    dataset_name: str,
    image_paths: list[Path],
    model,
    device: torch.device,
    seed: int,
    max_iter: int,
    out_dir: Path,
) -> list[dict[str, str | float]]:
    from scripts.evaluate_cbsd68_six_noises import table1_noises
    from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

    summary_rows: list[dict[str, str | float]] = []
    detail_rows: list[dict[str, str | float]] = []

    for spec_idx, spec in enumerate(table1_noises()):
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        for image_idx, image_path in enumerate(tqdm(image_paths, desc=f"{dataset_name} {spec.name}")):
            clean = load_rgb(image_path)
            rng = np.random.default_rng(seed + spec_idx * 100_000 + image_idx)
            noisy = spec.add(clean, rng)

            x = tensor_from_image(noisy, device)
            y = tensor_from_image(clean, device)
            with torch.inference_mode():
                pred = model(
                    x,
                    adaptive_iter=False,
                    max_iter=max_iter,
                    alpha_schedule=None,
                ).clamp(0.0, 1.0)
                psnr = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean().item()
                ssim = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean().item()
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "noise": spec.name,
                    "setting": spec.label,
                    "image": image_path.name,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

        mean_psnr = float(np.mean(psnr_values))
        mean_ssim = float(np.mean(ssim_values))
        summary_rows.append(
            {
                "dataset": dataset_name,
                "noise": spec.name,
                "setting": spec.label,
                "psnr": mean_psnr,
                "ssim": mean_ssim,
            }
        )
        print(f"{dataset_name:10s} {spec.label:28s} PSNR={mean_psnr:.4f} SSIM={mean_ssim:.4f}", flush=True)

    avg_psnr = float(np.mean([float(r["psnr"]) for r in summary_rows]))
    avg_ssim = float(np.mean([float(r["ssim"]) for r in summary_rows]))
    summary_rows.append(
        {
            "dataset": dataset_name,
            "noise": "average",
            "setting": "Average",
            "psnr": avg_psnr,
            "ssim": avg_ssim,
        }
    )

    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dataset_dir / "six_noise_metrics.csv", summary_rows)
    write_csv(dataset_dir / "six_noise_metrics_per_image.csv", detail_rows)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    test_root = Path(args.test_root).resolve()
    out_dir = resolve(repo, args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(repo, resolve(repo, args.model_config), resolve(repo, args.checkpoint), device)

    all_rows: list[dict[str, str | float]] = []
    for dataset_name in args.datasets:
        dataset_root = test_root / dataset_name
        image_paths = load_images(dataset_root, max_images=args.max_images)
        print(f"[dataset] {dataset_name}: {len(image_paths)} images from {image_dir(dataset_root)}", flush=True)
        rows = evaluate_dataset(dataset_name, image_paths, model, device, args.seed, args.max_iter, out_dir)
        all_rows.extend(rows)

    write_csv(out_dir / "all_testsets_six_noise_metrics.csv", all_rows)

    avg_rows = [r for r in all_rows if r["noise"] == "average"]
    overall = {
        "dataset": "overall",
        "noise": "average",
        "setting": "Average over datasets",
        "psnr": float(np.mean([float(r["psnr"]) for r in avg_rows])),
        "ssim": float(np.mean([float(r["ssim"]) for r in avg_rows])),
    }
    write_csv(out_dir / "overall_average.csv", [overall])
    print(f"[done] summary: {out_dir / 'all_testsets_six_noise_metrics.csv'}", flush=True)
    print(f"[done] overall average PSNR={overall['psnr']:.4f} SSIM={overall['ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()

