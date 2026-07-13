from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    label: str
    add: Callable[[np.ndarray, np.random.Generator], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate IDF on CBSD68 with the six synthetic OOD noises from Table 1."
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataroot", default="data/CBSD68")
    parser.add_argument("--model-config", default="configs/models/idfnet.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/idf_g_15.ckpt")
    parser.add_argument("--out", default="runs/reproduce_cbsd68_six_noises/cbsd68_six_noises.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--modes", nargs="+", default=["dic", "fixed"], choices=["dic", "fixed"])
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--save-noisy", action="store_true")
    parser.add_argument("--noisy-dir", default="runs/reproduce_cbsd68_six_noises/noisy")
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


MIXTURE_LEVELS = {
    1: (0.003, 0.003, 1.0, 0.002, 0.003),
    2: (0.004, 0.004, 1.0, 0.002, 0.004),
    3: (0.006, 0.006, 1.0, 0.003, 0.006),
    4: (0.008, 0.008, 1.0, 0.004, 0.008),
}


def mixture_noise(level: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    var_g, var_s1, alpha, density, var_s2 = MIXTURE_LEVELS[level]

    def add(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = clip01(img + rng.normal(0.0, np.sqrt(var_g), img.shape).astype(np.float32))
        out = speckle_noise(var_s1)(out, rng)
        out = poisson_noise(alpha)(out, rng)
        out = salt_pepper_noise(density)(out, rng)
        out = speckle_noise(var_s2)(out, rng)
        return out

    return add


def table1_noises() -> list[NoiseSpec]:
    return [
        NoiseSpec("gaussian_sigma50", "Gaussian sigma=50", gaussian_noise(50.0)),
        NoiseSpec("spatial_gaussian_sigma55", "Spatial Gaussian sigma=55", spatial_gaussian_noise(55.0)),
        NoiseSpec("poisson_alpha3p5", "Poisson alpha=3.5", poisson_noise(3.5)),
        NoiseSpec("salt_pepper_d0p02", "Salt & Pepper d=0.02", salt_pepper_noise(0.02)),
        NoiseSpec("speckle_var0p04", "Speckle sigma^2=0.04", speckle_noise(0.04)),
        NoiseSpec("mixture_level4", "Mixture level 4", mixture_noise(4)),
    ]


def load_images(dataroot: Path) -> list[Path]:
    image_paths = sorted(
        [p for p in dataroot.iterdir() if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if len(image_paths) != 68:
        print(f"warning: expected 68 CBSD68 images, found {len(image_paths)} in {dataroot}")
    return image_paths


def load_model(repo: Path, model_config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from idf.utils.common import instantiate_from_config, load_state_dict

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_state_dict(model, state, strict=True)
    model.to(device)
    model.eval()
    return model


def tensor_from_image(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)


def evaluate() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    dataroot = Path(args.dataroot).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo / out_path

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_config = (repo / args.model_config).resolve()
    checkpoint = (repo / args.checkpoint).resolve()
    image_paths = load_images(dataroot)
    model = load_model(repo, model_config, checkpoint, device)

    from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt

    noisy_root = Path(args.noisy_dir)
    if not noisy_root.is_absolute():
        noisy_root = repo / noisy_root
    if args.save_noisy:
        noisy_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []
    summary: list[dict[str, str | float]] = []

    for spec_idx, spec in enumerate(table1_noises()):
        for mode in args.modes:
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            desc = f"{spec.name} [{mode}]"
            for image_idx, image_path in enumerate(tqdm(image_paths, desc=desc)):
                clean = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
                rng = np.random.default_rng(args.seed + spec_idx * 100_000 + image_idx)
                noisy = spec.add(clean, rng)

                if args.save_noisy and mode == args.modes[0]:
                    save_dir = noisy_root / spec.name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((noisy * 255.0).round().clip(0, 255).astype(np.uint8)).save(save_dir / image_path.name)

                x = tensor_from_image(noisy, device)
                y = tensor_from_image(clean, device)
                with torch.inference_mode():
                    pred = model(
                        x,
                        adaptive_iter=(mode == "dic"),
                        max_iter=args.max_iter,
                        alpha_schedule=None,
                    ).clamp(0.0, 1.0)
                    psnr = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean().item()
                    ssim = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean().item()
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                rows.append(
                    {
                        "noise": spec.name,
                        "setting": spec.label,
                        "mode": mode,
                        "image": image_path.name,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )

            mean_psnr = float(np.mean(psnr_values))
            mean_ssim = float(np.mean(ssim_values))
            summary.append(
                {
                    "noise": spec.name,
                    "setting": spec.label,
                    "mode": mode,
                    "psnr": mean_psnr,
                    "ssim": mean_ssim,
                }
            )
            print(f"{spec.label:28s} {mode:5s} PSNR={mean_psnr:.4f} SSIM={mean_ssim:.4f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_name(out_path.stem + "_per_image.csv")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["noise", "setting", "mode", "image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["noise", "setting", "mode", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"saved summary: {out_path}")
    print(f"saved per-image results: {detail_path}")


if __name__ == "__main__":
    evaluate()

