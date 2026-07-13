from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SCF-IDF per-iteration images and kernel summaries.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataset-root", default="data/cycleisp_cbsd68_rgb_dnd")
    parser.add_argument("--model-config", default="configs/models/sc2a_dkf.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sc2a_dkf_grad4_last.ckpt")
    parser.add_argument("--out-dir", default="runs/sc2a_dkf_iter_visuals")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256, help="Center crop size for saved visualization. 0 disables crop.")
    return parser.parse_args()


def resolve(repo: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo / path


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def tensor_from_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def image_from_tensor(x: torch.Tensor, crop_size: int = 0) -> Image.Image:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    if crop_size > 0:
        _, h, w = x.shape
        top = max((h - crop_size) // 2, 0)
        left = max((w - crop_size) // 2, 0)
        x = x[:, top : top + min(crop_size, h), left : left + min(crop_size, w)]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def paired_paths(dataset_root: Path, limit: int) -> list[tuple[Path, Path]]:
    noisy_dir = dataset_root / "noisy"
    clean_dir = dataset_root / "clean"
    noisy_paths = sorted(
        [p for p in noisy_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )[:limit]
    pairs: list[tuple[Path, Path]] = []
    for noisy_path in noisy_paths:
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean image for {noisy_path}: expected {clean_path}")
        pairs.append((noisy_path, clean_path))
    if not pairs:
        raise FileNotFoundError(f"No pairs found under {dataset_root}")
    return pairs


def load_model(repo: Path, model_config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    from idf.utils.common import instantiate_from_config, load_state_dict

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_state_dict(model, state, strict=True)
    model.to(device).eval()
    return model


def average_spatial_kernel(kernels: torch.Tensor) -> np.ndarray:
    b, cout, cin_kk, hw = kernels.shape
    cin = 3
    kk = cin_kk // cin
    k = int(round(kk ** 0.5))
    spatial = kernels.detach().float().view(b, cout, cin, kk, hw).sum(dim=2)
    avg = spatial.mean(dim=(0, 1, 3)).view(k, k)
    avg = avg / avg.sum().clamp_min(1e-8)
    return avg.cpu().numpy()


def kernel_stats(kernels: torch.Tensor, spatial_mod: torch.Tensor | None) -> dict[str, float | str]:
    b, cout, cin_kk, hw = kernels.shape
    cin = 3
    kk = cin_kk // cin
    k = int(round(kk ** 0.5))
    reshaped = kernels.detach().float().view(b, cout, cin, kk, hw)
    same_mask = torch.eye(cin, device=kernels.device, dtype=torch.bool).view(1, cin, cin, 1, 1)
    center_idx = kk // 2
    center_values = torch.stack([reshaped[:, ch, ch, center_idx, :] for ch in range(min(cout, cin))], dim=1)
    entropy = -(kernels.clamp_min(1e-12) * kernels.clamp_min(1e-12).log()).sum(dim=2).mean()
    spatial_avg = average_spatial_kernel(kernels)
    stats: dict[str, float | str] = {
        "kernel_size": f"{k}x{k}x{cin}",
        "kernel_entries_per_output": int(cin_kk),
        "kernel_entropy": float(entropy.item()),
        "center_weight": float(center_values.mean().item()),
        "offdiag_rgb_mass": float(reshaped.masked_fill(same_mask, 0).sum(dim=(2, 3)).mean().item()),
        "avg_spatial_kernel": " ".join(f"{v:.6f}" for v in spatial_avg.reshape(-1)),
    }
    if spatial_mod is not None:
        spatial = spatial_mod.detach().float()
        spatial_dim = 2
        stats["spatial_mod_entropy"] = float(
            (-(spatial.clamp_min(1e-12) * spatial.clamp_min(1e-12).log()).sum(dim=spatial_dim).mean()).item()
        )
        stats["spatial_mod_center_weight"] = float(spatial.select(spatial_dim, center_idx).mean().item())
    return stats


def save_kernel_heatmap(kernel: np.ndarray, path: Path, scale: int = 80) -> None:
    arr = kernel.astype(np.float32)
    arr = arr / max(arr.max(), 1e-8)
    img = Image.fromarray((arr * 255).round().astype(np.uint8), mode="L").resize(
        (arr.shape[1] * scale, arr.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    draw = ImageDraw.Draw(img)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            text = f"{kernel[y, x]:.3f}"
            draw.text((x * scale + 8, y * scale + scale // 2 - 7), text, fill=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_contact_sheet(images: list[tuple[str, Image.Image]], path: Path, cols: int = 4) -> None:
    thumb_w = max(img.width for _, img in images)
    thumb_h = max(img.height for _, img in images)
    label_h = 22
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(images):
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
        sheet.paste(img, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_iterations(model, noisy: torch.Tensor, max_iter: int) -> tuple[list[torch.Tensor], list[dict], list[np.ndarray]]:
    outputs: list[torch.Tensor] = []
    rows: list[dict] = []
    heatmaps: list[np.ndarray] = []
    x = model.normalize(noisy)
    state = (x, None, None)
    block = model.model.block
    for iter_idx in range(max_iter):
        state = block(state, use_dilation=(iter_idx % 2 == 0), mix_alpha=None)
        denorm = model.normalize(state[0], reverse=True).clamp(0.0, 1.0)
        outputs.append(denorm)
        kernels = block.last_kernels
        spatial_mod = getattr(block, "last_spatial_mod", None)
        if kernels is None:
            rows.append({"iteration": iter_idx + 1})
            heatmaps.append(np.full((3, 3), np.nan, dtype=np.float32))
            continue
        row = {"iteration": iter_idx + 1, **kernel_stats(kernels, spatial_mod)}
        rows.append(row)
        heatmaps.append(average_spatial_kernel(kernels))
    return outputs, rows, heatmaps


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out_dir = resolve(repo, args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(repo, resolve(repo, args.model_config), resolve(repo, args.checkpoint), device)
    pairs = paired_paths(Path(args.dataset_root), args.num_images)

    all_rows: list[dict] = []
    for sample_idx, (noisy_path, clean_path) in enumerate(pairs, start=1):
        sample_name = f"{sample_idx:02d}_{noisy_path.stem}"
        sample_dir = out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        noisy = tensor_from_image(load_rgb(noisy_path), device)
        clean = tensor_from_image(load_rgb(clean_path), device)
        image_from_tensor(noisy, args.image_size).save(sample_dir / "00_noisy.png")
        image_from_tensor(clean, args.image_size).save(sample_dir / "00_clean.png")

        with torch.inference_mode():
            outputs, rows, heatmaps = run_iterations(model, noisy, args.max_iter)

        contact_images: list[tuple[str, Image.Image]] = [
            ("noisy", image_from_tensor(noisy, args.image_size)),
            ("clean", image_from_tensor(clean, args.image_size)),
        ]
        for i, output in enumerate(outputs, start=1):
            img = image_from_tensor(output, args.image_size)
            img.save(sample_dir / f"iter_{i:02d}.png")
            contact_images.append((f"iter {i}", img))

        for row, heatmap in zip(rows, heatmaps):
            row = {"sample": sample_name, "image": noisy_path.name, **row}
            all_rows.append(row)
            save_kernel_heatmap(heatmap, sample_dir / f"kernel_iter_{int(row['iteration']):02d}_avg_spatial.png")

        write_csv(sample_dir / "kernel_stats.csv", [r for r in all_rows if r["sample"] == sample_name])
        make_contact_sheet(contact_images, sample_dir / "iterations_contact_sheet.png")
        print(f"[saved] {sample_dir}", flush=True)

    write_csv(out_dir / "kernel_stats_all.csv", all_rows)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()

