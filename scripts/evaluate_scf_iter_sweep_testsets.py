from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from evaluate_best_on_testsets import (
    evaluate_dataset,
    image_dir,
    load_images,
    load_model,
    resolve,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model with several fixed iteration counts.")
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
    parser.add_argument(
        "--out-dir",
        default="runs/sc2a_dkf_testsets_eval_iter_sweep",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-iter", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    test_root = Path(args.test_root).resolve()
    out_dir = resolve(repo, args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = load_model(repo, resolve(repo, args.model_config), resolve(repo, args.checkpoint), device)
    dataset_paths: dict[str, list[Path]] = {}
    for dataset_name in args.datasets:
        dataset_root = test_root / dataset_name
        paths = load_images(dataset_root, max_images=args.max_images)
        dataset_paths[dataset_name] = paths
        print(f"[dataset] {dataset_name}: {len(paths)} images from {image_dir(dataset_root)}", flush=True)

    all_rows: list[dict[str, str | float | int]] = []
    average_rows: list[dict[str, str | float | int]] = []
    for num_iter in range(args.min_iter, args.max_iter + 1):
        iter_dir = out_dir / f"iter_{num_iter:02d}"
        print(f"\n[iter] max_iter={num_iter} -> {iter_dir}", flush=True)

        iter_rows: list[dict[str, str | float | int]] = []
        for dataset_name in args.datasets:
            rows = evaluate_dataset(
                dataset_name=dataset_name,
                image_paths=dataset_paths[dataset_name],
                model=model,
                device=device,
                seed=args.seed,
                max_iter=num_iter,
                out_dir=iter_dir,
            )
            for row in rows:
                row_with_iter = {"max_iter": num_iter, **row}
                iter_rows.append(row_with_iter)
                all_rows.append(row_with_iter)

        avg_rows = [row for row in iter_rows if row["noise"] == "average"]
        overall = {
            "max_iter": num_iter,
            "dataset": "overall",
            "noise": "average",
            "setting": "Average over datasets",
            "psnr": float(np.mean([float(row["psnr"]) for row in avg_rows])),
            "ssim": float(np.mean([float(row["ssim"]) for row in avg_rows])),
        }
        iter_rows.append(overall)
        all_rows.append(overall)
        average_rows.extend(avg_rows)
        average_rows.append(overall)
        write_csv(iter_dir / "all_testsets_six_noise_metrics.csv", iter_rows)
        print(
            f"[iter done] max_iter={num_iter} overall PSNR={overall['psnr']:.4f} "
            f"SSIM={overall['ssim']:.4f}",
            flush=True,
        )

    write_csv(out_dir / "all_iters_six_noise_metrics.csv", all_rows)
    write_csv(out_dir / "iter_average_summary.csv", average_rows)

    best = max(
        (row for row in average_rows if row["dataset"] == "overall"),
        key=lambda row: float(row["psnr"]),
    )
    print(f"\n[done] full CSV: {out_dir / 'all_iters_six_noise_metrics.csv'}", flush=True)
    print(f"[done] average CSV: {out_dir / 'iter_average_summary.csv'}", flush=True)
    print(
        f"[best overall PSNR] max_iter={best['max_iter']} PSNR={float(best['psnr']):.4f} "
        f"SSIM={float(best['ssim']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

