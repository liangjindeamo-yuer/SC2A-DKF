# SC2A-DKF

SC2A-DKF is a compact research code release for image-domain iterative dynamic
filtering with structure/color-aware kernels. It is distilled from the working
IDF experimental tree and keeps only the code that is useful for the final
SC2A-DKF model, its ablations, and evaluation.



## Installation

```bash
conda create -n sc2adkf python=3.10 -y
conda activate sc2adkf
pip install -r requirements.txt
```

The code expects PyTorch, PyTorch Lightning, OmegaConf, PIL/Pillow, NumPy, tqdm,
and image-quality metric dependencies listed in `requirements.txt`.

## Data Layout

The provided training template assumes:

```text
data/
  CBSD432/
    *.png
  CBSD68/
    *.png
```

Edit these files if your dataset paths are different:

```text
configs/datasets/cbsd432_train_g15.yaml
configs/datasets/cbsd68_val_g15.yaml
```

## Training

Train the default SC2A-DKF Grad4 model:

```bash
python main.py --config configs/train/train_sc2a_dkf_30k.yaml
```

Run the gradient representation ablation:

```bash
python scripts/run_grad_repr_ablation.py \
  --repo . \
  --python python \
  --device cuda \
  --variants grad8 grad4 sobel \
  --seed 0 \
  --num-iter 10 \
  --max-steps 30000 \
  --val-interval 10000
```

## Evaluation

Evaluate CBSD68 on the six synthetic OOD noises:

```bash
python scripts/evaluate_cbsd68_six_noises.py \
  --model-config configs/models/sc2a_dkf.yaml \
  --checkpoint checkpoints/sc2a_dkf_grad4_last.ckpt \
  --dataroot data/CBSD68 \
  --out runs/sc2a_dkf_eval/six_noise_metrics.csv \
  --device cuda \
  --max-iter 10
```

Profile efficiency:

```bash
python scripts/profile_model_efficiency.py \
  --ours-config configs/models/sc2a_dkf.yaml \
  --ours-ckpt checkpoints/sc2a_dkf_grad4_last.ckpt
```

Export per-iteration visualizations:

```bash
python scripts/export_scf_iter_visuals.py \
  --model-config configs/models/sc2a_dkf.yaml \
  --checkpoint checkpoints/sc2a_dkf_grad4_last.ckpt \
  --out-dir runs/sc2a_dkf_iter_visuals
```






