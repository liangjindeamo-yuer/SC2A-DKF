import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from idf.archs.idf_arch import (
    GlobalChannelAttention,
    compute_local_correlation,
    rms_norm,
    same_padding,
)


def _power_norm_sum1(x: torch.Tensor, dim: int, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    """Non-negative power normalization with a per-pixel sum of one."""
    x = torch.abs(x).pow(alpha)
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def _directional_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Reflect-padded one-pixel directional shift."""
    _, _, h, w = x.shape
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    y0 = 1 + dy
    x0 = 1 + dx
    return padded[:, :, y0 : y0 + h, x0 : x0 + w]


GRAD8_DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

GRAD4_DIRECTIONS = [
    (0, 1),   # right
    (1, 0),   # down
    (1, 1),   # down-right / up-left axis
    (1, -1),  # down-left / up-right axis
]


def _normalize_grad_context(context: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean = context.mean(dim=(1, 2, 3), keepdim=True)
    std = context.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (context - mean) / (std + eps)


def gradient_context_channels(grad_repr: str) -> int:
    if grad_repr == "grad8":
        return 8
    if grad_repr == "grad4":
        return 4
    if grad_repr == "sobel":
        return 2
    raise ValueError(f"Unsupported grad_repr: {grad_repr}")


def compute_grad8_context(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Eight-direction gradient context from the current iterative image x_t.

    Returns a normalized [B, 8, H, W] map ordered as:
    up, down, left, right, left-up, right-up, left-down, right-down.
    """
    grads = []
    for dy, dx in GRAD8_DIRECTIONS:
        grad = (_directional_shift(x, dy, dx) - x).abs().mean(dim=1, keepdim=True)
        grads.append(grad)
    return _normalize_grad_context(torch.cat(grads, dim=1), eps=eps)


def compute_grad4_context(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Four representative directional gradients: right, down, and two diagonals."""
    grads = []
    for dy, dx in GRAD4_DIRECTIONS:
        grad = (_directional_shift(x, dy, dx) - x).abs().mean(dim=1, keepdim=True)
        grads.append(grad)
    return _normalize_grad_context(torch.cat(grads, dim=1), eps=eps)


def compute_sobel_context(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Sobel Gx/Gy context on channel-averaged intensity.

    Gx/Gy is the smallest Sobel representation to fuse with the existing
    appearance features. For the spatial modulation bias, it is projected onto
    each 3x3 offset direction to recover a per-offset structure penalty.
    """
    gray = x.mean(dim=1, keepdim=True)
    gx_kernel = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    gy_kernel = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gray_pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(gray_pad, gx_kernel)
    gy = F.conv2d(gray_pad, gy_kernel)
    return _normalize_grad_context(torch.cat([gx, gy], dim=1), eps=eps)


def compute_gradient_context(x: torch.Tensor, grad_repr: str = "grad8", eps: float = 1e-4) -> torch.Tensor:
    if grad_repr == "grad8":
        return compute_grad8_context(x, eps=eps)
    if grad_repr == "grad4":
        return compute_grad4_context(x, eps=eps)
    if grad_repr == "sobel":
        return compute_sobel_context(x, eps=eps)
    raise ValueError(f"Unsupported grad_repr: {grad_repr}")

def compute_grad8_magnitude(x: torch.Tensor) -> torch.Tensor:
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    grads = [(_directional_shift(x, dy, dx) - x).abs().mean(dim=1, keepdim=True) for dy, dx in directions]
    return torch.cat(grads, dim=1).mean(dim=1, keepdim=True)


def compute_spatial_gradient_bias(
    x: torch.Tensor,
    kernel_size: int = 3,
    dilation: int = 1,
    beta: float = 0.5,
    grad_repr: str = "grad8",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Structure bias for dynamic kernels.

    Large cross-structure responses receive a negative bias, so the spatial
    modulator is discouraged from collecting pixels across strong local changes.
    For Sobel, the Gx/Gy vector is projected onto each kernel offset direction;
    for Grad4, opposite offsets share the corresponding representative axis.
    The center offset bias is kept at zero.
    """
    _, _, h, w = x.shape
    pad = same_padding(kernel_size, dilation=dilation)
    radius = kernel_size // 2
    center_idx = kernel_size * kernel_size // 2
    diffs = []

    if grad_repr == "sobel":
        sobel = compute_sobel_context(x)
        gx, gy = sobel[:, 0:1], sobel[:, 1:2]
        for yy in range(kernel_size):
            for xx in range(kernel_size):
                dy = float((yy - radius) * dilation)
                dx = float((xx - radius) * dilation)
                norm = math.sqrt(dx * dx + dy * dy)
                if norm == 0.0:
                    diffs.append(torch.zeros_like(gx))
                else:
                    diffs.append((gx * (dx / norm) + gy * (dy / norm)).abs())
        diff = torch.cat(diffs, dim=1)
    elif grad_repr == "grad4":
        grad4 = torch.cat(
            [(_directional_shift(x, dy, dx) - x).abs().mean(dim=1, keepdim=True) for dy, dx in GRAD4_DIRECTIONS],
            dim=1,
        )
        for yy in range(kernel_size):
            for xx in range(kernel_size):
                dy = yy - radius
                dx = xx - radius
                if dy == 0 and dx == 0:
                    diffs.append(torch.zeros_like(grad4[:, 0:1]))
                elif dy == 0:
                    diffs.append(grad4[:, 0:1])
                elif dx == 0:
                    diffs.append(grad4[:, 1:2])
                elif dy * dx > 0:
                    diffs.append(grad4[:, 2:3])
                else:
                    diffs.append(grad4[:, 3:4])
        diff = torch.cat(diffs, dim=1)
    elif grad_repr == "grad8":
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        center = padded[:, :, pad : pad + h, pad : pad + w]
        for yy in range(kernel_size):
            for xx in range(kernel_size):
                dy = (yy - radius) * dilation
                dx = (xx - radius) * dilation
                y0 = pad + dy
                x0 = pad + dx
                neighbor = padded[:, :, y0 : y0 + h, x0 : x0 + w]
                diffs.append((neighbor - center).abs().mean(dim=1, keepdim=True))
        diff = torch.cat(diffs, dim=1)
    else:
        raise ValueError(f"Unsupported grad_repr: {grad_repr}")

    diff[:, center_idx : center_idx + 1] = 0.0
    scale = diff.mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    normalized = diff / scale
    normalized[:, center_idx : center_idx + 1] = 0.0
    return -beta * normalized

def _outer_ring_bias(
    kernel_size: int,
    bias: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    if kernel_size <= 3:
        mask = torch.zeros(kernel_size, kernel_size, device=device, dtype=dtype)
    else:
        inner_lo = kernel_size // 2 - 1
        inner_hi = kernel_size // 2 + 1
        inner = (yy >= inner_lo) & (yy <= inner_hi) & (xx >= inner_lo) & (xx <= inner_hi)
        mask = (~inner).to(dtype)
    return (mask.reshape(1, 1, kernel_size * kernel_size, 1) * bias).to(dtype)


def _apply_dynamic_rgb3d_kernel(
    patches: torch.Tensor,
    kernels: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    b, c, kk, hw = patches.shape
    patches_flat = patches.view(b, c * kk, hw)
    return torch.einsum("boqn,bqn->bon", kernels, patches_flat).view(b, c, h, w)


class RGB3DKernelHead(nn.Module):
    """RGB-aware dynamic kernel head with optional diagonal prior and outer-ring bias."""

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        kernel_size: int,
        power_alpha: float = 2.0,
        variant: str = "diag_gate",
        norm: str = "power",
        offdiag_gate_max: float = 0.1,
        offdiag_gate_init: float = -6.0,
        use_outer_ring_bias: bool = False,
        outer_ring_bias_init: float = -4.0,
    ):
        super().__init__()
        if variant not in {"diag_gate", "full"}:
            raise ValueError(f"Unsupported RGB3D kernel variant: {variant}")
        if norm not in {"power", "softmax"}:
            raise ValueError(f"Unsupported RGB3D kernel norm: {norm}")
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.power_alpha = power_alpha
        self.variant = variant
        self.norm = norm
        self.offdiag_gate_max = offdiag_gate_max
        self.use_outer_ring_bias = use_outer_ring_bias
        self.outer_ring_bias_init = outer_ring_bias_init
        kk = kernel_size * kernel_size

        if variant == "full":
            self.kernel_predictor = nn.Conv2d(
                in_channels,
                num_channels * num_channels * kk,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        else:
            self.diag_kernel_predictor = nn.Conv2d(
                in_channels,
                num_channels * kk,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.offdiag_kernel_predictor = nn.Conv2d(
                in_channels,
                num_channels * (num_channels - 1) * kk,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.offdiag_gate_head = nn.Conv2d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.offdiag_gate_head.weight)
            nn.init.constant_(self.offdiag_gate_head.bias, offdiag_gate_init)

        self.last_offdiag_gate: Optional[torch.Tensor] = None

    def _normalize(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if self.norm == "softmax":
            return torch.softmax(x, dim=dim)
        return _power_norm_sum1(x, dim=dim, alpha=self.power_alpha)

    def _maybe_add_outer_bias(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_outer_ring_bias:
            return logits
        bias = _outer_ring_bias(
            self.kernel_size,
            self.outer_ring_bias_init,
            logits.device,
            logits.dtype,
        )
        while bias.ndim < logits.ndim:
            bias = bias.unsqueeze(2)
        return logits + bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, _, h, w = features.shape
        c = self.num_channels
        kk = self.kernel_size * self.kernel_size
        hw = h * w

        if self.variant == "full":
            logits = self.kernel_predictor(features).view(b, c, c, kk, hw)
            logits = self._maybe_add_outer_bias(logits)
            kernels = self._normalize(logits.view(b, c, c * kk, hw), dim=2)
            self.last_offdiag_gate = None
            return kernels

        diag_logits = self.diag_kernel_predictor(features).view(b, c, kk, hw)
        diag_logits = self._maybe_add_outer_bias(diag_logits)
        diag_weights = self._normalize(diag_logits, dim=2)

        offdiag_logits = self.offdiag_kernel_predictor(features).view(b, c, c - 1, kk, hw)
        offdiag_logits = self._maybe_add_outer_bias(offdiag_logits)
        offdiag_weights = self._normalize(offdiag_logits.view(b, c, (c - 1) * kk, hw), dim=2)
        offdiag_weights = offdiag_weights.view(b, c, c - 1, kk, hw)

        gate = self.offdiag_gate_max * torch.sigmoid(self.offdiag_gate_head(features))
        gate = gate.view(b, c, 1, hw)
        self.last_offdiag_gate = gate.detach()

        kernels = diag_weights.new_zeros(b, c, c * kk, hw)
        for out_ch in range(c):
            same_slice = slice(out_ch * kk, (out_ch + 1) * kk)
            kernels[:, out_ch, same_slice, :] = (1.0 - gate[:, out_ch]) * diag_weights[:, out_ch]

            off_idx = 0
            for in_ch in range(c):
                if in_ch == out_ch:
                    continue
                src_slice = slice(in_ch * kk, (in_ch + 1) * kk)
                kernels[:, out_ch, src_slice, :] = gate[:, out_ch] * offdiag_weights[:, out_ch, off_idx]
                off_idx += 1
        return kernels


class RGB3DSpatialModKernelHead(nn.Module):
    """Original RGB3D kernel multiplied by a learned per-pixel 3x3 spatial modulator.

    The base RGB3D kernel C_{o,i,q}(p) is produced exactly as in idf_grad8_rgb3d.
    A separate spatial modulator S_{o,q}(p) is initialized to uniform weights.
    The final kernel is:

        K_{o,i,q}(p) = normalize_{i,q}(C_{o,i,q}(p) * S_{o,q}(p))

    Uniform S leaves C unchanged after normalization, so this branch starts from
    the original RGB3D behavior and only learns an extra spatial prior.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        kernel_size: int,
        power_alpha: float = 2.0,
        variant: str = "diag_gate",
        norm: str = "power",
        offdiag_gate_max: float = 0.1,
        offdiag_gate_init: float = -6.0,
        use_outer_ring_bias: bool = False,
        outer_ring_bias_init: float = -4.0,
        spatial_mod_per_output: bool = True,
        spatial_delta_max: float = 0.5,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.spatial_mod_per_output = spatial_mod_per_output
        self.spatial_delta_max = spatial_delta_max
        kk = kernel_size * kernel_size
        self.base_head = RGB3DKernelHead(
            in_channels,
            num_channels,
            kernel_size,
            power_alpha=power_alpha,
            variant=variant,
            norm=norm,
            offdiag_gate_max=offdiag_gate_max,
            offdiag_gate_init=offdiag_gate_init,
            use_outer_ring_bias=use_outer_ring_bias,
            outer_ring_bias_init=outer_ring_bias_init,
        )
        spatial_channels = num_channels * kk if spatial_mod_per_output else kk
        self.spatial_mod_head = nn.Conv2d(
            in_channels,
            spatial_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.spatial_mod_head.weight)
        nn.init.zeros_(self.spatial_mod_head.bias)

        self.last_base_kernels: Optional[torch.Tensor] = None
        self.last_spatial_mod: Optional[torch.Tensor] = None
        self.last_spatial_bias: Optional[torch.Tensor] = None
        self.last_offdiag_gate: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor, spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = features.shape
        c = self.num_channels
        kk = self.kernel_size * self.kernel_size
        hw = h * w

        base = self.base_head(features)
        spatial_logits = self.spatial_mod_head(features)
        if self.spatial_delta_max > 0:
            spatial_logits = self.spatial_delta_max * torch.tanh(spatial_logits)
        if self.spatial_mod_per_output:
            spatial_logits = spatial_logits.view(b, c, kk, hw)
            if spatial_bias is not None:
                spatial_logits = spatial_logits + spatial_bias.view(b, 1, kk, hw)
            spatial = torch.softmax(spatial_logits, dim=2)
            spatial = spatial.view(b, c, 1, kk, hw)
        else:
            spatial_logits = spatial_logits.view(b, 1, kk, hw)
            if spatial_bias is not None:
                spatial_logits = spatial_logits + spatial_bias.view(b, 1, kk, hw)
            spatial = torch.softmax(spatial_logits, dim=2)
            spatial = spatial.view(b, 1, 1, kk, hw)

        modulated = base.view(b, c, c, kk, hw) * spatial
        kernels = modulated.view(b, c, c * kk, hw)
        kernels = kernels / kernels.sum(dim=2, keepdim=True).clamp_min(1e-8)

        self.last_base_kernels = base.detach()
        self.last_spatial_mod = spatial.squeeze(2).view(b, -1, kk, h, w).detach()
        self.last_spatial_bias = None if spatial_bias is None else spatial_bias.detach()
        self.last_offdiag_gate = self.base_head.last_offdiag_gate
        return kernels


class DensityPriorNet(nn.Module):
    def __init__(self, hidden_channels: int = 32, gate_init: float = -5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.net[-1].bias, gate_init)

    def forward(self, x: torch.Tensor, resi: Optional[torch.Tensor]) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((gray - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        hf_energy = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs().mean(dim=1, keepdim=True)
        mad = F.avg_pool2d((gray - local_mean).abs(), kernel_size=5, stride=1, padding=2)
        outlier = ((gray - local_mean).abs() / (mad + 1e-4)).clamp(0, 10) / 10.0
        residual_mag = torch.zeros_like(gray) if resi is None else resi.abs().mean(dim=1, keepdim=True)
        grad_mag = compute_grad8_magnitude(x)
        maps = torch.cat([grad_mag, local_var, hf_energy, outlier, residual_mag], dim=1)
        maps = rms_norm(maps)
        return self.net(maps)


class DensityResidualGateNet(nn.Module):
    def __init__(self, hidden_channels: int = 32, gate_init: float = -6.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.net[-1].bias, gate_init)

    def forward(self, x: torch.Tensor, y5: torch.Tensor, y3: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((gray - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        hf_energy = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs().mean(dim=1, keepdim=True)
        mad = F.avg_pool2d((gray - local_mean).abs(), kernel_size=5, stride=1, padding=2)
        outlier = ((gray - local_mean).abs() / (mad + 1e-4)).clamp(0, 10) / 10.0
        residual_mag = (y5 - y3).abs().mean(dim=1, keepdim=True)
        grad_mag = compute_grad8_magnitude(x)
        maps = torch.cat([grad_mag, local_var, hf_energy, outlier, residual_mag], dim=1)
        maps = rms_norm(maps)
        return self.net(maps)


class DensityResidual5x5Block(nn.Module):
    """Trainable 5x5 RGB-aware residual branch on top of a frozen 3x3 IDF step."""

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        unfold_dilation: int = 2,
        outer_ring_bias_init: float = -6.0,
        density_g_max: float = 0.05,
        density_gate_init: float = -6.0,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        self.kernel_size = 5
        self.num_channels = num_channels
        self.unfold_dilation = unfold_dilation
        self.density_g_max = density_g_max
        self.grad_repr = grad_repr
        self.reflect_pad5 = nn.ReflectionPad2d(same_padding(5, dilation=unfold_dilation))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.gca = GlobalChannelAttention(num_channels, hidden_channels)
        self.head5 = RGB3DKernelHead(
            hidden_channels + gradient_context_channels(grad_repr),
            num_channels,
            5,
            variant="full",
            norm="softmax",
            use_outer_ring_bias=True,
            outer_ring_bias_init=outer_ring_bias_init,
        )
        self.gate = DensityResidualGateNet(hidden_channels=max(16, hidden_channels // 2), gate_init=density_gate_init)

        self.last_kernels: Optional[torch.Tensor] = None
        self.last_density_gate: Optional[torch.Tensor] = None

    def _unfold5(self, x: torch.Tensor, use_dilation: bool) -> torch.Tensor:
        padding, dilation = (0, self.unfold_dilation) if use_dilation else (same_padding(5), 1)
        src = self.reflect_pad5(x) if use_dilation else x
        patches = F.unfold(src, kernel_size=5, padding=padding, dilation=dilation)
        b, c, _, _ = x.shape
        return patches.view(b, c, 25, -1)

    def forward(
        self,
        x_in: torch.Tensor,
        y3: torch.Tensor,
        resi: Optional[torch.Tensor],
        use_dilation: bool,
    ) -> torch.Tensor:
        b, _, h, w = x_in.shape
        features = self.feature_extractor(rms_norm(x_in))
        if resi is not None:
            features = self.gca(features, cond=resi)
        features = torch.cat([features, compute_gradient_context(x_in, self.grad_repr)], dim=1)
        features = rms_norm(features)

        patches5 = self._unfold5(x_in, use_dilation=use_dilation)
        kernels5 = self.head5(features)
        y5 = _apply_dynamic_rgb3d_kernel(patches5, kernels5, h, w)
        gate = self.density_g_max * torch.sigmoid(self.gate(x_in, y5, y3))

        self.last_kernels = kernels5.clone()
        self.last_density_gate = gate.detach()
        return y3 + gate * (y5 - y3)


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.exists():
        return path
    repo_path = Path.cwd() / checkpoint
    if repo_path.exists():
        return repo_path
    parent = repo_path.parent.parent if repo_path.parent.name == "checkpoints" else repo_path.parent
    if parent.exists():
        matches = sorted(parent.glob("**/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve base checkpoint: {checkpoint}")


def _load_base_idf_checkpoint(model: nn.Module, checkpoint: str) -> None:
    ckpt_path = _resolve_checkpoint_path(checkpoint)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[len("model.") :]
        elif key.startswith("denoiser."):
            new_key = key[len("denoiser.") :]
        else:
            new_key = key
        new_key = new_key.replace("block.diag_kernel_predictor.", "block.rgb3d_head.diag_kernel_predictor.")
        new_key = new_key.replace("block.offdiag_kernel_predictor.", "block.rgb3d_head.offdiag_kernel_predictor.")
        new_key = new_key.replace("block.offdiag_gate_head.", "block.rgb3d_head.offdiag_gate_head.")
        model_state[new_key] = value
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("loss.")]
    if missing or unexpected:
        raise RuntimeError(f"Failed to load base checkpoint {ckpt_path}: missing={missing}, unexpected={unexpected}")


class IDFGrad8RGB3DDensityResidualNet(nn.Module):
    """Frozen idf_grad8_rgb3d plus a conservative trainable 5x5 density residual."""

    def __init__(
        self,
        base_checkpoint: str,
        freeze_base_3x3: bool = True,
        use_density_residual_5x5: bool = True,
        density_g_max: float = 0.05,
        density_gate_init: float = -6.0,
        target_gate_mean: float = 0.03,
        gate_loss_weight: float = 0.05,
        outer_ring_bias_init: float = -6.0,
        num_iter: int = 10,
        num_channels: int = 3,
        hidden_channels: int = 64,
        power_alpha: float = 3.0,
        halt_threshold: float = 0.015,
    ):
        super().__init__()
        if not use_density_residual_5x5:
            raise ValueError("This experiment requires use_density_residual_5x5=True")
        self.num_iter = num_iter
        self.target_gate_mean = target_gate_mean
        self.gate_loss_weight = gate_loss_weight
        self.base = IDFStructuredNet(
            num_iter=num_iter,
            kernel_size=3,
            num_channels=num_channels,
            hidden_channels=hidden_channels,
            halt_threshold=halt_threshold,
            lcm_type="grad8",
            kernel_mode="rgb3d",
            power_alpha=power_alpha,
            offdiag_gate_max=0.1,
            offdiag_gate_init=-6.0,
        )
        _load_base_idf_checkpoint(self.base, base_checkpoint)
        if freeze_base_3x3:
            self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False

        self.residual5 = DensityResidual5x5Block(
            num_channels=num_channels,
            hidden_channels=hidden_channels,
            unfold_dilation=2,
            outer_ring_bias_init=outer_ring_bias_init,
            density_g_max=density_g_max,
            density_gate_init=density_gate_init,
        )
        self.last_density_gate: Optional[torch.Tensor] = None
        self.last_kernels: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        adaptive_iter: bool = False,
        max_iter: Optional[int] = None,
        alpha_schedule: Optional[List[float]] = None,
    ) -> torch.Tensor:
        del adaptive_iter, alpha_schedule
        if max_iter is None:
            max_iter = self.num_iter

        x_t = x
        resi = None
        base_prev = None
        for i in range(max_iter):
            use_dilation = i % 2 == 0
            with torch.no_grad():
                y3, _, base_prev = self.base.block(
                    (x_t.detach(), resi.detach() if resi is not None else None, base_prev),
                    use_dilation=use_dilation,
                    mix_alpha=None,
                )
                y3 = y3.detach()
            y = self.residual5(x_t, y3, resi, use_dilation=use_dilation)
            resi = x_t - y
            x_t = y

        self.last_density_gate = self.residual5.last_density_gate
        self.last_kernels = self.residual5.last_kernels
        return x_t

    def get_extra_losses(self) -> dict[str, torch.Tensor]:
        gate = self.last_density_gate
        if gate is None:
            return {}
        gate_over = torch.relu(gate.mean() - self.target_gate_mean)
        return {"gate_loss": self.gate_loss_weight * gate_over.pow(2)}

    def get_diagnostics(self) -> dict[str, torch.Tensor]:
        kernels = self.last_kernels
        gate = self.last_density_gate
        stats: dict[str, torch.Tensor] = {}
        if gate is not None:
            stats["density_gate_mean"] = gate.mean()
            stats["density_gate_p90"] = torch.quantile(gate.flatten().float(), 0.9)
            stats.update(self.get_extra_losses())
        if kernels is None:
            return stats
        b, cout, cin_kk, hw = kernels.shape
        c = cout
        kk = cin_kk // c
        k = int(kk ** 0.5)
        kernels_reshaped = kernels.view(b, cout, c, kk, hw)
        same_mask = torch.eye(c, device=kernels.device, dtype=torch.bool).view(1, c, c, 1, 1)
        stats["offdiag_rgb_mass"] = kernels_reshaped.masked_fill(same_mask, 0).sum(dim=(2, 3)).mean()
        center_idx = kk // 2
        stats["center_weight"] = torch.stack(
            [kernels_reshaped[:, ch, ch, center_idx, :] for ch in range(c)], dim=1
        ).mean()
        stats["kernel_entropy"] = -(kernels.clamp_min(1e-12) * kernels.clamp_min(1e-12).log()).sum(dim=2).mean()
        coords = torch.arange(k, device=kernels.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        inner_lo = k // 2 - 1
        inner_hi = k // 2 + 1
        inner = ((yy >= inner_lo) & (yy <= inner_hi) & (xx >= inner_lo) & (xx <= inner_hi)).view(1, 1, 1, kk, 1)
        stats["inner_3x3_weight_mass"] = kernels_reshaped.masked_fill(~inner, 0).sum(dim=(2, 3)).mean()
        stats["outer_ring_weight_mass"] = kernels_reshaped.masked_fill(inner, 0).sum(dim=(2, 3)).mean()
        return stats


class StructureColorFactorizedDynamicFilter(nn.Module):
    """Apply pairwise factorized dynamic filtering.

    The effective kernel is K_{o,i,q}(p) = C_{o,i}(p) * S_{o,i,q}(p):
    every output-input color pair owns an independent 3x3 spatial kernel,
    while color mixing remains separately normalized.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        color_diagonal_prior: bool = True,
        color_diag_prior_value: float = 4.0,
        color_offdiag_prior_value: float = -4.0,
        spatial_center_prior: bool = False,
        spatial_center_prior_value: float = 0.0,
        spatial_pairwise: bool = True,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        if grad_repr not in {"grad8", "grad4", "sobel"}:
            raise ValueError(f"Unsupported grad_repr: {grad_repr}")
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.color_diagonal_prior = color_diagonal_prior
        self.spatial_center_prior = spatial_center_prior
        self.spatial_pairwise = spatial_pairwise
        self.reflect_pad = nn.ReflectionPad2d(same_padding(kernel_size))

        color_prior = torch.full((1, out_channels, in_channels, 1, 1), color_offdiag_prior_value)
        for idx in range(min(out_channels, in_channels)):
            color_prior[:, idx, idx] = color_diag_prior_value
        if not color_diagonal_prior:
            color_prior.zero_()
        self.register_buffer("color_prior", color_prior, persistent=False)

        if spatial_pairwise:
            spatial_prior = torch.zeros(1, out_channels, in_channels, kernel_size * kernel_size, 1, 1)
            if spatial_center_prior:
                spatial_prior[:, :, :, kernel_size * kernel_size // 2] = spatial_center_prior_value
        else:
            spatial_prior = torch.zeros(1, out_channels, kernel_size * kernel_size, 1, 1)
            if spatial_center_prior:
                spatial_prior[:, :, kernel_size * kernel_size // 2] = spatial_center_prior_value
        self.register_buffer("spatial_prior", spatial_prior, persistent=False)

        self.last_spatial: Optional[torch.Tensor] = None
        self.last_color: Optional[torch.Tensor] = None
        self.last_effective_kernel: Optional[torch.Tensor] = None

    def _unfold(
        self,
        x: torch.Tensor,
        use_dilation: bool = False,
        dilation: int = 2,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        if use_dilation:
            pad = nn.ReflectionPad2d(same_padding(self.kernel_size, dilation=dilation)).to(device=x.device)
            src = pad(x)
            patches = F.unfold(src, kernel_size=self.kernel_size, padding=0, dilation=dilation)
        else:
            src = self.reflect_pad(x)
            patches = F.unfold(src, kernel_size=self.kernel_size, padding=0, dilation=1)
        return patches.view(b, c, self.kernel_size * self.kernel_size, h, w)

    def compose_weights(
        self,
        spatial_logits: torch.Tensor,
        color_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = spatial_logits.shape
        k2 = self.kernel_size * self.kernel_size
        color_logits = color_logits.view(b, self.out_channels, self.in_channels, h, w)

        color = torch.softmax(color_logits + self.color_prior.to(color_logits), dim=2)
        if self.spatial_pairwise:
            spatial_logits = spatial_logits.view(b, self.out_channels, self.in_channels, k2, h, w)
            spatial = torch.softmax(spatial_logits + self.spatial_prior.to(spatial_logits), dim=3)
            effective = (color.unsqueeze(3) * spatial).reshape(
                b, self.out_channels, self.in_channels * k2, h * w
            )
        else:
            spatial_logits = spatial_logits.view(b, self.out_channels, k2, h, w)
            spatial = torch.softmax(spatial_logits + self.spatial_prior.to(spatial_logits), dim=2)
            effective = (color.unsqueeze(3) * spatial.unsqueeze(2)).reshape(
                b, self.out_channels, self.in_channels * k2, h * w
            )
        return spatial, color, effective

    def forward(
        self,
        x: torch.Tensor,
        spatial_logits: torch.Tensor,
        color_logits: torch.Tensor,
        use_dilation: bool = False,
        dilation: int = 2,
    ) -> torch.Tensor:
        patches = self._unfold(x, use_dilation=use_dilation, dilation=dilation)
        spatial, color, effective = self.compose_weights(spatial_logits, color_logits)
        if self.spatial_pairwise:
            y = torch.einsum("boihw,boikhw,bikhw->bohw", color, spatial, patches)
        else:
            y = torch.einsum("boihw,bokhw,bikhw->bohw", color, spatial, patches)
        self.last_spatial = spatial.detach()
        self.last_color = color.detach()
        self.last_effective_kernel = effective.detach()
        return y


class FactorizedKernelPredictionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        image_channels: int = 3,
    ):
        super().__init__()
        del hidden_channels
        self.kernel_size = kernel_size
        self.image_channels = image_channels
        self.spatial_head = nn.Conv2d(
            in_channels,
            image_channels * image_channels * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.color_head = nn.Conv2d(
            in_channels,
            image_channels * image_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spatial_head(feat), self.color_head(feat)


class SCFDIDBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        num_channels: int = 3,
        hidden_channels: int = 64,
        unfold_dilation: int = 2,
        color_diagonal_prior: bool = True,
        color_diag_prior_value: float = 4.0,
        color_offdiag_prior_value: float = -4.0,
        spatial_center_prior: bool = False,
        spatial_center_prior_value: float = 0.0,
        spatial_pairwise: bool = True,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        if grad_repr not in {"grad8", "grad4", "sobel"}:
            raise ValueError(f"Unsupported grad_repr: {grad_repr}")
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.unfold_dilation = unfold_dilation
        self.grad_repr = grad_repr
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        predictor_in = hidden_channels + gradient_context_channels(grad_repr)
        self.head = FactorizedKernelPredictionHead(
            predictor_in,
            hidden_channels,
            kernel_size=kernel_size,
            image_channels=num_channels,
        )
        self.filter = StructureColorFactorizedDynamicFilter(
            kernel_size=kernel_size,
            in_channels=num_channels,
            out_channels=num_channels,
            color_diagonal_prior=color_diagonal_prior,
            color_diag_prior_value=color_diag_prior_value,
            color_offdiag_prior_value=color_offdiag_prior_value,
            spatial_center_prior=spatial_center_prior,
            spatial_center_prior_value=spatial_center_prior_value,
            spatial_pairwise=spatial_pairwise,
            grad_repr=grad_repr,
        )
        self.gca = GlobalChannelAttention(num_channels, hidden_channels)
        self.last_kernels: Optional[torch.Tensor] = None
        self.prev_kernels: Optional[torch.Tensor] = None
        self.last_spatial: Optional[torch.Tensor] = None
        self.last_color: Optional[torch.Tensor] = None

    def forward(
        self,
        inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        use_dilation: bool = True,
        mix_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_in, resi, prev_kernels = inp
        b, _, _, _ = x_in.shape
        features = self.feature_extractor(rms_norm(x_in))
        if resi is not None:
            features = self.gca(features, cond=resi)
        features = torch.cat([features, compute_gradient_context(x_in, self.grad_repr)], dim=1)
        features = rms_norm(features)

        spatial_logits, color_logits = self.head(features)
        filtered = self.filter(
            x_in,
            spatial_logits,
            color_logits,
            use_dilation=use_dilation,
            dilation=self.unfold_dilation,
        )
        self.prev_kernels = prev_kernels
        self.last_kernels = self.filter.last_effective_kernel.clone()
        self.last_spatial = self.filter.last_spatial
        self.last_color = self.filter.last_color

        if self.training:
            mix_alpha = torch.rand((b, 1, 1, 1), device=x_in.device)
            out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered
        else:
            out = filtered if mix_alpha is None else mix_alpha * x_in + (1.0 - mix_alpha) * filtered

        if use_dilation:
            return out, x_in - out, self.prev_kernels
        return out, x_in - out, self.last_kernels.clone()


class IDFGrad8SCFKernelNet(nn.Module):
    def __init__(
        self,
        num_iter: int = 10,
        kernel_size: int = 3,
        num_channels: int = 3,
        hidden_channels: int = 64,
        halt_threshold: float = 0.015,
        lcm_type: str = "grad8",
        grad8_source: str = "current",
        grad8_use_abs: bool = True,
        grad8_normalize: bool = True,
        kernel_mode: str = "scf_factorized",
        color_diagonal_prior: bool = True,
        color_diag_prior_value: float = 4.0,
        color_offdiag_prior_value: float = -4.0,
        spatial_center_prior: bool = False,
        spatial_center_prior_value: float = 0.0,
        spatial_pairwise: bool = True,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        del lcm_type, grad8_source, grad8_use_abs, grad8_normalize, kernel_mode
        self.num_iter = num_iter
        self.halt_threshold = halt_threshold
        self.block = SCFDIDBlock(
            kernel_size=kernel_size,
            num_channels=num_channels,
            hidden_channels=hidden_channels,
            unfold_dilation=2,
            color_diagonal_prior=color_diagonal_prior,
            color_diag_prior_value=color_diag_prior_value,
            color_offdiag_prior_value=color_offdiag_prior_value,
            spatial_center_prior=spatial_center_prior,
            spatial_center_prior_value=spatial_center_prior_value,
            spatial_pairwise=spatial_pairwise,
            grad_repr=grad_repr,
        )

    def forward(
        self,
        x: torch.Tensor,
        adaptive_iter: bool = False,
        max_iter: Optional[int] = None,
        alpha_schedule: Optional[List[float]] = None,
    ) -> torch.Tensor:
        output = (x, None, None)
        if max_iter is None:
            max_iter = self.num_iter
        for i in range(max_iter):
            mix_alpha = alpha_schedule[i] if alpha_schedule is not None else None
            output = self.block(output, use_dilation=(i % 2 == 0), mix_alpha=mix_alpha)
            if adaptive_iter and i % 2 == 1 and i > 1:
                loss = self._compute_adaptive_loss(self.block)
                if loss < self.halt_threshold:
                    break
        return output[0]

    def _compute_adaptive_loss(self, block: SCFDIDBlock) -> float:
        kernels = block.last_kernels
        prev_kernels = block.prev_kernels
        if kernels is None or prev_kernels is None:
            return float("inf")
        avg_kernel = kernels.mean(dim=-1)
        prev_avg_kernel = prev_kernels.mean(dim=-1)
        center_idx = block.kernel_size * block.kernel_size // 2
        same_channel_centers = [
            ch * block.kernel_size * block.kernel_size + center_idx
            for ch in range(block.num_channels)
        ]
        center = torch.as_tensor(same_channel_centers, device=avg_kernel.device)
        return F.l1_loss(avg_kernel[:, :, center], prev_avg_kernel[:, :, center]).item()

    def get_diagnostics(self) -> dict[str, torch.Tensor]:
        spatial = self.block.last_spatial
        color = self.block.last_color
        kernels = self.block.last_kernels
        stats: dict[str, torch.Tensor] = {}
        if spatial is not None:
            spatial_dim = 3 if spatial.dim() == 6 else 2
            stats["spatial_kernel_entropy"] = -(
                spatial.clamp_min(1e-12) * spatial.clamp_min(1e-12).log()
            ).sum(dim=spatial_dim).mean()
            stats["spatial_center_weight"] = spatial.select(spatial_dim, spatial.shape[spatial_dim] // 2).mean()
            stats["spatial_max_weight"] = spatial.max(dim=spatial_dim).values.mean()
            stats["spatial_min_weight"] = spatial.min(dim=spatial_dim).values.mean()
        if color is not None:
            c = color.shape[1]
            eye = torch.eye(c, device=color.device, dtype=color.dtype).view(1, c, c, 1, 1)
            diag = color * eye
            offdiag = color * (1.0 - eye)
            stats["color_diag_mass"] = diag.sum(dim=2).mean()
            stats["color_offdiag_mass"] = offdiag.sum(dim=2).mean()
            stats["color_entropy"] = -(
                color.clamp_min(1e-12) * color.clamp_min(1e-12).log()
            ).sum(dim=2).mean()
            stats["color_identity_deviation"] = (color - eye).abs().mean()
        if kernels is not None:
            b, cout, cin_kk, hw = kernels.shape
            c = cout
            kk = cin_kk // c
            center_idx = kk // 2
            kernels_reshaped = kernels.view(b, cout, c, kk, hw)
            same_mask = torch.eye(c, device=kernels.device, dtype=torch.bool).view(1, c, c, 1, 1)
            stats["effective_kernel_entropy"] = -(
                kernels.clamp_min(1e-12) * kernels.clamp_min(1e-12).log()
            ).sum(dim=2).mean()
            stats["effective_center_weight"] = torch.stack(
                [kernels_reshaped[:, ch, ch, center_idx, :] for ch in range(c)], dim=1
            ).mean()
            stats["effective_offdiag_rgb_mass"] = kernels_reshaped.masked_fill(same_mask, 0).sum(dim=(2, 3)).mean()
            stats["params"] = kernels.new_tensor(float(sum(p.numel() for p in self.parameters())))
            stats["trainable_params"] = kernels.new_tensor(float(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        return stats


class StructuredDIDBlock(nn.Module):
    """IDF DIDBlock with configurable LCM context and RGB-aware kernels.

    lcm_type:
        corr   - original local correlation module.
        grad8  - eight-direction gradient context computed from current x_t.

    kernel_mode:
        spatial - original channel-independent 3x3 dynamic kernel.
        rgb3d   - constrained 3x3x3 RGB-aware kernel with diagonal prior.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        num_channels: int = 3,
        hidden_channels: int = 64,
        unfold_dilation: int = 2,
        power_alpha: float = 2.0,
        lcm_type: str = "corr",
        kernel_mode: str = "spatial",
        offdiag_gate_max: float = 0.1,
        offdiag_gate_init: float = -6.0,
        rgb3d_variant: str = "diag_gate",
        rgb3d_norm: str = "power",
        use_outer_ring_bias: bool = False,
        outer_ring_bias_init: float = -4.0,
        spatial_mod_per_output: bool = True,
        use_spatial_grad_bias: bool = False,
        spatial_grad_bias_beta: float = 0.5,
        spatial_delta_max: float = 0.5,
        spatial_mod_kl_weight: float = 0.0,
        spatial_mod_tv_weight: float = 0.0,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        if lcm_type in {"grad4", "sobel"}:
            grad_repr = lcm_type
            lcm_type = "grad8"
        if lcm_type not in {"corr", "grad8", "zero"}:
            raise ValueError(f"Unsupported lcm_type: {lcm_type}")
        if grad_repr not in {"grad8", "grad4", "sobel"}:
            raise ValueError(f"Unsupported grad_repr: {grad_repr}")
        if kernel_mode not in {"spatial", "rgb3d", "rgb3d_spatial_mod"}:
            raise ValueError(f"Unsupported kernel_mode: {kernel_mode}")
        if num_channels < 2 and kernel_mode in {"rgb3d", "rgb3d_spatial_mod"}:
            raise ValueError("rgb3d kernel mode requires at least two channels.")

        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.unfold_dilation = unfold_dilation
        self.reflect_pad = nn.ReflectionPad2d(same_padding(kernel_size, dilation=unfold_dilation))
        self.power_alpha = power_alpha
        self.lcm_type = lcm_type
        self.grad_repr = grad_repr
        self.kernel_mode = kernel_mode
        self.offdiag_gate_max = offdiag_gate_max
        self.offdiag_gate_init = offdiag_gate_init
        self.rgb3d_variant = rgb3d_variant
        self.rgb3d_norm = rgb3d_norm
        self.use_outer_ring_bias = use_outer_ring_bias
        self.outer_ring_bias_init = outer_ring_bias_init
        self.spatial_mod_per_output = spatial_mod_per_output
        self.use_spatial_grad_bias = use_spatial_grad_bias
        self.spatial_grad_bias_beta = spatial_grad_bias_beta
        self.spatial_delta_max = spatial_delta_max
        self.spatial_mod_kl_weight = spatial_mod_kl_weight
        self.spatial_mod_tv_weight = spatial_mod_tv_weight

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        lcm_channels = gradient_context_channels(grad_repr) if lcm_type == "grad8" else kernel_size * kernel_size
        predictor_in = hidden_channels + lcm_channels

        if kernel_mode == "spatial":
            self.kernel_predictor = nn.Conv2d(
                predictor_in,
                kernel_size * kernel_size,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        elif kernel_mode == "rgb3d":
            self.rgb3d_head = RGB3DKernelHead(
                predictor_in,
                num_channels,
                kernel_size,
                power_alpha=power_alpha,
                variant=rgb3d_variant,
                norm=rgb3d_norm,
                offdiag_gate_max=offdiag_gate_max,
                offdiag_gate_init=offdiag_gate_init,
                use_outer_ring_bias=use_outer_ring_bias,
                outer_ring_bias_init=outer_ring_bias_init,
            )
        else:
            self.rgb3d_head = RGB3DSpatialModKernelHead(
                predictor_in,
                num_channels,
                kernel_size,
                power_alpha=power_alpha,
                variant=rgb3d_variant,
                norm=rgb3d_norm,
                offdiag_gate_max=offdiag_gate_max,
                offdiag_gate_init=offdiag_gate_init,
                use_outer_ring_bias=use_outer_ring_bias,
                outer_ring_bias_init=outer_ring_bias_init,
                spatial_mod_per_output=spatial_mod_per_output,
                spatial_delta_max=spatial_delta_max,
            )

        self.gca = GlobalChannelAttention(num_channels, hidden_channels)

        self.last_kernels: Optional[torch.Tensor] = None
        self.prev_kernels: Optional[torch.Tensor] = None
        self.last_offdiag_gate: Optional[torch.Tensor] = None
        self.last_spatial_mod: Optional[torch.Tensor] = None
        self.last_spatial_bias: Optional[torch.Tensor] = None

    def _make_lcm_context(
        self,
        x_in: torch.Tensor,
        patches: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        if self.lcm_type == "corr":
            return compute_local_correlation(patches, image_size=image_size)
        if self.lcm_type == "zero":
            b = x_in.shape[0]
            h, w = image_size
            return x_in.new_zeros(b, self.kernel_size * self.kernel_size, h, w)
        return compute_gradient_context(x_in, self.grad_repr)

    def _predict_spatial_kernel(self, features: torch.Tensor, b: int) -> torch.Tensor:
        kk = self.kernel_size * self.kernel_size
        kernel_logits = self.kernel_predictor(features)
        kernels = kernel_logits.view(b, kk, -1)
        return _power_norm_sum1(kernels, dim=1, alpha=self.power_alpha)

    def _predict_rgb3d_kernel(
        self,
        features: torch.Tensor,
        b: int,
        spatial_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kernel_mode == "rgb3d_spatial_mod":
            kernels = self.rgb3d_head(features, spatial_bias=spatial_bias)
        else:
            kernels = self.rgb3d_head(features)
        self.last_offdiag_gate = self.rgb3d_head.last_offdiag_gate
        self.last_spatial_mod = getattr(self.rgb3d_head, "last_spatial_mod", None)
        self.last_spatial_bias = getattr(self.rgb3d_head, "last_spatial_bias", None)
        return kernels

    def forward(
        self,
        inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        use_dilation: bool = True,
        mix_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_in, resi, prev_kernels = inp
        b, c, h, w = x_in.shape
        kk = self.kernel_size * self.kernel_size

        padding, dilation = (0, self.unfold_dilation) if use_dilation else (
            same_padding(self.kernel_size),
            1,
        )
        src = self.reflect_pad(x_in) if use_dilation else x_in
        patches = F.unfold(src, kernel_size=self.kernel_size, padding=padding, dilation=dilation)
        patches = patches.view(b, c, kk, -1)

        lcm_context = self._make_lcm_context(x_in, patches, image_size=(h, w))

        x_normed = rms_norm(x_in)
        features = self.feature_extractor(x_normed)
        if resi is not None:
            features = self.gca(features, cond=resi)

        features = torch.cat([features, lcm_context], dim=1)
        features = rms_norm(features)

        if self.kernel_mode == "spatial":
            kernels = self._predict_spatial_kernel(features, b)
            filtered = (patches * kernels.unsqueeze(1)).sum(dim=2).view(b, c, h, w)
        else:
            spatial_bias = None
            if self.kernel_mode == "rgb3d_spatial_mod" and self.use_spatial_grad_bias:
                spatial_bias = compute_spatial_gradient_bias(
                    x_in,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    beta=self.spatial_grad_bias_beta,
                    grad_repr=self.grad_repr,
                )
            kernels = self._predict_rgb3d_kernel(features, b, spatial_bias=spatial_bias)
            filtered = _apply_dynamic_rgb3d_kernel(patches, kernels, h, w)

        self.prev_kernels = prev_kernels
        self.last_kernels = kernels.clone()

        if self.training:
            mix_alpha = torch.rand((b, 1, 1, 1), device=x_in.device)
            out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered
        else:
            if mix_alpha is None:
                out = filtered
            else:
                out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered

        if use_dilation:
            return out, x_in - out, self.prev_kernels
        return out, x_in - out, self.last_kernels.clone()


class DensityFusionDIDBlock(nn.Module):
    """Grad8 RGB3D 3x3/5x5 dynamic filtering with density-prior fusion."""

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        unfold_dilation: int = 2,
        power_alpha: float = 2.0,
        offdiag_gate_max: float = 0.1,
        offdiag_gate_init: float = -6.0,
        large_kernel_size: int = 5,
        use_outer_ring_bias: bool = True,
        outer_ring_bias_init: float = -4.0,
        density_g_max: float = 0.3,
        density_gate_init: float = -5.0,
        grad_repr: str = "grad8",
    ):
        super().__init__()
        if grad_repr not in {"grad8", "grad4", "sobel"}:
            raise ValueError(f"Unsupported grad_repr: {grad_repr}")
        self.kernel_size = large_kernel_size
        self.small_kernel_size = 3
        self.large_kernel_size = large_kernel_size
        self.num_channels = num_channels
        self.unfold_dilation = unfold_dilation
        self.power_alpha = power_alpha
        self.density_g_max = density_g_max
        self.grad_repr = grad_repr
        self.reflect_pad3 = nn.ReflectionPad2d(same_padding(3, dilation=unfold_dilation))
        self.reflect_pad5 = nn.ReflectionPad2d(same_padding(large_kernel_size, dilation=unfold_dilation))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        predictor_in = hidden_channels + gradient_context_channels(grad_repr)
        self.head3 = RGB3DKernelHead(
            predictor_in,
            num_channels,
            3,
            power_alpha=power_alpha,
            variant="diag_gate",
            norm="power",
            offdiag_gate_max=offdiag_gate_max,
            offdiag_gate_init=offdiag_gate_init,
        )
        self.head5 = RGB3DKernelHead(
            predictor_in,
            num_channels,
            large_kernel_size,
            power_alpha=power_alpha,
            variant="diag_gate",
            norm="softmax",
            offdiag_gate_max=offdiag_gate_max,
            offdiag_gate_init=offdiag_gate_init,
            use_outer_ring_bias=use_outer_ring_bias,
            outer_ring_bias_init=outer_ring_bias_init,
        )
        self.density_prior = DensityPriorNet(hidden_channels=max(16, hidden_channels // 2), gate_init=density_gate_init)
        self.gca = GlobalChannelAttention(num_channels, hidden_channels)

        self.last_kernels: Optional[torch.Tensor] = None
        self.prev_kernels: Optional[torch.Tensor] = None
        self.last_kernels3: Optional[torch.Tensor] = None
        self.last_kernels5: Optional[torch.Tensor] = None
        self.last_density_gate: Optional[torch.Tensor] = None
        self.last_offdiag_gate: Optional[torch.Tensor] = None

    def _unfold(self, x: torch.Tensor, kernel_size: int, use_dilation: bool) -> torch.Tensor:
        padding, dilation = (0, self.unfold_dilation) if use_dilation else (same_padding(kernel_size), 1)
        if use_dilation:
            src = self.reflect_pad3(x) if kernel_size == 3 else self.reflect_pad5(x)
        else:
            src = x
        patches = F.unfold(src, kernel_size=kernel_size, padding=padding, dilation=dilation)
        b, c, _, _ = x.shape
        return patches.view(b, c, kernel_size * kernel_size, -1)

    def _embed_k3_to_k5(self, kernels3: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, c, _, hw = kernels3.shape
        kk3 = 9
        kk5 = self.large_kernel_size * self.large_kernel_size
        kernels3 = kernels3.view(b, c, c, kk3, hw)
        embedded = kernels3.new_zeros(b, c, c, kk5, hw)
        small_idx = 0
        center = self.large_kernel_size // 2
        for yy in range(center - 1, center + 2):
            for xx in range(center - 1, center + 2):
                large_idx = yy * self.large_kernel_size + xx
                embedded[:, :, :, large_idx, :] = kernels3[:, :, :, small_idx, :]
                small_idx += 1
        return embedded.view(b, c, c * kk5, hw)

    def forward(
        self,
        inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        use_dilation: bool = True,
        mix_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_in, resi, prev_kernels = inp
        b, c, h, w = x_in.shape

        features = self.feature_extractor(rms_norm(x_in))
        if resi is not None:
            features = self.gca(features, cond=resi)
        features = torch.cat([features, compute_gradient_context(x_in, self.grad_repr)], dim=1)
        features = rms_norm(features)

        patches3 = self._unfold(x_in, 3, use_dilation=use_dilation)
        kernels3 = self.head3(features)
        y3 = _apply_dynamic_rgb3d_kernel(patches3, kernels3, h, w)

        patches5 = self._unfold(x_in, self.large_kernel_size, use_dilation=use_dilation)
        kernels5 = self.head5(features)
        y5 = _apply_dynamic_rgb3d_kernel(patches5, kernels5, h, w)

        gate = self.density_g_max * torch.sigmoid(self.density_prior(x_in, resi))
        filtered = (1.0 - gate) * y3 + gate * y5

        kernels3_embedded = self._embed_k3_to_k5(kernels3, h, w)
        gate_flat = gate.view(b, 1, 1, h * w)
        combined_kernels = (1.0 - gate_flat) * kernels3_embedded + gate_flat * kernels5

        self.prev_kernels = prev_kernels
        self.last_kernels = combined_kernels.clone()
        self.last_kernels3 = kernels3.detach()
        self.last_kernels5 = kernels5.detach()
        self.last_density_gate = gate.detach()
        self.last_offdiag_gate = self.head5.last_offdiag_gate

        if self.training:
            mix_alpha = torch.rand((b, 1, 1, 1), device=x_in.device)
            out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered
        else:
            out = filtered if mix_alpha is None else mix_alpha * x_in + (1.0 - mix_alpha) * filtered

        if use_dilation:
            return out, x_in - out, self.prev_kernels
        return out, x_in - out, self.last_kernels.clone()


class IDFStructuredNet(nn.Module):
    """Original IDF loop with configurable Grad8 LCM and constrained RGB3D kernels."""

    def __init__(
        self,
        num_iter: int = 10,
        kernel_size: int = 3,
        num_channels: int = 3,
        hidden_channels: int = 64,
        halt_threshold: float = 0.015,
        lcm_type: str = "corr",
        kernel_mode: str = "spatial",
        use_density_fusion: bool = False,
        **block_kwargs,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.halt_threshold = halt_threshold
        if use_density_fusion:
            self.block = DensityFusionDIDBlock(
                num_channels=num_channels,
                hidden_channels=hidden_channels,
                unfold_dilation=2,
                power_alpha=block_kwargs.pop("power_alpha", 2.0),
                **block_kwargs,
            )
        else:
            self.block = StructuredDIDBlock(
                kernel_size=kernel_size,
                num_channels=num_channels,
                hidden_channels=hidden_channels,
                unfold_dilation=2,
                lcm_type=lcm_type,
                kernel_mode=kernel_mode,
                **block_kwargs,
            )

    def forward(
        self,
        x: torch.Tensor,
        adaptive_iter: bool = False,
        max_iter: Optional[int] = None,
        alpha_schedule: Optional[List[float]] = None,
    ) -> torch.Tensor:
        output = (x, None, None)

        if max_iter is None:
            max_iter = self.num_iter

        for i in range(max_iter):
            mix_alpha = alpha_schedule[i] if alpha_schedule is not None else None
            output = self.block(output, use_dilation=(i % 2 == 0), mix_alpha=mix_alpha)
            if adaptive_iter and i % 2 == 1 and i > 1:
                loss = self._compute_adaptive_loss(self.block)
                if loss < self.halt_threshold:
                    break

        return output[0]

    def get_diagnostics(self) -> dict[str, torch.Tensor]:
        block = self.block
        kernels = getattr(block, "last_kernels", None)
        stats: dict[str, torch.Tensor] = {}
        gate = getattr(block, "last_density_gate", None)
        if gate is not None:
            stats["density_gate_mean"] = gate.mean()
            stats["density_gate_p90"] = torch.quantile(gate.flatten().float(), 0.9)
        spatial_mod = getattr(block, "last_spatial_mod", None)
        if spatial_mod is not None:
            spatial_dim = 2
            stats["spatial_mod_entropy"] = -(
                spatial_mod.clamp_min(1e-12) * spatial_mod.clamp_min(1e-12).log()
            ).sum(dim=spatial_dim).mean()
            stats["spatial_mod_center_weight"] = spatial_mod.select(spatial_dim, spatial_mod.shape[spatial_dim] // 2).mean()
            stats["spatial_mod_max_weight"] = spatial_mod.max(dim=spatial_dim).values.mean()
            stats["spatial_mod_min_weight"] = spatial_mod.min(dim=spatial_dim).values.mean()
        spatial_bias = getattr(block, "last_spatial_bias", None)
        if spatial_bias is not None:
            stats["spatial_bias_mean"] = spatial_bias.mean()
            stats["spatial_bias_min"] = spatial_bias.min()
            stats["spatial_bias_max"] = spatial_bias.max()
        if kernels is None or kernels.ndim != 4:
            return stats

        b, cout, cin_kk, hw = kernels.shape
        c = getattr(block, "num_channels", cout)
        kk = cin_kk // c
        k = int(kk ** 0.5)
        kernels_reshaped = kernels.view(b, cout, c, kk, hw)

        same_mask = torch.eye(c, device=kernels.device, dtype=torch.bool).view(1, c, c, 1, 1)
        offdiag_mass = kernels_reshaped.masked_fill(same_mask, 0).sum(dim=(2, 3)).mean()
        stats["offdiag_rgb_mass"] = offdiag_mass

        center_idx = kk // 2
        center_weights = []
        for ch in range(min(cout, c)):
            center_weights.append(kernels_reshaped[:, ch, ch, center_idx, :])
        if center_weights:
            stats["center_weight"] = torch.stack(center_weights, dim=1).mean()

        entropy = -(kernels.clamp_min(1e-12) * kernels.clamp_min(1e-12).log()).sum(dim=2).mean()
        stats["kernel_entropy"] = entropy

        if k >= 5:
            coords = torch.arange(k, device=kernels.device)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            inner_lo = k // 2 - 1
            inner_hi = k // 2 + 1
            inner = ((yy >= inner_lo) & (yy <= inner_hi) & (xx >= inner_lo) & (xx <= inner_hi)).view(1, 1, 1, kk, 1)
            inner_mass = kernels_reshaped.masked_fill(~inner, 0).sum(dim=(2, 3)).mean()
            outer_mass = kernels_reshaped.masked_fill(inner, 0).sum(dim=(2, 3)).mean()
            stats["inner_3x3_weight_mass"] = inner_mass
            stats["outer_ring_weight_mass"] = outer_mass
        else:
            stats["inner_3x3_weight_mass"] = kernels_reshaped.sum(dim=(2, 3)).mean()
            stats["outer_ring_weight_mass"] = kernels.new_tensor(0.0)
        return stats

    def get_extra_losses(self) -> dict[str, torch.Tensor]:
        block = self.block
        spatial_mod = getattr(block, "last_spatial_mod", None)
        if spatial_mod is None:
            return {}

        losses: dict[str, torch.Tensor] = {}
        kl_weight = float(getattr(block, "spatial_mod_kl_weight", 0.0))
        tv_weight = float(getattr(block, "spatial_mod_tv_weight", 0.0))

        if kl_weight > 0:
            kk = spatial_mod.shape[2]
            uniform_log = -math.log(float(kk))
            kl = (
                spatial_mod.clamp_min(1e-12)
                * (spatial_mod.clamp_min(1e-12).log() - uniform_log)
            ).sum(dim=2).mean()
            losses["spatial_mod_kl"] = spatial_mod.new_tensor(kl_weight) * kl

        if tv_weight > 0 and spatial_mod.dim() == 5:
            tv_h = (spatial_mod[..., 1:, :] - spatial_mod[..., :-1, :]).abs().mean()
            tv_w = (spatial_mod[..., :, 1:] - spatial_mod[..., :, :-1]).abs().mean()
            losses["spatial_mod_tv"] = spatial_mod.new_tensor(tv_weight) * (tv_h + tv_w)

        return losses

    def _compute_adaptive_loss(self, block: StructuredDIDBlock) -> float:
        kernels = block.last_kernels
        prev_kernels = block.prev_kernels
        if kernels is None or prev_kernels is None:
            return float("inf")

        avg_kernel = kernels.mean(dim=-1)
        prev_avg_kernel = prev_kernels.mean(dim=-1)
        center_idx = block.kernel_size * block.kernel_size // 2

        if block.kernel_mode == "spatial":
            loss = F.l1_loss(avg_kernel[:, center_idx], prev_avg_kernel[:, center_idx])
        else:
            same_channel_centers = [
                ch * block.kernel_size * block.kernel_size + center_idx
                for ch in range(block.num_channels)
            ]
            center = torch.as_tensor(same_channel_centers, device=avg_kernel.device)
            loss = F.l1_loss(avg_kernel[:, :, center], prev_avg_kernel[:, :, center])

        return loss.item()








