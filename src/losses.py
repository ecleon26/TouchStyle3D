"""SSIM loss used during fine-tuning.

The paper combines pixel-wise reconstruction with a structural-similarity term
so that the generated heightfields match the ground truth in both overall
intensity and local structure (Section 5.2).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    g_1d = _gaussian(window_size).unsqueeze(1)
    g_2d = g_1d @ g_1d.t()
    return g_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    val_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Standard SSIM between two batches of images, shape ``(B, C, H, W)``."""
    if val_range is None:
        max_val = 1.0 if torch.max(img1) <= 1 else 255.0
        min_val = 0.0
    else:
        min_val, max_val = val_range
    L = max_val - min_val

    pad = window_size // 2
    window = _create_window(window_size, img1.size(1)).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=pad, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=pad, groups=img2.size(1))

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=img1.size(1)) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean([1, 2, 3])


class SSIMLoss(nn.Module):
    """1 - SSIM, suitable as a minimization objective."""

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, self.window_size, self.size_average)
