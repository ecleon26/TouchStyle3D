"""TactStyle modified Stable-Diffusion VAE for heightfield generation.

The architecture is described in Section 5.2 of the paper:

* Backbone: the VAE (`AutoencoderKL`) of Stable Diffusion v1.4.
* The decoder's final ``conv_out`` is replaced with a ``Sequential`` that
  appends four additional convolutional layers (to learn heightfield-specific
  features) plus a final 3->1 convolution that produces a single-channel
  grayscale heightfield image.
* During fine-tuning the encoder is frozen; only ``decoder.conv_out``,
  ``quant_conv``, and ``post_quant_conv`` are updated.
"""

from __future__ import annotations

import torch
from torch import nn
from diffusers import AutoencoderKL


SD_VAE_PRETRAINED = "CompVis/stable-diffusion-v1-4"


def _build_heightfield_head() -> nn.Sequential:
    """Four feature layers + a final 3->1 grayscale layer."""
    return nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
    )


def build_tactstyle_vae(pretrained: str = SD_VAE_PRETRAINED) -> AutoencoderKL:
    """Load the SD v1.4 VAE and attach the heightfield head to its decoder."""
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    head = _build_heightfield_head()

    # Wrap the existing final decoder layer with the new head so that the
    # original conv_out (and its pre-trained weights) is preserved.
    vae.decoder.conv_out = nn.Sequential(vae.decoder.conv_out, *head)
    return vae


def freeze_encoder(vae: AutoencoderKL) -> None:
    """Freeze every parameter, then unfreeze the decoder head + quant convs.

    Matches the training procedure described in Section 5.3 of the paper.
    """
    for param in vae.parameters():
        param.requires_grad = False

    for name, param in vae.named_parameters():
        if (
            "decoder.conv_out" in name
            or "quant_conv" in name
            or "post_quant_conv" in name
        ):
            param.requires_grad = True


def normalize_output(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid + min-max normalize the VAE output to [0, 1]."""
    out = torch.sigmoid(logits)
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out
