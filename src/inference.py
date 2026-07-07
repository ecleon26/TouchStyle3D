"""Generate a tactile heightfield from a single texture image.

Example::

    python -m src.inference \\
        --checkpoint checkpoints/ckpt_vae_final.pth \\
        --texture examples/airpods/wood_diffuse.jpg \\
        --output  examples/airpods/wood_heightfield.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from .model import build_tactstyle_vae, normalize_output


IMG_SIZE = 256


@torch.no_grad()
def generate_heightfield(
    texture_image: Image.Image | str | Path,
    checkpoint_path: str | Path,
    device: str | None = None,
    output_size: int = IMG_SIZE,
) -> Image.Image:
    """Run the fine-tuned VAE on a single texture and return a grayscale PIL image."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tactstyle_vae().to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()

    if not isinstance(texture_image, Image.Image):
        texture_image = Image.open(texture_image)
    texture_image = texture_image.convert("RGB").resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )

    x = ToTensor()(texture_image).unsqueeze(0).to(device)
    pred = normalize_output(model(x).sample)
    return ToPILImage()(pred[0].cpu())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a heightfield from a texture image.")
    p.add_argument("--checkpoint", required=True, help="Path to the fine-tuned VAE state dict (.pth).")
    p.add_argument("--texture", required=True, help="Input texture image (RGB).")
    p.add_argument("--output", required=True, help="Where to save the generated heightfield (grayscale).")
    p.add_argument("--size", type=int, default=IMG_SIZE)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    height = generate_heightfield(args.texture, args.checkpoint, output_size=args.size)
    height.save(args.output)
    print(f"Wrote heightfield to {args.output}")
