"""Texture / heightfield dataset used by TactStyle.

The original paper trains on the CGAxis "PBR 20 Parquets" repository, which is
distributed under a commercial license and is therefore *not* included with
this code. To use this code you can either:

1. Purchase / download the CGAxis textures and arrange them as::

       <root_dir>/
         4K_Physical_Parquets/<texture_name>/<texture_name>_diffuse.jpg
         4K_Physical_Parquets/<texture_name>/<texture_name>_height.jpg
         4K_Physical_Wood/...
         4K_Physical_Rocks/...
         4K_Physical_Walls/...
         4K_Physical_Roofs/...

2. Use a freely available alternative such as **MatSynth**
   (https://gvecchio.com/matsynth) which provides ~4k PBR materials
   including diffuse and height maps. ``MatSynthAdapter`` below shows the
   minimum interface ``train.py`` expects, so any dataset providing
   ``(diffuse_3xHxW, height_1xHxW)`` tensors will work.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


IMG_SIZE = 256
DIFFUSE_SUFFIX = "diffuse.jpg"
HEIGHT_SUFFIX = "height.jpg"


class ParquetDataset(Dataset):
    """Pairs of (diffuse texture, heightfield) loaded from the CGAxis layout.

    ``root_dir`` should contain one or more category sub-folders, each of which
    contains per-texture folders with files ending in ``_diffuse.jpg`` and
    ``_height.jpg``.

    Set ``augment=True`` to apply the four 90deg rotations described in
    Section 5.4 of the paper, multiplying the dataset size by 4.
    """

    def __init__(self, root_dir: str | os.PathLike, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.image_pairs = self._gather_image_pairs()
        if not self.image_pairs:
            raise FileNotFoundError(
                f"No (diffuse, height) pairs found under {self.root_dir}. "
                "See dataset.py for the expected directory layout."
            )
        self.augment = augment
        self.rotations = (0, 90, 180, 270) if augment else (0,)

    def _gather_image_pairs(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for category in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for texture in sorted(p for p in category.iterdir() if p.is_dir()):
                diffuse = next(texture.glob(f"*{DIFFUSE_SUFFIX}"), None)
                height = next(texture.glob(f"*{HEIGHT_SUFFIX}"), None)
                if diffuse and height:
                    pairs.append((diffuse, height))
        return pairs

    def __len__(self) -> int:
        return len(self.image_pairs) * len(self.rotations)

    def __getitem__(self, idx: int):
        pair_idx, rot_idx = divmod(idx, len(self.rotations))
        diffuse_path, height_path = self.image_pairs[pair_idx]
        rotation = self.rotations[rot_idx]

        diffuse = Image.open(diffuse_path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS
        )
        height = Image.open(height_path).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS
        )

        if rotation:
            diffuse = diffuse.rotate(rotation)
            height = height.rotate(rotation)

        return transforms.ToTensor()(diffuse), transforms.ToTensor()(height)


class MatSynthAdapter(Dataset):
    """Minimal adapter for the MatSynth dataset.

    Expects each sample folder to contain a ``basecolor.png`` (or .jpg) and
    a ``height.png`` (or .jpg). MatSynth is released under CC-BY 4.0 and can
    be obtained from https://gvecchio.com/matsynth .
    """

    BASECOLOR_NAMES = ("basecolor.png", "basecolor.jpg", "diffuse.png", "diffuse.jpg")
    HEIGHT_NAMES = ("height.png", "height.jpg", "displacement.png")

    def __init__(self, root_dir: str | os.PathLike):
        self.root_dir = Path(root_dir)
        self.samples: list[tuple[Path, Path]] = []
        for sample in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            base = next((sample / n for n in self.BASECOLOR_NAMES if (sample / n).exists()), None)
            height = next((sample / n for n in self.HEIGHT_NAMES if (sample / n).exists()), None)
            if base and height:
                self.samples.append((base, height))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        base_path, height_path = self.samples[idx]
        base = Image.open(base_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        height = Image.open(height_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        return TF.to_tensor(base), TF.to_tensor(height)
