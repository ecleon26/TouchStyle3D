"""Fine-tune the modified VAE to map texture images to heightfields.

Reproduces the procedure described in Section 5.3 of the paper:

* SD v1.4 VAE backbone with heightfield head (see ``model.py``)
* Encoder frozen; only the decoder head + quant convs are trained
* RMSprop optimizer with split learning rates
* Loss: MSE (pixel-wise intensity) + SSIM (local structural similarity), as
  described in Section 5.2 of the paper
* ``ReduceLROnPlateau`` scheduler, gradient clipping at ``max_norm=1.0``
* Periodic checkpoints + sample grids of (input texture, generated height)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from .dataset import ParquetDataset
from .losses import SSIMLoss
from .model import build_tactstyle_vae, freeze_encoder, normalize_output


def _log_validation_grid(model, loader, out_dir: Path, epoch: int, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        diffuse, _ = next(iter(loader))
        diffuse = diffuse.to(device)
        prediction = normalize_output(model(diffuse).sample)
        grid = make_grid(
            torch.cat([diffuse.cpu(), prediction.repeat(1, 3, 1, 1).cpu()], dim=0),
            nrow=diffuse.shape[0],
            normalize=True,
            scale_each=True,
        )
        ToPILImage()(grid).save(out_dir / f"grid_epoch_{epoch:03d}.png")
    model.train()


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("tactstyle.train")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = ParquetDataset(args.data_root, augment=args.augment)
    train_size = int(args.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(
        full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    log.info("Train pairs: %d | Test pairs: %d", len(train_set), len(test_set))

    model = build_tactstyle_vae().to(device)
    freeze_encoder(model)

    mse = nn.MSELoss()
    ssim_loss = SSIMLoss(window_size=8).to(device)

    head = model.decoder.conv_out
    optimizer = optim.RMSprop(
        [
            {"params": head[0].parameters(), "lr": args.lr_existing},
            {"params": head[1].parameters(), "lr": args.lr_new},
            {"params": head[2].parameters(), "lr": args.lr_new},
            {"params": head[3].parameters(), "lr": args.lr_new},
            {"params": head[4].parameters(), "lr": args.lr_new},
            {"params": head[5].parameters(), "lr": args.lr_new},
            {"params": model.post_quant_conv.parameters(), "lr": args.lr_existing},
            {"params": model.quant_conv.parameters(), "lr": args.lr_existing},
        ]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    val_dir = save_dir / "validation"

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for diffuse, height in tqdm(train_loader, desc=f"epoch {epoch}"):
            diffuse, height = diffuse.to(device), height.to(device)
            optimizer.zero_grad()

            prediction = normalize_output(model(diffuse).sample)
            loss = args.mse_weight * mse(prediction, height) + args.ssim_weight * ssim_loss(prediction, height)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item() * diffuse.size(0)
            n += diffuse.size(0)

        avg = running / max(n, 1)
        scheduler.step(avg)
        log.info("Epoch %d | avg loss = %.4f", epoch, avg)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), save_dir / f"ckpt_vae_{epoch:03d}.pth")
            _log_validation_grid(model, test_loader, val_dir, epoch, device)

    torch.save(model.state_dict(), save_dir / "ckpt_vae_final.pth")
    log.info("Training finished. Checkpoints in %s", save_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TactStyle VAE on (diffuse, height) pairs.")
    p.add_argument("--data-root", required=True, help="Directory containing category sub-folders of textures.")
    p.add_argument("--save-dir", default="checkpoints", help="Where to write checkpoints + validation grids.")
    p.add_argument("--epochs", type=int, default=60, help="Paper uses 60 epochs.")
    p.add_argument("--batch-size", type=int, default=10, help="Paper uses 10.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr-existing", type=float, default=1e-5, help="LR for the original SD-VAE layers.")
    p.add_argument("--lr-new", type=float, default=1e-3, help="LR for the new heightfield head layers.")
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="4x rotational augmentation (Section 5.4 of the paper). Use --no-augment to disable.",
    )
    p.add_argument("--mse-weight", type=float, default=1.0, help="Weight on the MSE term in the loss.")
    p.add_argument("--ssim-weight", type=float, default=1.0, help="Weight on the (1 - SSIM) term in the loss.")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
