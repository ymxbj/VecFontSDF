"""Training entry point for VecFontSDF reconstruction.

Usage:
    python -m VecFontSDF.train --img_path ./data/img --sdf_path ./data/sdf \
                               --font_list ./data/font_list.txt
"""

import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torchvision.utils import save_image

from .dataloader import build_loaders
from .losses import build_grid, compute_losses
from .model import VecFontSDF
from .options import get_recon_parser


def cycle(loader):
    """Infinite iterator over a dataloader (for iteration-based training)."""
    while True:
        for batch in loader:
            yield batch


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_ckpt(model: nn.Module, optimizer: torch.optim.Optimizer,
              step: int, path: str) -> None:
    torch.save({
        'step': step,
        'model': _unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_ckpt(model: nn.Module, optimizer: torch.optim.Optimizer,
              path: str, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    _unwrap(model).load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return int(ckpt.get('step', 0))


@torch.no_grad()
def validate(model: nn.Module, val_loader, opts, grid_coords: torch.Tensor,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {'image': 0.0, 'grid': 0.0, 'contour': 0.0, 'regular': 0.0, 'total': 0.0}
    n = 0
    for batch in val_loader:
        img = batch['image'].to(device, non_blocking=True)
        grid_sdf = batch['grid_sdf'].to(device, non_blocking=True)
        contour_sdf = batch['contour_sdf'].to(device, non_blocking=True)
        params = model(img)
        losses, _ = compute_losses(params, img, grid_sdf, contour_sdf,
                                   opts, grid_coords)
        bs = img.size(0)
        for k in sums:
            sums[k] += float(losses[k].item()) * bs
        n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def main():
    opts = get_recon_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = os.path.join(opts.out_dir, opts.experiment_name)
    sample_dir = os.path.join(exp_dir, 'samples')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    for d in (sample_dir, ckpt_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(exp_dir, 'opts.txt'), 'w') as f:
        for k, v in vars(opts).items():
            f.write(f'{k}: {v}\n')

    train_loader, val_loader = build_loaders(opts)
    print(f'train samples: {len(train_loader.dataset)}, '
          f'val samples: {len(val_loader.dataset)}')

    model = VecFontSDF(opts.feat_dim, opts.fc_channel,
                       opts.v_dim, opts.p_dim).to(device)
    if opts.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr,
                                 betas=(opts.beta1, opts.beta2), eps=opts.eps,
                                 weight_decay=opts.weight_decay)

    start_step = 0
    if opts.resume:
        start_step = load_ckpt(model, optimizer, opts.resume, device)
        print(f'resumed from {opts.resume} at step {start_step}')

    writer = None
    if opts.tboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
        except ImportError:
            print('tensorboard not available, skipping logging')

    grid_coords = build_grid(opts.image_size, device)

    model.train()
    iterator = cycle(train_loader)
    t0 = time.time()

    for step in range(start_step + 1, opts.n_iters + 1):
        batch = next(iterator)
        img = batch['image'].to(device, non_blocking=True)
        grid_sdf = batch['grid_sdf'].to(device, non_blocking=True)
        contour_sdf = batch['contour_sdf'].to(device, non_blocking=True)

        params = model(img)
        losses, img_pred = compute_losses(params, img, grid_sdf, contour_sdf,
                                          opts, grid_coords)

        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        optimizer.step()

        if step % opts.log_every == 0:
            dt = time.time() - t0
            its = opts.log_every / max(dt, 1e-6)
            t0 = time.time()
            print(f'[{step:>7d}/{opts.n_iters}] '
                  f'total={losses["total"].item():.4f} '
                  f'img={losses["image"].item():.4f} '
                  f'grid={losses["grid"].item():.4f} '
                  f'contour={losses["contour"].item():.4f} '
                  f'reg={losses["regular"].item():.4f} '
                  f'k2={losses["k2_mean"].item():.3f} '
                  f'({its:.1f} it/s)')
            if writer is not None:
                for k, v in losses.items():
                    writer.add_scalar(f'train/{k}',
                                      v.item() if torch.is_tensor(v) else v, step)

        if step % opts.sample_every == 0:
            with torch.no_grad():
                # img is RGB [B, 3, H, W], img_pred is gray [B, 1, H, W];
                # collapse RGB to gray before stacking them along the height axis.
                gt_gray = img.detach().mean(dim=1, keepdim=True)
                pair = torch.cat([gt_gray, img_pred.detach()], dim=-2)
                save_image(pair[:32],
                           os.path.join(sample_dir, f'step_{step:07d}.png'),
                           nrow=8, normalize=False)

        if step % opts.val_every == 0:
            val_losses = validate(model, val_loader, opts, grid_coords, device)
            print('[val @ {}] '.format(step) +
                  '  '.join(f'{k}={v:.4f}' for k, v in val_losses.items()))
            if writer is not None:
                for k, v in val_losses.items():
                    writer.add_scalar(f'val/{k}', v, step)

        if step % opts.ckpt_every == 0 or step == opts.n_iters:
            save_ckpt(model, optimizer, step,
                      os.path.join(ckpt_dir, f'vecfontsdf_{step:07d}.pth'))
            save_ckpt(model, optimizer, step,
                      os.path.join(ckpt_dir, 'latest.pth'))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
