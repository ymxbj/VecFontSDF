"""Load a trained VecFontSDF and reconstruct a glyph image.

Usage:
    python3 inference.py \
        --ckpt experiments/vecfontsdf/checkpoints/latest.pth \
        --input some_glyph.png \
        --out_dir ./recon_out
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from losses import build_grid, gamma_rasterize, pseudo_distance
from model import VecFontSDF
from options import get_recon_parser


def load_model(ckpt_path: str, device: torch.device,
               cli_opts: argparse.Namespace) -> Tuple[VecFontSDF, argparse.Namespace]:
    """Restore the model architecture from the sibling opts.txt, then load weights.

    If no opts.txt is found (manually managed checkpoint), fall back to the CLI
    hyperparameters.
    """
    opts = cli_opts
    opts_txt = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'opts.txt')
    if os.path.exists(opts_txt):
        with open(opts_txt) as f:
            for line in f:
                k, _, v = line.strip().partition(':')
                k, v = k.strip(), v.strip()
                if hasattr(opts, k) and v not in ('', 'None'):
                    cur = getattr(opts, k)
                    try:
                        setattr(opts, k, type(cur)(v) if cur is not None else v)
                    except (TypeError, ValueError):
                        pass

    model = VecFontSDF(opts.feat_dim, opts.fc_channel,
                       opts.v_dim, opts.p_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, opts


@torch.no_grad()
def render_at(params: torch.Tensor, image_size: int, gamma: float,
              v_dim: int, p_dim: int, device: torch.device) -> torch.Tensor:
    """Rasterize the predicted curve parameters at the requested resolution."""
    grid = build_grid(image_size, device).unsqueeze(0).expand(params.size(0), -1, -1)
    g = pseudo_distance(params, grid, v_dim, p_dim)
    return gamma_rasterize(g, gamma).view(params.size(0), 1, image_size, image_size)


def load_image(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)  # [1, 3, H, W]


def main():
    cli = get_recon_parser()
    cli.add_argument('--ckpt', type=str, required=True)
    cli.add_argument('--input', type=str, required=True,
                     help='path to a glyph image, or a directory of .png files')
    cli.add_argument('--render_size', type=int, default=None,
                     help='output resolution; defaults to the training image_size')
    cli.add_argument('--save_params', action='store_true',
                     help='also dump the (k,p,q,d,e,f) parabolic curve params as .npy')
    opts = cli.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, opts = load_model(opts.ckpt, device, opts)
    render_size = opts.render_size or opts.image_size

    out_dir = os.path.join(opts.out_dir, opts.experiment_name, 'inference')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(opts.input):
        files = sorted(f for f in os.listdir(opts.input)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
        paths = [os.path.join(opts.input, f) for f in files]
    else:
        paths = [opts.input]

    for path in paths:
        img = load_image(path, opts.image_size).to(device)
        params = model(img)                                  # [1, v*a, 6]
        recon = render_at(params, render_size, opts.gamma,
                          opts.v_dim, opts.p_dim, device)    # [1, 1, R, R]

        stem = os.path.splitext(os.path.basename(path))[0]
        # Downsample the input back to render_size, collapse RGB to gray so the
        # channel count matches the reconstruction, then concat side by side.
        gt_resized = torch.nn.functional.interpolate(
            img, size=render_size, mode='bilinear', align_corners=False)
        gt_gray = gt_resized.mean(dim=1, keepdim=True)
        pair = torch.cat([gt_gray, recon], dim=-1)
        save_image(pair, os.path.join(out_dir, f'{stem}_recon.png'),
                   normalize=False)
        print(f'[{stem}] -> {out_dir}/{stem}_recon.png')

        if opts.save_params:
            np.save(os.path.join(out_dir, f'{stem}_params.npy'),
                    params[0].detach().cpu().numpy())


if __name__ == '__main__':
    main()
