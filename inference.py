"""Load a trained VecFontSDF and reconstruct a glyph image.

The model is class-conditional, so every input glyph needs a character label
(which letter it is). Supply it with --char for a single file, or name the files
after the glyph (e.g. A.png, g.png, or the ASCII codepoint 65.png) for a
directory.

Usage:
    python3 inference.py \
        --ckpt experiments/vecfontsdf/checkpoints/latest.pth \
        --input some_glyph.png --char A \
        --render_size 256 --save_params
"""

import argparse
import os
import string
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from losses import build_grid, gamma_rasterize, pseudo_distance
from model import VecFontSDF
from options import get_recon_parser
from sdf2svg import params_to_svg


# Class label ordering: A-Z -> 0..25, a-z -> 26..51. Index == one-hot hot index.
LABEL_CHARS = string.ascii_uppercase + string.ascii_lowercase


def char_to_label(ch: str) -> int:
    if ch not in LABEL_CHARS:
        raise ValueError(f'unsupported glyph {ch!r}; only A-Z and a-z are modeled')
    return LABEL_CHARS.index(ch)


def resolve_char(path: str, explicit: str) -> str:
    """Figure out which character a glyph image represents."""
    if explicit:
        return explicit
    stem = os.path.splitext(os.path.basename(path))[0]
    if len(stem) == 1 and stem in LABEL_CHARS:
        return stem
    if stem.isdigit():                      # ASCII codepoint, e.g. 65 -> 'A'
        return chr(int(stem))
    raise ValueError(
        f'cannot infer the character for {path!r}; pass --char or name the file '
        f'like A.png / g.png / 65.png')


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

    model = VecFontSDF(opts.fc_channel, opts.v_dim, opts.p_dim,
                       opts.char_categories).to(device)
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
    img = Image.open(path).convert('L')
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)  # [1, 1, H, W]


def main():
    cli = get_recon_parser()
    cli.add_argument('--ckpt', type=str, required=True)
    cli.add_argument('--input', type=str, required=True,
                     help='path to a glyph image, or a directory of .png files')
    cli.add_argument('--char', type=str, default=None,
                     help='which character the input glyph is (A-Z / a-z); '
                          'required for a single file unless inferable from its name')
    cli.add_argument('--render_size', type=int, default=None,
                     help='output resolution; defaults to the training image_size')
    cli.add_argument('--save_params', action='store_true',
                     help='also dump the (k,p,q,d,e,f) parabolic curve params as .npy')
    cli.add_argument('--save_svg', action='store_true',
                     help='also convert the predicted curves to a vector .svg glyph')
    cli.add_argument('--svg_merge', type=float, default=0.02,
                     help='endpoint-merge threshold for stitching SVG contours')
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
        ch = resolve_char(path, opts.char)
        label = char_to_label(ch)
        clss = torch.zeros(1, opts.char_categories, device=device)
        clss[0, label] = 1.0

        img = load_image(path, opts.image_size).to(device)
        params = model(img, clss)                            # [1, v*a, 6]
        recon = render_at(params, render_size, opts.gamma,
                          opts.v_dim, opts.p_dim, device)    # [1, 1, R, R]

        stem = os.path.splitext(os.path.basename(path))[0]
        # Resize the input to render_size and concat side by side with the recon.
        gt_resized = torch.nn.functional.interpolate(
            img, size=render_size, mode='bilinear', align_corners=False)
        pair = torch.cat([gt_resized, recon], dim=-1)
        save_image(pair, os.path.join(out_dir, f'{stem}_recon.png'),
                   normalize=False)
        print(f'[{stem}] char={ch!r} -> {out_dir}/{stem}_recon.png')

        if opts.save_params:
            np.save(os.path.join(out_dir, f'{stem}_params.npy'),
                    params[0].detach().cpu().numpy())

        if opts.save_svg:
            svg_path = os.path.join(out_dir, f'{stem}.svg')
            try:
                params_to_svg(params[0].detach().cpu().numpy(),
                              opts.v_dim, opts.p_dim, svg_path, merge=opts.svg_merge)
                print(f'         svg  -> {svg_path}')
            except Exception as ex:
                # The contour-stitching geometry has rare data-dependent failures;
                # skip this glyph's SVG rather than aborting the whole run.
                print(f'         svg  FAILED for {stem}: {type(ex).__name__}: {ex}')


if __name__ == '__main__':
    main()
