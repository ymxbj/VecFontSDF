"""Rasterize each SVG glyph into a PNG that the dataloader can consume.

Output layout: `<output>/<font>/<glyph_index>.png` (glyph_index is the
0..N-1 position of the codepoint in the codepoint list — by default the
62 ASCII characters 0-9 A-Z a-z, in that order).

Requires `cairosvg` for SVG rasterization:
    pip install cairosvg

Usage:
    python3 data_prep/svg_to_png.py \
        --svg_root  /path/to/svg \
        --out_root  /path/to/img \
        --image_size 128 \
        --workers 8
"""

import argparse
import io
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np
from PIL import Image


DEFAULT_CODEPOINTS: List[int] = (
    list(range(48, 58))
    + list(range(65, 91))
    + list(range(97, 123))
)


def _rasterize(svg_path: str, image_size: int) -> Image.Image:
    """Render a single SVG to a white-on-black PIL Image."""
    try:
        import cairosvg
    except ImportError as e:
        raise SystemExit('cairosvg is required: pip install cairosvg') from e
    with open(svg_path, 'rb') as f:
        png = cairosvg.svg2png(
            bytestring=f.read(),
            output_width=image_size,
            output_height=image_size,
        )
    rgba = np.array(Image.open(io.BytesIO(png)))
    alpha = rgba[..., -1]               # 0 = transparent background, 255 = ink
    rendered = 255 - alpha              # white background, black ink
    return Image.fromarray(rendered, mode='L')


def _process_one(args: Tuple[str, str, str, str, int, dict]) -> None:
    font_id, svg_name, svg_root, out_root, image_size, codepoint_to_index = args
    codepoint = int(os.path.splitext(svg_name)[0])
    glyph_idx = codepoint_to_index.get(codepoint)
    if glyph_idx is None:
        return
    out_dir = os.path.join(out_root, font_id)
    out_path = os.path.join(out_dir, f'{glyph_idx}.png')
    if os.path.exists(out_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    img = _rasterize(os.path.join(svg_root, font_id, svg_name), image_size)
    img.save(out_path)


def _iter_svgs(svg_root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for font_id in sorted(os.listdir(svg_root)):
        font_dir = os.path.join(svg_root, font_id)
        if not os.path.isdir(font_dir):
            continue
        for name in sorted(os.listdir(font_dir)):
            if name.lower().endswith('.svg'):
                pairs.append((font_id, name))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--svg_root', type=str, required=True)
    p.add_argument('--out_root', type=str, required=True)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    codepoint_to_index = {cp: i for i, cp in enumerate(DEFAULT_CODEPOINTS)}
    jobs = [(fid, name, args.svg_root, args.out_root, args.image_size,
             codepoint_to_index)
            for fid, name in _iter_svgs(args.svg_root)]
    print(f'{len(jobs)} glyphs to rasterize with {args.workers} workers')

    if args.workers <= 1:
        for job in jobs:
            _process_one(job)
    else:
        with mp.Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_process_one, jobs), 1):
                if i % 50 == 0:
                    print(f'  {i}/{len(jobs)}')


if __name__ == '__main__':
    main()
