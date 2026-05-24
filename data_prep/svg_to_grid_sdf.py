"""Compute a per-pixel signed distance field for every glyph SVG in a tree.

Output layout matches what the training dataloader expects: for each font
directory containing `*.svg`, write `<output>/<font>/sdf/<codepoint>_grid.npy`
(an `[image_size, image_size]` float32 array stored as (col, row); the
dataloader will transpose it back to (row, col) at load time).

Usage:
    python3 data_prep/svg_to_grid_sdf.py \
        --svg_root  /path/to/svg \
        --out_root  /path/to/sdf \
        --image_size 128 \
        --workers 8
"""

import argparse
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np

from geometry import Pos
from glyph import Glyph


def compute_grid_sdf(glyph: Glyph, image_size: int) -> np.ndarray:
    """Return an [image_size, image_size] SDF in (col, row) order.

    Each pixel center (i + 0.5, j + 0.5) is sampled with x = i (column) and
    y = j (row); the returned array has axis 0 as x and axis 1 as y. The
    training-time dataloader transposes this to standard (row, col).
    """
    out = np.zeros((image_size, image_size), dtype=np.float32)
    for i in range(image_size):
        for j in range(image_size):
            out[i, j] = glyph.signed_distance(Pos(i + 0.5, j + 0.5))
    return out


def _iter_svgs(svg_root: str) -> List[Tuple[str, str]]:
    """Yield (font_id, svg_filename) for every .svg file under svg_root."""
    pairs: List[Tuple[str, str]] = []
    for font_id in sorted(os.listdir(svg_root)):
        font_dir = os.path.join(svg_root, font_id)
        if not os.path.isdir(font_dir):
            continue
        for name in sorted(os.listdir(font_dir)):
            if name.lower().endswith('.svg'):
                pairs.append((font_id, name))
    return pairs


def _codepoint_from_filename(name: str) -> int:
    """SVG files in the source dataset are named by the leading codepoint."""
    stem = os.path.splitext(name)[0]
    return int(stem)


def _process_one(args: Tuple[str, str, str, str, int]) -> None:
    font_id, svg_name, svg_root, out_root, image_size = args
    out_dir = os.path.join(out_root, font_id, 'sdf')
    codepoint = _codepoint_from_filename(svg_name)
    out_path = os.path.join(out_dir, f'{codepoint}_grid.npy')
    if os.path.exists(out_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    glyph = Glyph.from_svg_file(os.path.join(svg_root, font_id, svg_name))
    sdf = compute_grid_sdf(glyph, image_size)
    np.save(out_path, sdf)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--svg_root', type=str, required=True)
    p.add_argument('--out_root', type=str, required=True)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    jobs = [(fid, name, args.svg_root, args.out_root, args.image_size)
            for fid, name in _iter_svgs(args.svg_root)]
    print(f'{len(jobs)} glyphs to process with {args.workers} workers')

    if args.workers <= 1:
        for job in jobs:
            _process_one(job)
            print('done', job[0], job[1])
    else:
        with mp.Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_process_one, jobs), 1):
                if i % 50 == 0:
                    print(f'  {i}/{len(jobs)}')


if __name__ == '__main__':
    main()
