"""Sample points near the contour of every glyph and record their SDF.

For each font directory containing `*.svg`, write
`<output>/<font>/sdf/<codepoint>_contour.npy` — an `[N, 3]` float32 array
where each row is `(x, y, signed_distance)` in pixel units. Points are
sampled uniformly by arc length along all contours and then jittered with
uniform noise in [-1, 1] pixels.

Usage:
    python -m VecFontSDF.data_prep.svg_to_contour_sdf \
        --svg_root  /path/to/svg \
        --out_root  /path/to/sdf \
        --num_points 4000 \
        --workers 8
"""

import argparse
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np

from .geometry import Pos
from .glyph import Glyph


def sample_contour_sdf(glyph: Glyph, num_points: int,
                       jitter: float = 1.0) -> np.ndarray:
    """Return [num_points, 3] = (x, y, signed_distance) sampled near contours."""
    lens = [c.cumulative_lengths()[-1] for c in glyph.contours]
    total = sum(lens)
    if total <= 0.0:
        return np.zeros((num_points, 3), dtype=np.float32)

    per_contour = []
    assigned = 0
    for length in lens[:-1]:
        n = int(length / total * num_points)
        per_contour.append(n)
        assigned += n
    per_contour.append(num_points - assigned)

    samples: List[Pos] = []
    for contour, n in zip(glyph.contours, per_contour):
        samples.extend(contour.sample(n))

    noise = (np.random.rand(num_points, 2) * 2.0 - 1.0) * jitter
    out = np.zeros((num_points, 3), dtype=np.float32)
    for i, pt in enumerate(samples):
        out[i, 0] = pt.x + noise[i, 0]
        out[i, 1] = pt.y + noise[i, 1]
        out[i, 2] = glyph.signed_distance(Pos(float(out[i, 0]), float(out[i, 1])))
    return out


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


def _process_one(args: Tuple[str, str, str, str, int, float]) -> None:
    font_id, svg_name, svg_root, out_root, num_points, jitter = args
    out_dir = os.path.join(out_root, font_id, 'sdf')
    codepoint = int(os.path.splitext(svg_name)[0])
    out_path = os.path.join(out_dir, f'{codepoint}_contour.npy')
    if os.path.exists(out_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    glyph = Glyph.from_svg_file(os.path.join(svg_root, font_id, svg_name))
    arr = sample_contour_sdf(glyph, num_points, jitter)
    np.save(out_path, arr)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--svg_root', type=str, required=True)
    p.add_argument('--out_root', type=str, required=True)
    p.add_argument('--num_points', type=int, default=4000)
    p.add_argument('--jitter', type=float, default=1.0,
                   help='uniform noise radius (pixel units) added to each sample')
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    jobs = [(fid, name, args.svg_root, args.out_root,
             args.num_points, args.jitter)
            for fid, name in _iter_svgs(args.svg_root)]
    print(f'{len(jobs)} glyphs to process with {args.workers} workers')

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
