"""Downgrade cubic-Bezier SVG paths to quadratic-Bezier paths.

The downstream SDF / dataloader pipeline only understands the SVG path
commands `M`, `L`, and `Q`. This script walks every `<path d="...">` in
every input SVG and rewrites `C` (cubic Bezier) segments as `Q` or `L`
whenever the cubic can be represented exactly (within tolerance) by a
single quadratic curve or a straight line.

Conversion rule. A cubic Bezier defined by control points (p0, p1, p2, p3)
can be downgraded to a quadratic Bezier iff its left tangent at p0 and
right tangent at p3 meet at the same point — that meeting point becomes
the off-curve control of the quadratic. Specifically:

    left_meet  = 0.5 * (3 * p1 - p0)
    right_meet = 0.5 * (3 * p2 - p3)

If those two points coincide within `--curve_tol`, the cubic becomes
`Q left_meet p3`. If all four control points are collinear within
`--line_tol`, the cubic becomes `L p3`. Otherwise the conversion fails
for that glyph and the script reports it.

Usage:
    python3 data_prep/cubic_to_quadratic.py \
        --svg_root /path/to/cubic_svg \
        --out_root /path/to/quadratic_svg \
        --workers 8
"""

import argparse
import multiprocessing as mp
import os
from typing import List, Tuple


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _convert_path(d: str, line_tol: float, curve_tol: float
                  ) -> Tuple[bool, str]:
    """Convert the body of a single `d` attribute. Returns (ok, rewritten)."""
    tokens = [t for t in d.strip().split(' ') if t != '']
    out: List[str] = []
    x0 = y0 = 0.0
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd == 'M':
            x0 = float(tokens[i + 1])
            y0 = float(tokens[i + 2])
            out += ['M', repr(x0), repr(y0)]
            i += 3
        elif cmd == 'L':
            x1 = float(tokens[i + 1])
            y1 = float(tokens[i + 2])
            out += ['L', repr(x1), repr(y1)]
            x0, y0 = x1, y1
            i += 3
        elif cmd == 'Q':
            x1 = float(tokens[i + 1])
            y1 = float(tokens[i + 2])
            x2 = float(tokens[i + 3])
            y2 = float(tokens[i + 4])
            out += ['Q', repr(x1), repr(y1), repr(x2), repr(y2)]
            x0, y0 = x2, y2
            i += 5
        elif cmd == 'C':
            x1 = float(tokens[i + 1]); y1 = float(tokens[i + 2])
            x2 = float(tokens[i + 3]); y2 = float(tokens[i + 4])
            x3 = float(tokens[i + 5]); y3 = float(tokens[i + 6])

            colinear = (
                abs(_cross(x1 - x0, y1 - y0, x2 - x0, y2 - y0)) < line_tol
                and abs(_cross(x3 - x0, y3 - y0, x2 - x0, y2 - y0)) < line_tol
            )
            xml = 0.5 * (3 * x1 - x0)
            yml = 0.5 * (3 * y1 - y0)
            xmr = 0.5 * (3 * x2 - x3)
            ymr = 0.5 * (3 * y2 - y3)

            if colinear:
                out += ['L', repr(x3), repr(y3)]
            elif abs(xml - xmr) < curve_tol and abs(yml - ymr) < curve_tol:
                xm = 0.5 * (xml + xmr)
                ym = 0.5 * (yml + ymr)
                out += ['Q', repr(xm), repr(ym), repr(x3), repr(y3)]
            else:
                return False, ''
            x0, y0 = x3, y3
            i += 7
        elif cmd in ('Z', 'z'):
            out.append(cmd)
            i += 1
        else:
            return False, ''
    return True, ' '.join(out)


def convert_svg(svg_text: str, line_tol: float = 1e-2,
                curve_tol: float = 1.0) -> Tuple[bool, str]:
    """Rewrite every `<path d="...">` in the given SVG text. Returns
    (ok, rewritten_svg). If any path fails, ok is False and the second
    value is empty."""
    parts = svg_text.split('<path d="')
    pieces = [parts[0]]
    for chunk in parts[1:]:
        inner, rest = chunk.split('"', 1)
        ok, new_d = _convert_path(inner, line_tol, curve_tol)
        if not ok:
            return False, ''
        pieces.append('<path d="' + new_d + '"' + rest)
    return True, ''.join(pieces)


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


def _process_one(args: Tuple[str, str, str, str, float, float]
                 ) -> Tuple[str, str, bool]:
    font_id, svg_name, svg_root, out_root, line_tol, curve_tol = args
    out_dir = os.path.join(out_root, font_id)
    out_path = os.path.join(out_dir, svg_name)
    if os.path.exists(out_path):
        return font_id, svg_name, True
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(svg_root, font_id, svg_name)) as f:
        svg_text = f.read()
    ok, rewritten = convert_svg(svg_text, line_tol, curve_tol)
    if not ok:
        return font_id, svg_name, False
    with open(out_path, 'w') as f:
        f.write(rewritten)
    return font_id, svg_name, True


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--svg_root', type=str, required=True)
    p.add_argument('--out_root', type=str, required=True)
    p.add_argument('--line_tol', type=float, default=1e-2,
                   help='collinearity tolerance for collapsing a cubic to a line')
    p.add_argument('--curve_tol', type=float, default=1.0,
                   help='tangent-meet tolerance for collapsing a cubic to a quadratic')
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    jobs = [(fid, name, args.svg_root, args.out_root,
             args.line_tol, args.curve_tol)
            for fid, name in _iter_svgs(args.svg_root)]
    print(f'{len(jobs)} SVGs to convert with {args.workers} workers')

    failures: List[Tuple[str, str]] = []
    if args.workers <= 1:
        for job in jobs:
            fid, name, ok = _process_one(job)
            if not ok:
                failures.append((fid, name))
    else:
        with mp.Pool(args.workers) as pool:
            for i, (fid, name, ok) in enumerate(
                    pool.imap_unordered(_process_one, jobs), 1):
                if not ok:
                    failures.append((fid, name))
                if i % 200 == 0:
                    print(f'  {i}/{len(jobs)}  ({len(failures)} failures so far)')

    if failures:
        print(f'\n{len(failures)} glyphs could not be downgraded to quadratic:')
        for fid, name in failures[:20]:
            print(f'  {fid}/{name}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')
    else:
        print('all SVGs converted successfully')


if __name__ == '__main__':
    main()
