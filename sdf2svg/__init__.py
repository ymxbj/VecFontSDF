"""Convert VecFontSDF parabolic-curve parameters into a vector SVG glyph.

The model predicts, per glyph, `v_dim * p_dim` parabolic curves with parameters
(k, p, q, d, e, f) defining H = k(px + qy)^2 + dx + ey + f. Each group of p_dim
curves is intersected (max) into one convex primitive; the v_dim primitives are
unioned (min) into the glyph. `params_to_svg` reconstructs that boolean shape as
quadratic-Bezier contours and writes an SVG.

Example:
    from sdf2svg import params_to_svg
    params_to_svg(params, v_dim=16, p_dim=6, out_path='glyph.svg')
"""
import os

import numpy as np

from . import bspt
from .outliner import cleanmesh, cleanmesh_connect

__all__ = ['params_to_svg']


def params_to_svg(params, v_dim, p_dim, out_path, merge=0.02, fill=True):
    """Write the glyph described by `params` to `out_path` as an SVG.

    Args:
        params:  array-like of shape (v_dim*p_dim, 6), the (k,p,q,d,e,f) curves
                 for one glyph (e.g. the output of inference's --save_params).
        v_dim:   number of shape primitives.
        p_dim:   number of parabolic curves intersected per primitive.
        out_path: destination .svg path.
        merge:   endpoint-merge threshold used when stitching curve segments into
                 closed contours (the research code's `--merge`, default 0.02).
        fill:    True  -> connected, filled contours (vector-font deliverable);
                 False -> individual stroked boundary segments (outline only).

    Returns:
        out_path.
    """
    params = np.asarray(params).reshape(1, v_dim, p_dim, 6)

    oldmeshes = []
    for v in range(v_dim):
        hyperplanes = [params[0, v, p, :] for p in range(p_dim)]
        meshes = bspt.digest_bsp_curve(hyperplanes)
        if meshes is None or len(meshes) == 0:
            continue
        oldmeshes, _ = bspt.combine_meshes(oldmeshes, meshes)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    if fill:
        cleanmesh_connect(oldmeshes, params, outfilename=out_path, merge=merge)
    else:
        cleanmesh(oldmeshes, params, outfilename=out_path)
    return out_path
