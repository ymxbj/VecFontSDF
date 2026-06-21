"""Turn the combined parabolic-primitive meshes into clean SVG outlines.

Ported verbatim from the original VecFontSDF research code (conic16_6/
outliner_multi_v2.py); the only changes are decoupling from the training
`model`/`options` modules (v_dim/p_dim/merge are passed in) and the numpy>=2
`np.bool` -> `bool` rename. The geometry logic is unchanged.

The output SVG carries `<g transform="matrix(0 1 1 0 0 0)">`: the curves live in
the model's (x=row, y=col) parameter space, and this matrix swaps x<->y back to
screen orientation so the glyph renders upright.
"""
import numpy as np

sqr = lambda x: x * x


def calc_h1(pt, y_p_raw):
    # y_p_raw: batchsize, v_dim, p_dim, 6
    x = np.array(pt.x).reshape(1, 1, 1)
    y = np.array(pt.y).reshape(1, 1, 1)
    k, p, q, d, e, f = np.split(y_p_raw, 6, axis=3)
    h1 = k * sqr(p * x + q * y) + d * x + e * y + f
    h1 = np.max(h1, axis=2)
    h1 = np.min(h1, axis=1)
    return h1[0]


def cleanmesh(oldmeshes, y_p_raw, outfilename=None):
    """Keep only the curve segments that lie on the glyph boundary, written as
    individual stroked outline paths (fill=None)."""
    results = []
    for primitive in oldmeshes:
        for curve in primitive:
            midpoint = curve.getpos(0.5)
            if calc_h1(midpoint, y_p_raw) > -1e-5:
                results.append(curve)
    if outfilename is not None:
        f = open(outfilename, 'w')
        f.write('<svg width="200px" height="200px" viewBox="-1,-1,2,2" version="1.1" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<g transform="matrix(0 1 1 0 0 0)">\n')
        for l in results:
            f.write('<path d="M %f %f Q %f %f %f %f " stroke="black" stroke-width="0.01" fill="None"/>\n' % (l[0].x, l[0].y, l[1].x, l[1].y, l[2].x, l[2].y))
        f.write('</g>\n</svg>\n')
        f.close()
    return results


def cleanmesh_connect(oldmeshes, y_p_raw, outfilename=None, merge=0.02):
    """Connect the boundary curve segments into closed quadratic-Bezier contours
    and write them as filled paths (the vector-font deliverable)."""
    merge_MINE = merge
    raw_results = []
    pts = []  # for each point, (x,y),[(nump,numc,0/2),...]
    SDF_MINE = 1e-3
    for nump, primitive in enumerate(oldmeshes):
        for numc, curve in enumerate(primitive):
            midpoint = curve.getpos(0.5)
            if -SDF_MINE < calc_h1(midpoint, y_p_raw) < SDF_MINE:
                leftpoint = curve.positions[0]
                rightpoint = curve.positions[-1]
                pts.append([leftpoint, [(nump, numc, len(raw_results), 0)]])
                pts.append([rightpoint, [(nump, numc, len(raw_results), 2)]])
                raw_results.append([curve, len(pts) - 2, len(pts) - 1])
    lraw = len(raw_results)
    # mergepoints
    pts.sort(key=lambda p: p[0].x)
    lpts = len(pts)
    proc_pts = np.zeros((lpts,), dtype=bool)
    for i in range(lpts):
        if proc_pts[i]:
            continue
        proc_pts[i] = True
        pi = pts[i][0]
        raw_results_index = pts[i][1][0][2]
        if pts[i][1][0][3] == 0:
            raw_results[raw_results_index][1] = i
        else:
            # pts[i][1][0][3]==2
            raw_results[raw_results_index][2] = i
        for j in range(i + 1, lpts):
            pj = pts[j][0]
            if abs(pj - pi) < merge_MINE:
                proc_pts[j] = True
                raw_results_index = pts[j][1][0][2]
                if pts[j][1][0][3] == 0:
                    raw_results[raw_results_index][1] = i
                else:
                    # pts[i][1][0][3]==2
                    raw_results[raw_results_index][2] = i
    # update raw_results and remove 0-length segments
    proc_results = []
    for i in range(lraw):
        c, np0, np2 = raw_results[i]
        c.positions[0] = pts[np0][0]
        c.positions[2] = pts[np2][0]
        if abs(c.positions[2] - c.positions[0]) < merge_MINE:
            continue
        proc_results.append(c)
    lproc = len(proc_results)
    results = []
    used_results = np.zeros((lproc,), dtype=bool)
    for i in range(lproc):
        j = i
        if used_results[i]:
            continue
        curr_results = []
        while not used_results[j]:
            c = proc_results[j]
            p0, p1, p2 = c.positions
            c.positions[0] = p0
            c.positions[2] = p2
            curr_results.append(c)
            used_results[j] = True
            find = -1
            for k in range(lproc):
                if not used_results[k]:
                    cc = proc_results[k]
                    if abs(cc.positions[0] - p2) < merge_MINE:
                        find = k
                        break
            if find > 0:
                j = k
            else:
                break
        results.append(curr_results.copy())
    if outfilename is not None:
        f = open(outfilename, 'w')
        f.write('<svg width="200px" height="200px" viewBox="-1,-1,2,2" version="1.1" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<g transform="matrix(0 1 1 0 0 0)">\n')
        for c in results:
            for i, l in enumerate(c):
                if i == 0:
                    f.write('<path d="M %f %f Q %f %f %f %f ' % (l[0].x, l[0].y, l[1].x, l[1].y, l[2].x, l[2].y))
                else:
                    f.write('Q %f %f %f %f ' % (l[1].x, l[1].y, l[2].x, l[2].y))
            f.write('" stroke-width="1.0" fill="Black" opacity="1.0"/>\n')
        f.write('</g>\n</svg>\n')
        f.close()
    return results
