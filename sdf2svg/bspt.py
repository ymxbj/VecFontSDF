import os
from . import lines

from .pos import *
import math
import numpy as np

MINE = 1e-7

# sqr=lambda x:x*x
cubic = lambda x: x * x * x
sqrt = math.sqrt


def print_to_strs(meshes, status=1):
    results = []
    results.append(
        '<svg width="200px" height="200px" viewBox="-1,-1,2,2" version="1.1" xmlns="http://www.w3.org/2000/svg">' + '\n')
    results.append('<path d="')
    for i in range(len(meshes)):
        m = meshes[i]
        if isinstance(meshes[i], list):
            m = lines.quadratic_bezier_curve(pos(m[0]), pos(m[1]), pos(m[2]))
        results.append(m.print_to_svg_path_inner(isfirst=(i == 0)) + '\n')
    results.append('\n')
    if status > 0:
        results.append('" stroke="black" stroke-width="0.01"/>')
    else:
        results.append('" stroke="white" stroke-width="0.01" fill="white"/>')
    for i in range(len(meshes)):
        m = meshes[i]
        if isinstance(meshes[i], list):
            m = lines.quadratic_bezier_curve(pos(m[0]), pos(m[1]), pos(m[2]))
        results.append(m.print_to_svg_circle() + '\n')
    results.append('</svg>')
    return results


def print_to_svg_file(meshes, filename=None,crss=None):
    if filename is None:
        for mesh in meshes:
            for i in range(len(mesh)):
                print(mesh[i].print_to_svg_path_inner(isfirst=(i == 0)))
            for i in mesh:
                print(i.print_to_svg_circle())
    else:
        f = open(filename, 'w')
        f.write(
            '<svg width="200px" height="200px" viewBox="-1,-1,2,2" version="1.1" xmlns="http://www.w3.org/2000/svg">' + '\n')
        for mesh in meshes:
            f.write('<path d="')
            for i in range(len(mesh)):
                m = mesh[i]
                if isinstance(mesh[i], list):
                    m = lines.quadratic_bezier_curve(pos(m[0]), pos(m[1]), pos(m[2]))

                f.write(m.print_to_svg_path_inner(isfirst=(i == 0)) + '\n')
            f.write('\n')
            f.write('" stroke="black" stroke-width="0.01" fill="None"/>')
            for i in range(len(mesh)):
                m = mesh[i]
                if isinstance(mesh[i], list):
                    m = lines.quadratic_bezier_curve(pos(m[0]), pos(m[1]), pos(m[2]))
                f.write(m.print_to_svg_circle() + '\n')
            if crss is not None:
                for cpt in crss:
                    cptpr='<circle cx="' + str(cpt.x) + '" cy="' + str(cpt.y) + '" r="' + str(
            0.01) + '" stroke="Gold" stroke-width="'+str(0.01)+'"/>'
                    f.write(cptpr + '\n')
        f.write('</svg>')
        f.close()


def digest_bsp_curve(boxes, debug_folder=None):
    def calc_h1(pts):
        x = pts[1][0]
        y = pts[1][1]
        y3 = pts[2][1]
        x3 = pts[2][0]
        y1 = pts[0][1]
        x1 = pts[0][0]
        h1 = -((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1))
        return h1

    # init_box = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

    # init_meshes = [[(-1, -1), (-1, 0), (-1, 1)], [(-1, 1), (0, 1), (1, 1)], [(1, 1), (1, 0), (1, -1)],
    #               [(1, -1), (0, -1), (-1, -1)]]
    # init_meshes = [[[-1, 1], [-1, 0], [-1, -1]], [[1, 1], [0, 1], [-1, 1]], [[1, -1], [1, 0], [1, 1]],
    #               [[-1, -1], [0, -1], [1, -1]]]
    #init_meshes = [[[-1, -1], [0, -1], [1, -1]], [[1, -1], [1, 0], [1, 1]], [[1, 1], [0, 1], [-1, 1]],
    #               [[-1, 1], [-1, 0], [-1, -1]]]
    # 2022.11 edit: maybe more than one mesh
    init_meshes = [[[[-1, -1], [0, -1], [1, -1]], [[1, -1], [1, 0], [1, 1]], [[1, 1], [0, 1], [-1, 1]],
                   [[-1, 1], [-1, 0], [-1, -1]]]]


    # box = init_box
    for i in range(4):
        init_meshes[0][i]=lines.quadratic_bezier_curve(pos(init_meshes[0][i][0]), pos(init_meshes[0][i][1]), pos(init_meshes[0][i][2]))
    meshes = init_meshes
    newboxes = boxes
    '''
    # extend boxes by its type
    lb = len(boxes)
    for i in range(lb):
        h1 = calc_h1(boxes[i])
        if abs(h1) < MINE:
            newboxes.append(boxes[i])
            # print(boxes[i], h1)
        elif h1 > 0:
            # add two straight line and a curve
            p0 = pos(boxes[i][0])
            p1 = pos(boxes[i][1])
            p2 = pos(boxes[i][2])
            newboxes.append([p0, 0.5 * (p0 + p1), p1])
            newboxes.append(boxes[i])
            # newboxes.append([p2, p1, p0])
            newboxes.append([p1, 0.5 * (p1 + p2), p2])
            # print(boxes[i], h1, newboxes[-3:])
        else:
            # add two straight line only
            p0 = pos(boxes[i][0])
            p1 = pos(boxes[i][1])
            p2 = pos(boxes[i][2])
            newboxes.append([p0, 0.5 * (p0 + p1), p1])
            newboxes.append([p1, 0.5 * (p1 + p2), p2])
            # print(boxes[i], h1, newboxes[-2:])
    '''

    l = len(newboxes)
    for i in range(l):
        # box, meshes = join_meshes(boxes[i], meshes)
        newmeshes = join_meshes(newboxes[i], meshes)
        if newmeshes is None:
            return []
        if len(newmeshes) == 0:
            return []
        meshes = newmeshes
        # debug use
        if debug_folder is not None:
            if not os.path.exists(debug_folder):
                os.mkdir(debug_folder)
            print_to_svg_file(meshes, filename=os.path.join(debug_folder, str(i) + '.svg'))

    # DEPRECATED: generate box from meshes
    # return box, meshes
    return meshes


def linear_equ(k, b):
    # kx+b=0
    if abs(k) < MINE:
        return None
    return -b / k


def quadratic_equ(a, b, c):
    # ax^2+bx+c=0
    if abs(a) < MINE:
        return linear_equ(b, c)
    b /= a
    c /= a
    a = 1.0
    delta = sqr(b) - 4 * a * c
    if delta < -MINE:
        return None
    elif delta < MINE:
        return -b / 2 / a
    else:
        return (-b - sqrt(delta)) / 2 / a, (-b + sqrt(delta)) / 2 / a


def cubic_equ(a, b, c, d):
    if abs(a) < MINE:
        return quadratic_equ(b, c, d)
    p = b * c / (6 * sqr(a)) - cubic(b) / cubic(3 * a) - d / (2 * a)
    q = -sqr(b) / sqr(3 * a) + c / (3 * a)
    bdt = sqr(p) + cubic(q)
    ans = 0
    if (bdt >= 0):
        dt = math.sqrt(bdt)
        ans = np.cbrt(dt + p) + np.cbrt(-dt + p) - b / (3 * a)
        # return ans
    else:
        r = math.sqrt(sqr(p) - bdt)
        theta = 0
        t = 0
        if (p >= 0):
            theta = math.acos(p / r)
            t = np.cbrt(r) * math.cos(theta / 3) * 2
        else:
            theta = math.acos(-p / r)
            t = -np.cbrt(r) * math.cos(theta / 3) * 2
        ans = t - b / (3 * a)
        # return ans
    # (ax^3+bx^2+cx+d)/(x-ans)
    aa = a
    bb = b + a * ans
    cc = c + b * ans + a * sqr(ans)
    res = quadratic_equ(aa, bb, cc)
    if res is None:
        return ans
    elif isinstance(res, tuple):
        return ans, res[0], res[1]
    else:
        return ans, res


def quartic_equ(a, b, c, d, e):
    if abs(a) < MINE:
        return cubic_equ(b, c, d, e)
    quartic = lambda x: x * x * x * x

    alpha = -3 * sqr(b) / (8 * sqr(a)) + c / a
    beta = cubic(b) / cubic(2 * a) - b * c / (2 * sqr(a)) + d / a
    gamma = -3 * quartic(b) / (256 * quartic(a)) + sqr(b) * c / (16 * cubic(a)) - b * d / sqr(2 * a) + e / a
    # x^4+alpha*x^2+beta*x+gamma=0

    if abs(beta) < MINE:
        #fix: move results
        #x=u-b/(4a)
        # solve u^4+alpha u^2+gamma=0
        delta=sqr(alpha) - 4 * gamma
        u=None
        if delta < -MINE:
            return None
        elif delta < MINE:
            u2=-alpha/2
            if u2 < -MINE:
                return None
            elif u2 < MINE:
                u=0
                x=u-b/(4*a)
                return x
            else:
                u=sqrt(u2)
                # has two roots+-, showing one only.
                return u-b/(4*a),-u-b/(4*a)
        else:
            u2_1 = (-alpha - sqrt(delta)) / 2
            u2_2 = (-alpha + sqrt(delta)) / 2
            #us=[sqrt(u2_1),-sqrt(u2_1),sqrt(u2_2),-sqrt(u2_2)]
            us=[]
            if u2_1>MINE:
                us.append(sqrt(u2_1))
                us.append(-sqrt(u2_1))
            elif u2_1>-MINE:
                us.append(0.0)
            if u2_2>MINE:
                us.append(sqrt(u2_2))
                us.append(-sqrt(u2_2))
            elif u2_2>-MINE:
                us.append(0.0)
            xs=[i-b/(4*a) for i in us]
            return xs

    y = cubic_equ(a=1, b=2.5 * alpha, c=2 * sqr(alpha) - gamma, d=cubic(alpha) / 2 - alpha * gamma / 2 - sqr(beta) / 8)
    if isinstance(y, tuple):
        y = y[0]
    delta1 = alpha + 2 * y
    if delta1 < -MINE:
        ip_d1 = sqrt(-delta1)
        rp_d1_p = -(3 * alpha + 2 * y)
        ip_p = sqrt(2) / 2 * np.sign(2 * beta / sqrt(-delta1)) * sqrt(
            sqrt(sqr(3 * alpha + 2 * y) - sqr(2 * beta) / delta1) + (3 * alpha + 2 * y))
        ip_f = sqrt(-delta1)
        if abs(ip_p + ip_f) < MINE:
            # calc real part when zhengfut>0
            rp_p = sqrt(2) / 2 * sqrt(sqrt(sqr(3 * alpha + 2 * y) - sqr(2 * beta) / delta1) - (3 * alpha + 2 * y))
            return -b / (4 * a) + rp_p / 2, -b / (4 * a) - rp_p / 2
        elif abs(ip_p - ip_f) < MINE:
            # calc real part when zhengfut<0
            rp_p = sqrt(2) / 2 * sqrt(sqrt(sqr(3 * alpha + 2 * y) - sqr(2 * beta) / delta1) - (3 * alpha + 2 * y))
            return -b / (4 * a) + rp_p / 2, -b / (4 * a) - rp_p / 2
        else:
            return None
    elif delta1 < MINE:
        # similars to beta=0
        # discussed
        ## os._exit(200)
        ## however, if go into this branch, then a is too small to calculate
        ## so we use bx^3+cx^2+dx+e instead.
        #return cubic_equ(b, c, d, e)

        # the above was wrong.
        # It should be x:=u-b/4a
        # then process u^4+pu^2+q=0
        #fix: move results
        #x=u-b/(4a)
        # solve u^4+alpha u^2+gamma=0
        delta=sqr(alpha) - 4 * gamma
        u=None
        if delta < -MINE:
            return None
        elif delta < MINE:
            u2=-alpha/2
            if u2 < -MINE:
                return None
            elif u2 < MINE:
                u=0
                x=u-b/(4*a)
                return x
            else:
                u=sqrt(u2)
                # has two roots+-, showing one only.
                return u-b/(4*a),-u-b/(4*a)
        else:
            u2_1 = (-alpha - sqrt(delta)) / 2
            u2_2 = (-alpha + sqrt(delta)) / 2
            #us=[sqrt(u2_1),-sqrt(u2_1),sqrt(u2_2),-sqrt(u2_2)]
            us=[]
            if u2_1>MINE:
                us.append(sqrt(u2_1))
                us.append(-sqrt(u2_1))
            elif u2_1>-MINE:
                us.append(0.0)
            if u2_2>MINE:
                us.append(sqrt(u2_2))
                us.append(-sqrt(u2_2))
            elif u2_2>-MINE:
                us.append(0.0)
            xs=[i-b/(4*a) for i in us]
            return xs

    else:
        # delta>0
        # 4 solutions
        delta2_1 = -(3 * alpha + 2 * y + 2 * beta / sqrt(delta1))
        delta2_2 = -(3 * alpha + 2 * y - 2 * beta / sqrt(delta1))
        solutions = []
        if delta2_1 < -MINE:
            pass
        elif delta2_1 < MINE:
            # add 0 solution
            solutions.append(-b / (4 * a) + sqrt(delta1) / 2)
        else:
            # delta2_1>MINE
            solutions.append(-b / (4 * a) + (sqrt(delta1) + sqrt(delta2_1)) / 2)
            solutions.append(-b / (4 * a) + (sqrt(delta1) - sqrt(delta2_1)) / 2)
        if delta2_2 < -MINE:
            pass
        elif delta2_2 < MINE:
            # add 0 solution
            solutions.append(-b / (4 * a) + -sqrt(delta1) / 2)
        else:
            # delta2_2>MINE
            solutions.append(-b / (4 * a) + (-sqrt(delta1) + sqrt(delta2_2)) / 2)
            solutions.append(-b / (4 * a) + (-sqrt(delta1) - sqrt(delta2_2)) / 2)
        if len(solutions) == 0:
            return None
        elif len(solutions) == 1:
            return solutions[0]
        else:
            return tuple(solutions)


def calc_two_plane_crosses_safe(p1, p2):
    def calca123(pc):
        a3c = pc[2] + pc[0] - 2 * pc[1]
        a2c = -2 * pc[0] + 2 * pc[1]
        a1c = pc[0]
        return a3c, a2c, a1c

    pc = [pos(i) for i in p1]
    dlt2c = pc[2] - pc[0]
    dlt1c = pc[1] - pc[0]
    dltc = dlt1c.x * dlt2c.y - dlt1c.y * dlt2c.x

    pd = [pos(i) for i in p2]
    dlt2d = pd[2] - pd[0]
    dlt1d = pd[1] - pd[0]
    dltd = dlt1d.x * dlt2d.y - dlt1d.y * dlt2d.x

    ac = calca123(pc)
    bc = calca123(pd)

    a3, a2, a1 = [i.x for i in ac]
    b3, b2, b1 = [i.x for i in bc]
    c3, c2, c1 = [i.y for i in ac]
    d3, d2, d1 = [i.y for i in bc]

    fm = a2 * c3 - a3 * c2
    nq = 0
    if abs(fm) < MINE:
        A = (b3 * c3 - a3 * d3)
        if abs(A) < MINE:
            if abs(a3) < MINE:
                if abs(a2) < MINE:
                    t2 = quadratic_equ(b3, b2, b1 - a1)
                    nq = -2
                else:
                    A = ((sqr(b3) * c3) / sqr(a2))
                    B = ((2 * b2 * b3 * c3) / sqr(a2))
                    C = ((b3 * c2) / a2 - (c3 * (- sqr(b2) + 2 * b3 * (a1 - b1))) / sqr(a2) - d3)
                    D = ((b2 * c2) / a2 - d2 - (2 * b2 * c3 * (a1 - b1)) / sqr(a2))
                    E = c1 - d1 - (c2 * (a1 - b1)) / a2 + (c3 * sqr(a1 - b1)) / sqr(a2)
                    t2 = quartic_equ(A, B, C, D, E)
                    nq = 4
            elif abs(c3) < MINE:
                if abs(c2) < MINE:
                    t2 = quadratic_equ(d3, d2, d1 - c1)
                    nq = -22
                else:
                    A = ((a3 * sqr(d3)) / sqr(c2))
                    B = ((2 * a3 * d2 * d3) / sqr(c2))
                    C = ((a2 * d3) / c2 - (a3 * (- sqr(d2) + 2 * d3 * (c1 - d1))) / sqr(c2) - b3)
                    D = ((a2 * d2) / c2 - b2 - (2 * a3 * d2 * (c1 - d1)) / sqr(c2))
                    E = a1 - b1 - (a2 * (c1 - d1)) / c2 + (a3 * sqr(c1 - d1)) / sqr(c2)
                    t2 = quartic_equ(A, B, C, D, E)
                    nq = 4
            else:
                B = (b2 * c3 - a3 * d2)
                if abs(B) < MINE:
                    t2 = None
                else:
                    C = -a3 * d1 + b1 * c3 - (a1 * c3 - a3 * c1)
                    t2 = linear_equ(B, C)
                    nq = 1
        else:
            B = (b2 * c3 - a3 * d2)
            C = -a3 * d1 + b1 * c3 - (a1 * c3 - a3 * c1)
            t2 = quadratic_equ(A, B, C)
            nq = 2
    else:
        A = ((a3 * sqr(a3 * d3 - b3 * c3)) / sqr(a2 * c3 - a3 * c2))
        B = ((2 * a3 * (a3 * d2 - b2 * c3) * (a3 * d3 - b3 * c3)) / sqr(a2 * c3 - a3 * c2))
        C = ((a3 * (sqr(a3 * d2 - b2 * c3) + 2 * (a3 * d3 - b3 * c3) * (a1 * c3 - a3 * c1 + a3 * d1 - b1 * c3))) / sqr(
            a2 * c3 - a3 * c2) - b3 - (a2 * (a3 * d3 - b3 * c3)) / (a2 * c3 - a3 * c2))
        D = ((2 * a3 * (a3 * d2 - b2 * c3) * (a1 * c3 - a3 * c1 + a3 * d1 - b1 * c3)) / sqr(a2 * c3 - a3 * c2) - (
                a2 * (a3 * d2 - b2 * c3)) / (a2 * c3 - a3 * c2) - b2)
        E = a1 - b1 + (a3 * sqr(a1 * c3 - a3 * c1 + a3 * d1 - b1 * c3)) / sqr(a2 * c3 - a3 * c2) - (
                a2 * (a1 * c3 - a3 * c1 + a3 * d1 - b1 * c3)) / (a2 * c3 - a3 * c2)
        t2 = quartic_equ(A, B, C, D, E)
        nq = 4
    # then check if t2 is valid
    rawresults = []
    if t2 is None:
        return []
    if not isinstance(t2, tuple):
        t2 = (t2,)
    for t in t2:
        if nq == 4:
            if abs(A * t * t * t * t + B * t * t * t + C * t * t + D * t + E) > 0.01:
                continue
        t1 = quadratic_equ(a3, a2, a1 - (sqr(t) * b3 + t * b2 + b1))
        if t1 is None:
            t1 = quadratic_equ(c3, c2, c1 - (d3 * sqr(t) + d2 * t + d1))
        if t1 is None:
            return []
        if not isinstance(t1, tuple):
            t1 = (t1,)
        for tt1 in t1:
            rawresults.append((tt1, t))

    # final check
    result = []
    for t1, t2 in rawresults:
        lx = a3 * sqr(t1) + a2 * t1 + a1
        rx = b3 * sqr(t2) + b2 * t2 + b1
        ly = c3 * sqr(t1) + c2 * t1 + c1
        ry = d3 * sqr(t2) + d2 * t2 + d1
        if abs(lx - rx) < sqrt(MINE) and abs(ly - ry) < sqrt(MINE):
            result.append((t1, t2))

    return result


def calc_two_plane_crosses(p1, p2):
    pc = [pos(i) for i in p1]
    if abs(pc[2] - pc[0]) < 1e-6 and abs(pc[1] - pc[0]) < 1e-6:
        return None

    pc = [pos(i) for i in p2]
    if abs(pc[2] - pc[0]) < 1e-6 and abs(pc[1] - pc[0]) < 1e-6:
        return None
    return calc_two_plane_crosses_safe(p1, p2)


def calc_h1_pts(pcurve,exm):
    # pc: existing (p0,p1,p2)curve
    # pt: new point
    pcsegj = pcurve
    x = pcsegj.positions[1].x
    y = pcsegj.positions[1].y
    y3 = pcsegj.positions[2].y
    x3 = pcsegj.positions[2].x
    y1 = pcsegj.positions[0].y
    x1 = pcsegj.positions[0].x
    h1 = -((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1))
    if abs(h1) < MINE:
        linecrv = lines.straight_line(pcsegj.positions[0], pcsegj.positions[2])
    else:
        #t_min_dist = pcsegj.argmindist(exm, mint=-math.inf, maxt=math.inf)
        t_min_dist = pcsegj.argmindist(exm, mint=0, maxt=1)
        p_nearest = pcsegj.getpos(t_min_dist)
        v_mid_dist = pcsegj.speed(t_min_dist)
        linecrv = lines.straight_line(p_nearest, p_nearest + v_mid_dist)
    x = exm.x
    y = exm.y
    y3 = linecrv.positions[1].y
    x3 = linecrv.positions[1].x
    y1 = linecrv.positions[0].y
    x1 = linecrv.positions[0].x
    h1 = -((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1))
    return h1


def inside_curr(pt,curr_exists):
    for ln in curr_exists:
        cln = lines.quadratic_bezier_curve(pos(ln[0]), pos(ln[1]), pos(ln[2]))
        if calc_h1_pts(cln,pt)<0:
            return False
    return True

def join_meshes(curve_plane, exists):
    # curve_plane: array of [6]
    ## p=a=y[0]
    ## q=b=y[1]
    ## d=y[3],e=y[4],f=y[5]
    ## h1 = sqr(p * x + q * y) + d * x + e * y + f
    # new_version:
    # h1 = k * sqr(p * x + q * y) + d * x + e * y + f
    # k=a,p=b,q=c
    # exists: 3-point representation
    # n*3*2

    def expand_equation_1k(pc, exist_line):
        # exist_line:[p0,p1,p2]
        # p=(1-t)^2*P0+2t(1-t)*P1+t^2P2
        exc = [pos(i) for i in exist_line]
        x = [i.x for i in exc]
        y = [i.y for i in exc]
        # x=(1-t)^2*x0+2t(1-t)*x1+t^2x2
        # y=(1-t)^2*y0+2t(1-t)*y1+t^2y2
        # k*(p * x + q * y)^2+d * x + e * y + f=0
        k, p, q, d, e, f = pc
        x0, x1, x2 = x
        y0, y1, y2 = y
        aa=k*sqr(p*(x0 - 2*x1 + x2) + q*(y0 - 2*y1 + y2))
        bb=(-2*k*(p*(2*x0 - 2*x1) + q*(2*y0 - 2*y1))*(p*(x0 - 2*x1 + x2) + q*(y0 - 2*y1 + y2)))
        cc=(k*(sqr(p*(2*x0 - 2*x1) + q*(2*y0 - 2*y1)) + 2*(p*(x0 - 2*x1 + x2) + q*(y0 - 2*y1 + y2))*(p*x0 + q*y0)) + d*(x0 - 2*x1 + x2) + e*(y0 - 2*y1 + y2))
        dd=(- d*(2*x0 - 2*x1) - e*(2*y0 - 2*y1) - 2*k*(p*(2*x0 - 2*x1) + q*(2*y0 - 2*y1))*(p*x0 + q*y0))
        ee=f + k*sqr(p*x0 + q*y0) + d*x0 + e*y0
        return aa,bb,cc,dd,ee

    def cpt_1k(pc,pt):
        k, p, q, d, e, f = pc
        x = pt.x
        y = pt.y
        h1 = k * sqr(p * x + q * y) + d * x + e * y + f
        return h1

    def pre1s_k(pt,pc):
        # k(p * x + q * y)^2+d * x + e * y + f=0
        #(2k(px0+qy0)p+d)x+(2k(px0+qy0)q+e)y=(2k(px0+qy0)p+d)x0+(2k(px0+qy0)q+e)y0
        k,p, q, d, e, f = pc
        x = pt.x
        y = pt.y
        mid1 = k*(p * x + q * y)
        A = 2 * mid1 * p + d
        B = 2 * mid1 * q + e
        C = A * x + B * y
        return A,B,C

    def ldist(pt,ln):
        a,b,c=ln
        x=pt.x
        y=pt.y
        c=-c # because the old equation is ax+by=e, not ax+by+c=0
        dist=abs(a*x+b*y+c)/sqrt(sqr(a)+sqr(b))
        return dist

    def equ2(a, b, c, d, e, f):
        delta = a * d - b * c
        if abs(delta) < MINE:
            return None
        x = (d * e - b * f) / delta
        y = (a * f - c * e) / delta
        return x, y

    def get_outer_middle_pt(leftpt,rightpt,curve_plane):
        # k(p * x + q * y)^2+d * x + e * y + f=0
        # 2k(px+qy)(p+qy')+d+ey'=0
        # y'=-(2k(px+qy)p+d)/(2k(px+qy)q+e)
        # Also, the tangent function is y-y0=y'(x-x0)
        # so the function is y-y0=-(2k(px0+qy0)p+d)/(2k(px0+qy0)q+e)(x-x0)
        # so, (2k(px0+qy0)q+e)(y-y0)+(2k(px0+qy0)p+d)(x-x0)=0
        # expand: (2k(px0+qy0)p+d)x+(2k(px0+qy0)q+e)y=(2k(px0+qy0)p+d)x0+(2k(px0+qy0)q+e)y0


        #lequ = pre1s(leftpt, curve_plane)
        #requ = pre1s(rightpt, curve_plane)
        lequ = pre1s_k(leftpt, curve_plane)
        requ = pre1s_k(rightpt, curve_plane)
        #ax+by=e
        #cx+dy=f
        a, b, e = lequ
        c, d, f = requ
        # decide if they are same line first. If so, force the middle point to middle of control points.
        dist1=ldist(rightpt,lequ)
        dist2=ldist(leftpt,requ)
        midpt_pre = equ2(a, b, c, d, e, f)
        if dist1<1e-3 and dist2<1e-3:
            # straight case
            midpt=0.5*(leftpt+rightpt)
            midpt_pre=(midpt.x,midpt.y)
        if midpt_pre is not None:
            midx, midy = midpt_pre
            midpt = pos(midx, midy)
            quad_line = lines.quadratic_bezier_curve(leftpt, midpt, rightpt)
            return midpt,quad_line
        else:
            # Only overlap cases, so do nothing
            # print('Error:',leftpt,rightpt,curve_plane)
            return None

    outer_results=[]
    for curr_exists in exists:
        results = []
        for ln in curr_exists:
            a, b, c, d, e = expand_equation_1k(curve_plane, ln)
            cln = lines.quadratic_bezier_curve(pos(ln[0]), pos(ln[1]), pos(ln[2]))
            ts = quartic_equ(a, b, c, d, e)
            new_ts = []
            if ts is not None:
                if not isinstance(ts,list) and not isinstance(ts,tuple):
                    ts=[ts]
                for t in ts:
                    if 0 <= t <= 1:
                        new_ts.append(t)
            if len(new_ts) == 0:
                # decide whether the whole is inside
                midpt = cln.getpos(0.5)
                hmid = cpt_1k(curve_plane, midpt)
                if hmid <= 0:
                    # inside
                    results.append(('E', ln))
                else:
                    continue
            else:
                # split and connect
                lnts = len(new_ts)
                new_ts.sort()
                for j in range(lnts + 1):
                    left = (0.0 if j == 0 else new_ts[j - 1])
                    right = (1.0 if j == lnts else new_ts[j])
                    l_, middle, r_ = cln.split2(left, right)
                    # then decide use which side
                    midpt = middle.getpos(0.5)
                    hmid = cpt_1k(curve_plane, midpt)
                    if hmid <= 0:
                        # inside
                        results.append(('E', middle))
                    else:
                        ## outside
                        ## split the outer curve
                        # lpt, _, rpt = middle.positions
                        # results.append(('P', [None if j == 0 else lpt, None if j == lnts else rpt]))
                        # add 2022.11
                        # calculate inverse to remove the slim curve
                        lpt,mpt,rpt=middle.positions

                        mids=get_outer_middle_pt(lpt,rpt,curve_plane)
                        if mids is not None:
                            _,outerquadline=mids
                            outermiddlepos=outerquadline.getpos(0.5)
                            #h1=-calc_h1_pts(middle,outermiddlepos)# calc middlepos

                            if inside_curr(outermiddlepos,curr_exists):
                                results.append(('P', [None if j == 0 else lpt, None if j == lnts else rpt]))
                            else:
                                results.append(('PC', [None if j == 0 else lpt, None if j == lnts else rpt]))
                        else:
                            results.append(('P', [None if j == 0 else lpt, None if j == lnts else rpt]))

        # connect
        lres = len(results)
        new_meshes = []
        first = -1
        for i in range(lres):
            if results[i][0] == 'E':
                first = i
                break
        if first < 0:
            continue
        new_list = list(range(first, lres))
        new_list.extend(range(first))
        # process states
        new_results=[]
        leftpt = None
        leftflag=False
        for i in new_list:
            if results[i][0]=='E':
                new_results.append(results[i])
            else:
                if leftpt is None:
                    leftpt = results[i][1][0]
                    leftflag=(results[i][0]=='PC')
                if (leftpt is not None) and (results[i][1][1] is not None):
                    # connect left and i[1][1]
                    rightpt = results[i][1][1]
                    midpt_pre=get_outer_middle_pt(leftpt,rightpt,curve_plane)
                    if midpt_pre is not None:
                        midpt,quad_line=midpt_pre
                        new_results.append([('PC' if leftflag else results[i][0]),quad_line])
                        leftpt = None
        # split pcs
        lnr=len(new_results)
        sps=[]
        processed=np.zeros((lnr,lnr),dtype=bool)
        for i in range(lnr):
            if new_results[i][0]=='PC':
                j=i
                if j==0:
                    j=lnr
                while j>0:
                    j-=1
                    if j==i:
                        j=-1
                        break
                    if new_results[j][0]=='P' or new_results[j][0]=='PC':
                        break
                    if j==0:
                        j=lnr
                k=i
                if k==lnr-1:
                    k=0
                while k<lnr-1:
                    k+=1
                    if k==i:
                        k=-1
                        break
                    if new_results[k][0]=='P' or new_results[k][0]=='PC':
                        break
                    if k==lnr-1:
                        k=0
                if j>=0:
                    if processed[i,j] or processed[j,i]:
                        continue
                    _,origin_midjq=get_outer_middle_pt(new_results[j][1].positions[0],new_results[j][1].positions[-1],curve_plane)
                    nearesti0=origin_midjq.argmindist(new_results[i][1].positions[0],-math.inf,math.inf)
                    nearesti1=origin_midjq.argmindist(new_results[i][1].positions[-1],-math.inf,math.inf)
                    posi0=origin_midjq.getpos(nearesti0)
                    posi1=origin_midjq.getpos(nearesti1)
                    if abs(posi0-new_results[i][1].positions[0])>1e-3:
                        continue
                    if abs(posi1-new_results[i][1].positions[-1])>1e-3:
                        continue
                    if 0<=nearesti0<=1 and 0<=nearesti1<=1:
                        jleft=new_results[j][1].positions[0],new_results[i][1].positions[-1]
                        jright=new_results[i][1].positions[0],new_results[j][1].positions[-1]
                        ls_tmp=get_outer_middle_pt(jleft[0],jleft[1],curve_plane)
                        if ls_tmp is None:
                            continue
                        lmid,lmidq=ls_tmp
                        rs_tmp=get_outer_middle_pt(jright[0],jright[1],curve_plane)
                        if rs_tmp is None:
                            continue
                        rmid,rmidq=rs_tmp
                        if inside_curr(lmid,curr_exists) or inside_curr(rmid,curr_exists) or True:
                            # swap
                            new_results[j][1]=lmidq
                            new_results[i][1]=rmidq
                            sps.append((j,i))
                            processed[j,i]=True
                            processed[i,j]=True
                if k>=0:
                    if processed[k,i] or processed[i,k]:
                        continue
                    _,origin_midkq=get_outer_middle_pt(new_results[k][1].positions[0],new_results[k][1].positions[-1],curve_plane)
                    nearesti0=origin_midkq.argmindist(new_results[i][1].positions[0],-math.inf,math.inf)
                    nearesti1=origin_midkq.argmindist(new_results[i][1].positions[-1],-math.inf,math.inf)
                    posi0=origin_midkq.getpos(nearesti0)
                    posi1=origin_midkq.getpos(nearesti1)
                    if abs(posi0-new_results[i][1].positions[0])>1e-3:
                        continue
                    if abs(posi1-new_results[i][1].positions[-1])>1e-3:
                        continue
                    if 0<=nearesti0<=1 and 0<=nearesti1<=1:
                        kleft=new_results[k][1].positions[0],new_results[i][1].positions[-1]
                        kright=new_results[i][1].positions[0],new_results[k][1].positions[-1]
                        ls_tmp=get_outer_middle_pt(kleft[0],kleft[1],curve_plane)
                        if ls_tmp is None:
                            continue
                        lmid,lmidq=ls_tmp
                        rs_tmp=get_outer_middle_pt(kright[0],kright[1],curve_plane)
                        if rs_tmp is None:
                            continue
                        rmid,rmidq=rs_tmp
                        if inside_curr(lmid,curr_exists) or inside_curr(rmid,curr_exists) or True:
                            # swap
                            new_results[i][1]=lmidq
                            new_results[k][1]=rmidq
                            sps.append((i,k))
                            processed[i,k]=True
                            processed[k,i]=True
                else:
                    new_results[i][0]='P'

        flags=np.zeros(lnr,dtype=np.bool_)
        for i in range(len(sps)):
            first=sps[i][0]+1
            last=sps[i][1]+1
            if last < first:
                last+=lnr
            new_mesh_single=[]
            for j in range(first,last):
                new_mesh_single.append(new_results[j%lnr][1])
                flags[j%lnr]=True
            new_meshes.append(new_mesh_single)

        # process remaining
        new_mesh_single=[]
        for j in range(lnr):
            if not flags[j]:
                new_mesh_single.append(new_results[j][1])
        new_meshes.append(new_mesh_single)
        outer_results.extend(new_meshes)

    return outer_results

def inside_multi(pt,meshes):
    lms=len(meshes)
    mind=2147483647
    mindi=-1
    mindj=-1
    tfh=1
    for i in range(lms):
        minid=2147483647
        minidj=-1
        mesh=meshes[i]
        lm=len(mesh)
        for j in range(lm):
            d2=mesh[j].minsqrdist(pt)
            if d2<mind:
                mind=d2
                mindi=i
                mindj=j
            if d2<minid:
                minid=d2
                minidj=i
        if minidj>0:
            j=minidj
            h1j=calc_h1_pts(mesh,j)
            fh=-1 if h1j<-MINE else 0 if h1j<MINE else 1
            tfh*=fh
    return tfh,tfh*sqrt(d2)

def is_clockwise(mesh):
    # 收集所有点
    points = []
    for line in mesh:
        for point in line[0]:   # 最后一个 pos 会出现在下一个点的第一个 pos 因此不必添加
            points.append(point)
    # 找到 x 最大的点 m, 然后考察它相邻两侧的点 l 和 r, 从而计算出当前轮廓的顺逆时针
    x_max, m_idx = -1e300, None
    for i, point in enumerate(points):
        if point.x > x_max:
            x_max = point.x
            m_idx = i
    m_x = points[m_idx].x
    l_idx = (m_idx - 1 + len(points)) % len(points)
    r_idx = (m_idx + 1) % len(points)

    if points[l_idx].x == points[r_idx].x == m_x:   # 如果三点共 x, 会有问题, 因此做一些修正!
        ok = False
        for _ in range(len(points) - 3):    # 先尝试调整 l 点
            l_idx = (l_idx - 1 + len(points)) % len(points)
            if points[l_idx].x != m_x:
                ok = True
                break
        if not ok:              # 调整 l 不行就调整 r 点
            for _ in range(len(points) - 3):
                r_idx = (r_idx + 1) % len(points)
                if points[r_idx].x != m_x:
                    break

    xl, yl = points[l_idx].x, points[l_idx].y
    xm, ym = points[m_idx].x, points[m_idx].y
    xr, yr = points[r_idx].x, points[r_idx].y

    v1_x, v1_y = (xl - xm, yl - ym)
    v2_x, v2_y = (xr - xm, yr - ym)

    # print(v1_x * v2_y - v2_x * v1_y)

    if v1_x * v2_y - v2_x * v1_y <= 0:  # 使用行列式计算定向
        return True
    else:
        return False

def combine_meshes(old_meshes:list,new_meshes:list):
    # both n meshes, not simple
    meshes=old_meshes.copy()
    lnms=len(new_meshes)
    crss=[]
    for i in range(lnms):
        out_meshes,crs=or_meshes(meshes,new_meshes[i])
        crss.extend(crs)
        meshes=out_meshes
    return meshes,crss

def or_meshes(old_meshes:list,new_mesh):
    # the single "new mesh" means single black connection
    meshes=old_meshes.copy()
    lms=len(meshes)
    new_meshes=[new_mesh]
    result_new_meshes=[]
    crossesj=[]
    crossesfinal=[]
    # after combinition, it may be a circle(0\O\D\A) or 8/B or something else
    for i in range(lms):
        om=meshes[i]# assume it is a simple mesh
        for j in range(0,1):
            lom=len(om)
            lmsj=len(new_meshes[j])
            src=[]

            for ii in range(lom):
                crosses=[]
                for jj in range(lmsj):
                    crosspts = calc_two_plane_crosses(om[ii],new_meshes[j][jj])
                    if crosspts is None:
                        crosspts = []
                    crossptsm = []
                    for x in crosspts:
                        t0, t1 = x
                        if 0 - MINE <= t0 <= 1 + MINE and 0 - MINE <= t1 <= 1 + MINE:
                            crossptsm.append([t0, t1])
                    m=len(crossptsm)
                    crossptsm.sort(key=lambda x: x[0])
                    crosses.extend(crossptsm.copy())
                m=len(crosses)
                crosses.sort(key=lambda x: x[0])
                for i3 in range(m+1):
                    lefti=(0.0 if i3 == 0 else crosses[i3 - 1][0])
                    righti = (1.0 if i3 == m else crosses[i3][0])
                    _, exspi, _ = om[ii].split2(lefti, righti)
                    exmi = exspi.getpos(0.5)
                    #exmij=inside_curr(exmi,meshes[j])
                    src.append([exspi,exmi,lefti,righti,i3,i3+1])
            result_new_meshes.append([x[0] for x in src])

            for jj in range(lmsj):
                crosses=[]
                for ii in range(lom):
                    crosspts = calc_two_plane_crosses(om[ii],new_meshes[j][jj])
                    if crosspts is None:
                        crosspts = []
                    crossptsm = []
                    for x in crosspts:
                        t0, t1 = x
                        if 0 - MINE <= t0 <= 1 + MINE and 0 - MINE <= t1 <= 1 + MINE:
                            crossptsm.append([t0, t1])
                    m=len(crossptsm)
                    crossptsm.sort(key=lambda x: x[1])
                    for im in range(m):
                        crossptsm[im].append(im)
                        crossptsm[im].append(ii)
                        crossptsm[im].append(jj)
                    crossesj.extend(crossptsm.copy())
    m=len(crossesj)
    crossesj.sort(key=lambda x: x[1])
    j=0
    lmsj=len(new_meshes[j])
    dst=[]
    for jj in range(lmsj):
        crossjs=[]
        for jjj in range(m):
            if crossesj[jjj][4]==jj:
                crossjs.append(crossesj[jjj])
        mm=len(crossjs)
        for jjj in range(mm + 1):
            leftj = (0.0 if jjj == 0 else crossjs[jjj - 1][1])
            rightj = (1.0 if jjj == mm else crossjs[jjj][1])
            leftorder=(-1 if jjj==0 else crossjs[jjj-1][2])
            rightorder=(-1 if jjj==mm else crossjs[jjj][2])
            leftnum=(-1 if jjj==0 else crossjs[jjj-1][3])
            rightnum=(-1 if jjj==mm else crossjs[jjj][3])
            _, exspj, _ = new_meshes[j][jj].split2(leftj, rightj)

            exmj = exspj.getpos(0.5)
            #exmji=inside_curr(exmj,meshes[j])
            dst.append([exspj,exmj,leftj,rightj,leftorder,rightorder,leftnum,rightnum])
        for jjj in range(mm):
            crossesfinal.append(new_meshes[j][jj].getpos(crossjs[jjj][1]))
    result_new_meshes.append([y[0] for y in dst])
    new_meshes=result_new_meshes
    return new_meshes,crossesfinal




'''
def combine_meshes(old_meshes:list,new_meshes:list):
    meshes=old_meshes.copy()
    lnms=len(new_meshes)
    outer_results=[]
    for i in range(lnms):
        nmsi=new_meshes[i]
        changed=True
        while changed:
            changed=False
            lms=len(meshes)
            for j in range(lms):
                # calc overlap from meshes[j] to nmsi
                lnmsi=len(nmsi)
                lmsj=len(meshes[j])
                src=[]
                dst=[]
                crosses=[]
                for ii in range(lnmsi):
                    for jj in range(lmsj):
                        crosspts = calc_two_plane_crosses(nmsi[ii],meshes[j][jj])
                        if crosspts is None:
                            crosspts = []
                        crossptsm = []
                        for x in crosspts:
                            t0, t1 = x
                            if 0 - MINE <= t0 <= 1 + MINE and 0 - MINE <= t1 <= 1 + MINE:
                                crossptsm.append([t0, t1])
                        m=len(crossptsm)
                        crossptsm.sort(key=lambda x: x[0])
                        for im in range(m):
                            crossptsm[im].append(im)
                        crosses.append(crossptsm.copy())
                        for i3 in range(m+1):
                            lefti=(0.0 if i3 == 0 else crossptsm[i3 - 1][0])
                            righti = (1.0 if i3 == m else crossptsm[i3][0])
                            _, exspi, _ = nmsi[ii].split2(lefti, righti)
                            exmi = exspi.getpos(0.5)
                            exmij=inside_curr(exmi,meshes[j])
                            src.append([exspi,exmi,exmij,lefti,righti,i3,i3+1])

                        crossptsm.sort(key=lambda x: x[1])
                        for jjj in range(m + 1):
                            leftj = (0.0 if jjj == 0 else crossptsm[jjj - 1][1])
                            rightj = (1.0 if jjj == m else crossptsm[jjj][1])
                            leftorder=(-1 if jjj==0 else crossptsm[jjj-1][2])
                            rightorder=(-1 if jjj==m else crossptsm[jjj][2])
                            _, exspj, _ = jj.split2(leftj, rightj)

                            exmj = exspj.getpos(0.5)
                            exmji=inside_curr(exmj,nmsi)
                            dst.append([exspj,exmj,exmji,leftj,rightj,leftorder,rightorder])
                lsrc=len(src)
                ldst=len(dst)
                used_src=np.zeros(lsrc,dtype=bool)
                used_dst=np.zeros(ldst,dtype=bool)

                results=[] # results of combining nmsi and meshes[j]

                while True:
                    stop_flag=None
                    for isrc in range(lsrc):
                        if not used_src[isrc]:
                            if not src[isrc][3]:
                                stop_flag=(0,isrc)
                                break
                    if stop_flag is None:
                        for idst in range(ldst):
                            if not used_dst[idst]:
                                if not dst[idst][3]:
                                    stop_flag=(1,idst)
                                    break
                    if stop_flag is None:
                        break
                    p,t=stop_flag
                    while True:
                        curr=None
                        if p==0:
                            used_src[t]=True
                            curr=src[t]
                        else:
                            used_dst[t]=True
                            curr=dst[t]
                        results.append(curr)
                        currline=curr[0]
                        leftpt=currline[0]
                        rightpt=currline[-1]
                        # same line, next curve
                        k=t
                        nextd=None
                        while True:
                            k+=1
                            if p==0:
                                if k>=lsrc:
                                    k-=lsrc
                                if k==t:
                                    k=-1
                                    break
                                nextd=src[k]
                                nextl=nextd[0]
                                nextp=nextl[0]
                                if abs(nextp-rightpt)<MINE:
                                    break
                            else:
                                # if p==1
                                if k>=ldst:
                                    k-=ldst
                                if k==t:
                                    k=-1
                                    break
                                nextd=dst[k]
                                nextl=nextd[0]
                                nextp=nextl[0]
                                if abs(nextp-rightpt)<MINE:
                                    break
                        if k>=0 and nextd is not None:
                            isin=nextd[2]
                            if not isin:
                                t=k
                                continue
                        # different line
                        p=1-p
                        if p==0:
                            fndsrc=-1
                            nextd=None
                            for isrc in range(lsrc):
                                nextd=dst[isrc]
                                nextl=nextd[0]
                                nextp=nextl[0]
                                if abs(nextp-rightpt)<MINE:
                                    fndsrc=isrc
                                    break
                            if fndsrc>0:
                                isin=nextd[2]
                                if not isin:
                                    t=isrc
                                    continue
                        else:
                            # p==1
                            fnddst=-1
                            nextd=None
                            for idst in range(ldst):
                                nextd=dst[idst]
                                nextl=nextd[0]
                                nextp=nextl[0]
                                if abs(nextp-rightpt)<MINE:
                                    fnddst=idst
                                    break
                            if fnddst>0:
                                isin=nextd[2]
                                if not isin:
                                    t=idst
                                    continue
                        # different line, but find backward
                        if p==0:
                            fndsrc=-1
                            nextd=None
                            for isrc in range(lsrc):
                                nextd=dst[isrc]
                                nextl=nextd[0]
                                nextp=nextl[-1]
                                if abs(nextp-rightpt)<MINE:
                                    fndsrc=isrc
                                    break
                            if fndsrc>0:
                                isin=nextd[2]
                                if not isin:
                                    t=isrc
                                    continue
                        else:
                            # p==1
                            fnddst=-1
                            nextd=None
                            for idst in range(ldst):
                                nextd=dst[idst]
                                nextl=nextd[0]
                                nextp=nextl[-1]
                                if abs(nextp-rightpt)<MINE:
                                    fnddst=idst
                                    break
                            if fnddst>0:
                                isin=nextd[2]
                                if not isin:
                                    t=idst
                                    continue
                nmsi=results
        outer_results.append(nmsi)
    return outer_results





                        overlap_qu.append([ii,jj,crossptsm])
                # combine
                if len(overlap_qu==0):
                    continue
                ii=0
                t=0
                combine_results=[]
                while ii<lnmsi:
                    if overlap_qu[t][0]==ii:
                        jj=overlap_qu[t][1]
                        crossptsm=overlap_qu[t][2]
                        m=len(crossptsm)
                        exsplits=[]
                        jjj=0
                        flag_reverse=False
                        while jjj<m+1:
                            # split
                            lefti=(0.0 if jjj == 0 else crossptsm[jjj - 1][0])
                            righti = (1.0 if jjj == m else crossptsm[jjj][0])
                            _, exspi, _ = nmsi[ii].split2(lefti, righti)
                            exmi = exspi.getpos(0.5)
                            exmij=inside_curr(exmi,meshes[j])
                            if not exmij:
                                combine_results.append(exspi)
                                leftj=0.0 if jjj==0 else crossptsm[jjj-1][1]
                                rightj=1.0 if jjj==m else crossptsm[jjj][1]
                                if m>1:
                                    if jjj==0:
                                        flag_reverse=crossptsm[0][1]>crossptsm[1][1]
                                        _, exspj, _ = meshes[j][jj].split2(leftj, rightj)
                                        exmj = exspj.getpos(0.5)
                                        exmji=inside_curr(exmj,nmsi)
                                    if flag_reverse:
                                        leftj=1.0 if jjj==0 else crossptsm[jjj-1][1]
                                        rightj=0.0 if jjj==m else crossptsm[jjj][1]
                                else:
                                    if jjj==0:
                                        _, exspj, _ = meshes[j][jj].split2(leftj, rightj)
                                        exmj = exspj.getpos(0.5)
                                        exmji=inside_curr(exmj,nmsi)
                                        flag_reverse=exmji
                                    if flag_reverse:
                                        leftj=1.0 if jjj==0 else crossptsm[jjj-1][1]
                                        rightj=0.0 if jjj==m else crossptsm[jjj][1]
                                _, exspj, _ = meshes[j][jj].split2(leftj, rightj)
                                exmj = exspj.getpos(0.5)
                                exmji=inside_curr(exmj,nmsi)
                            jjj+=1
                            ####
                            leftj = (0.0 if jjj == 0 else crossptsm[jjj - 1][1])
                            rightj = (1.0 if jjj == m else crossptsm[jjj][1])
                            _, exspj, _ = meshes[j][jj].split2(leftj, rightj)
                            exmj = exspj.getpos(0.5)
                            exmji=inside_curr(exmj,nmsi)

                            lefti=(0.0 if jjj == 0 else crossptsm[jjj - 1][0])
                            righti = (1.0 if jjj == m else crossptsm[jjj][0])
                            _, exspi, _ = nmsi[ii].split2(lefti, righti)
                            exmi = exspi.getpos(0.5)
                            exmij=inside_curr(exmi,meshes[j])
                            exsplits.append(((exspi,exmi,exmij),(exspj,exmj,exmji)))
                            if not exmij:
                                combine_results.append(exspi)
                                if not exmji:
                                    j4=jj
                                    combine_results.append(j4,)
                            else:
                                j4=jj
                            ####

                    else:
                        combine_results.append(nmsi[ii])

                ####
                        # split old(jj)
                        m = len(crossptsm)
                        exsplits = []
                        for jjj in range(m + 1):
                            left = (0.0 if jjj == 0 else crossptsm[jjj - 1][1])
                            right = (1.0 if jjj == m else crossptsm[jjj][1])
                            _, exsp, _ = jj.split2(left, right)
                            exsplits.append(exsp)
                            exm = exsp.getpos(0.5)
                            # deside if each part(exm) inside nmsi
                            # if inner, then connect it to inner_nmsi
                        '''
