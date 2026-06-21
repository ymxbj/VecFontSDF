from . import pos
import math
import numpy as np

MINE = 1e-6
downgrade_epilson = 0.5

nc = 24
c = np.zeros(shape=[nc, nc], dtype=np.int32)
c[0, 0] = 1
for i in range(nc):
    for j in range(i + 1):
        if (j == 0):
            c[i, j] = 1
        else:
            c[i, j] = c[i - 1, j] + c[i - 1, j - 1]


def sqr(x):
    return x * x


def cubic(x):
    return x * x * x


# def cbrt(x):
#    return (x**(1.0/3))if (x>=0)else -((-x)**(1.0/3))
# just use np.cbrt instead

def cubic_equ(a, b, c, d):
    p = b * c / (6 * sqr(a)) - cubic(b) / cubic(3 * a) - d / (2 * a)
    q = -sqr(b) / sqr(3 * a) + c / (3 * a)
    bdt = sqr(p) + cubic(q)
    if (bdt >= 0):
        dt = math.sqrt(bdt)
        ans = np.cbrt(dt + p) + np.cbrt(-dt + p) - b / (3 * a)
        return ans
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
        return ans


def quadratic_equ(a, b, c, d, e):
    P = (sqr(c) + 12.0 * a * e - 3.0 * b * d) / 9.0
    Q = (27.0 * a * sqr(d) + 2.0 * cubic(c) + 27.0 * sqr(b) * e - 72.0 * a * c * e - 9.0 * b * c * d) / 54.0
    sqrD = complex(sqr(Q) - cubic(P))
    D = sqrD ** (1.0 / 2)
    u1 = (Q + D) ** (1.0 / 3)
    u2 = (Q - D) ** (1.0 / 3)
    u = u2
    if (abs(u1) > abs(u2)):
        u = u1
    v = 0
    if (abs(u) > MINE):
        v = P / u
    w = -0.5 + math.sqrt(3) / 2 * 1.0j
    m1 = (sqr(b) - 8.0 * a * c / 3 + 4 * a * (u + v)) ** (1.0 / 2)
    m2 = (sqr(b) - 8.0 * a * c / 3 + 4 * a * (w * u + w * w * v)) ** (1.0 / 2)
    m3 = (sqr(b) - 8.0 * a * c / 3 + 4 * a * (w * w * u + w * v)) ** (1.0 / 2)
    m = 0
    S = sqr(b) - 8.0 / 3 * a * c
    T = 0
    if (abs(m1) < MINE) and (abs(m2) < MINE) and (abs(m3) < MINE):
        pass
    elif (abs(m1) > abs(m2)) and (abs(m1) > abs(m3)):
        # m1 route
        m = m1
        S = 2 * b * b - 16.0 / 3 * a * c - 4 * a * (u + v)
        T = (8 * a * b * c - 16 * a * a * d - 2 * cubic(b)) / m
    elif (abs(m2) > abs(m3)):
        # m2 route
        m = m2
        S = 2 * b * b - 16.0 / 3 * a * c - 4 * a * (w * u + w * w * v)
        T = (8 * a * b * c - 16 * a * a * d - 2 * cubic(b)) / m
    else:
        # m3
        m = m3
        S = 2 * b * b - 16.0 / 3 * a * c - 4 * a * (w * w * u + w * v)
        T = (8 * a * b * c - 16 * a * a * d - 2 * cubic(b)) / m
    x1 = (-b - m + ((S - T) ** (1.0 / 2))) / (4.0 * a)
    x2 = (-b - m - ((S - T) ** (1.0 / 2))) / (4.0 * a)
    x3 = (-b + m + ((S + T) ** (1.0 / 2))) / (4.0 * a)
    x4 = (-b + m - ((S + T) ** (1.0 / 2))) / (4.0 * a)
    return [x1, x2, x3, x4]


def new_axis():
    import matplotlib.pyplot as plt   # optional, only for debug plotting
    figure = plt.figure()
    return figure.gca()


class line(object):
    def __init__(self):
        self.num = None
        self.positions = []

    def __repr__(self):
        ans = '{' + str(self.num)
        for i in range(self.num + 1):
            ans = ans + ' ' + str(self.positions[i])
        ans = ans + '}'
        return ans

    def __getitem__(self, item):
        return self.positions[item]

    def __setitem__(self, key, value):
        self.positions[key]=value

    def print(self):
        print(self.num, end='')
        for i in self.positions:
            i.print()
        print("\n", end='')

    def draw(self, interpolation_points=20, color=None, alpha=None, ax=None):
        t_vals = np.linspace(0.0, 1.0, interpolation_points)
        xy_vals = [self.getpos(i) for i in t_vals]
        x_vals = [i.x for i in xy_vals]
        y_vals = [i.y for i in xy_vals]
        if (ax is None):
            ax = new_axis()
        ax.plot(x_vals, y_vals, color=color, alpha=alpha)

    def getpos(self, t):
        result = pos.pos()
        n = self.num
        for i in range(n + 1):
            result = result + (c[n, i] * ((1 - t) ** (n - i)) * (t ** i)) * self.positions[i]
        return result

    def speed(self, t):
        result = pos.pos()
        n = self.num
        for i in range(n):
            result = result + (c[n - 1, i] * ((1 - t) ** (n - 1 - i)) * (t ** i)) * n * (
                    self.positions[i + 1] - self.positions[i])
        return result

    def upgrade(self):
        newpositions = [self.positions[0]]
        n = len(self.positions)
        for i in range(n - 1):
            # P_i+1=(i+1)/(n+1)P_i
            j = i + 1
            newpos = j / (n + 1) * self.positions[i] + (n + 1 - j) / (n + 1) * self.positions[j]
            newpositions.append(newpos)
        newpositions.append(self.positions[n - 1])
        # That is because n-level bezier curve has n+1 points
        # self.num = n + 1
        newline = line()
        newline.num = n
        newline.positions = newpositions
        return newline

    def split(self, t):
        def b(i, n, t):
            return c[n, i] * (t ** i) * ((1 - t) ** (n - i))

        n = self.num
        sl = []
        sr = []
        for i in range(n + 1):
            sli = pos.pos()
            sri = pos.pos()
            for j in range(i + 1):
                sli = sli + b(j, i, t) * self.positions[j]
                sri = sri + b(j, i, t) * self.positions[n - i + j]
            sl.append(sli)
            sr.append(sri)
        sr.reverse()
        ll = line()
        ll.num = n
        ll.positions = sl
        ll = ll.to_detail_class()
        lr = line()
        lr.num = n
        lr.positions = sr
        lr = lr.to_detail_class()
        return ll, lr

    def reverse(self):
        self.positions.reverse()

    def split2(self, t0, t1):
        if abs(t1 - t0) < MINE:
            sl, sr = self.split(t0)
            pmid = self.getpos(t0)
            sm = quadratic_bezier_curve(pmid, pmid, pmid)
            return sl, sm, sr
        if t0 > t1:
            sr, sm, sl = self.split2(t1, t0)
            sl.reverse()
            sm.reverse()
            sr.reverse()
            return sl, sm, sr
        '''
        sl,smr=self.split(t0)
        if abs(1-t0)<MINE:
            
        t1_=t0+(t1-t0)/(1-t0)
        sm,sr=smr.split(t1_)

        return sl,sm,sr'''
        pl = self.getpos(t0)
        vl = self.speed(t0)
        pr = self.getpos(t1)
        vr = self.speed(t1)
        a = vl.x * vr.y - vl.y * vr.x
        if abs(a) < MINE:
            # straight line
            sm = quadratic_bezier_curve(pl, 0.5 * (pl + pr), pr)
        else:
            tl = ((pr.x - pl.x) * vr.y - (pr.y - pl.y) * vr.x) / a
            tr = ((pr.x - pl.x) * vl.y - (pr.y - pl.y) * vl.x) / a
            pm = pl + vl * tl
            sm = quadratic_bezier_curve(pl, pm, pr)
        sl, _ = self.split(t0)
        _, sr = self.split(t1)
        return sl, sm, sr

    def split_to_n(self, n):
        if (n == 1):
            return [self]
        else:
            result = []
            curr = self
            while (n > 1):
                l, r = curr.split(1.0 / n)
                result.append(l)
                n = n - 1
                curr = r
            result.append(curr)
            return result

    def forcedegrade(self):
        n = len(self.positions) - 1
        a = np.zeros([n + 1, n])
        for i in range(n):
            a[i, i] = 1 - i / n
            a[i + 1, i] = (i + 1) / n
        at = np.transpose(a)
        ata = np.matmul(at, a)
        ataf1 = np.linalg.inv(ata)
        p = np.array([i.tonparray() for i in self.positions])
        lv = np.matmul(ataf1, at)
        q = np.matmul(lv, p)
        q = list(q)
        q = [pos.pos(i) for i in q]
        newline = line()
        newline.positions = q
        newline.num = n - 1
        return newline

    def downgrade(self, epilson=downgrade_epilson):
        nl = self.forcedegrade()
        nlu = nl.upgrade()
        npos = len(self.positions)
        maxd = 0
        for i in range(npos):
            delpos = self.positions[i] - nlu.positions[i]
            if (abs(delpos) > maxd):
                maxd = abs(delpos)
        # self.print()
        if (maxd > epilson):
            l, r = self.split(t=0.5)
            # print('l:')
            ld = l.downgrade(epilson=epilson)
            # print('r:')
            rd = r.downgrade(epilson=epilson)
            ld.extend(rd)
            return ld
        else:
            return [nl]

    def downgrade_below_3(self, epilson=downgrade_epilson):
        lns = [self]
        while (lns[0].num > 3):
            ress = []
            for i in lns:
                if (i.num > 3):
                    res = i.downgrade(epilson=epilson)
                    ress.extend(res)
                else:
                    ress.extend(i)
            lns = ress
        return lns

    def to_detail_class(self):
        if (self.num == 1):
            return straight_line(self.positions[0], self.positions[1])
        elif (self.num == 2):
            return quadratic_bezier_curve(self.positions[0], self.positions[1], self.positions[2])
        elif (self.num == 3):
            return cubic_bezier_curve(self.positions[0], self.positions[1], self.positions[2], self.positions[3])
        else:
            return self

    def parallel_move(self, pos):
        newpositions = [(i + pos) for i in self.positions]
        self.positions = newpositions

    def normalize(self, max_min_pos):
        newpositions = [pos.pos(i.x / max_min_pos.x, i.y / max_min_pos.y) for i in self.positions]
        self.positions = newpositions


class straight_line(line):
    def __init__(self, p0, p1):
        line.__init__(self)
        self.num = 1
        self.positions = [p0, p1]

    def __repr__(self):
        ans = '{S'
        for i in range(2):
            ans = ans + ' ' + str(self.positions[i])
        ans = ans + '}'
        return ans

    def length(self):
        return abs(self.positions[1] - self.positions[0])

    def getpos(self, t):
        return (1 - t) * self.positions[0] + t * self.positions[1]

    def speed(self, t):
        return self.positions[1] - self.positions[0]

    def partintegral(self, t):
        return self.length() * t

    def draw(self, interpolation_points=2, color=None, alpha=None, ax=None):
        # x_vals = np.linspace(0.0, 1.0, interpolation_points)
        # y_vals = [self.getpos(i) for i in x_vals]
        if (ax is None):
            ax = new_axis()
        # ax.plot(x_vals, y_vals, color=color, alpha=alpha)
        xy_vals = [self.getpos(0.0), self.getpos(1.0)]
        x_vals = [i.x for i in xy_vals]
        y_vals = [i.y for i in xy_vals]
        # print(x_vals, y_vals)
        ax.plot(x_vals, y_vals, color=color, alpha=alpha)

    def argmindist(self, p, mint=0.0, maxt=1.0):
        delta = p - self.positions[0]
        selfdelta = self.positions[1] - self.positions[0]
        sqrlen = sqr(selfdelta.x) + sqr(selfdelta.y)
        if (sqrlen < MINE):
            return (mint + maxt) / 2
        t = (delta.x * selfdelta.x + delta.y * selfdelta.y) / sqrlen
        if (t < mint):
            return mint
        if (t > maxt):
            return maxt
        return t

    def minsqrdist(self, p, mint=0.0, maxt=1.0):
        t = self.argmindist(p, mint, maxt)
        q = self.getpos(t)
        d = p - q
        return sqr(d.x) + sqr(d.y)

    def upgrade(self, t=0.5):
        midpoint = self.getpos(t)
        return quadratic_bezier_curve(self.positions[0], midpoint, self.positions[1])

    def upgrade3(self):
        p0 = self.positions[0]
        p3 = self.positions[1]
        p1 = (1.0 / 3) * (2.0 * p0 + p3)
        p2 = (1.0 / 3) * (p0 + 2.0 * p3)
        return cubic_bezier_curve(p0, p1, p2, p3)

    def get_ts_from_x_raw(self, x):
        p0 = self.positions[0]
        p1 = self.positions[1]
        x0 = p0.x
        x1 = p1.x
        if (abs(x0 - x1) < MINE):
            # if(abs(x-x0)<MINE):
            #    return 0.5
            return []
        # x=(1-t)x_0+tx_1
        t = (x - x0) / (x1 - x0)
        return [t]

    def get_ts_from_x(self, x):
        val = self.get_ts_from_x_raw(x)
        if (len(val) > 0):
            t = val[0]
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                return [t]
        return []

    def get_ys_from_x(self, x):
        p0 = self.positions[0]
        p1 = self.positions[1]
        y0 = p0.y
        y1 = p1.y
        ts = self.get_ts_from_x(x)
        ys = []
        for t in ts:
            y = (1 - t) * y0 + t * y1
            ys.append(y)
        return ys

    def get_ts_from_y_raw(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        y0 = p0.y
        y1 = p1.y
        if (abs(y0 - y1) < MINE):
            # if(abs(y-y0)<MINE):
            #    return 0.5
            return []
        # y=(1-t)y_0+ty_1
        t = (y - y0) / (y1 - y0)
        return [t]

    def get_ts_from_y(self, y):
        val = self.get_ts_from_y_raw(y)
        if (len(val) > 0):
            t = val[0]
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                return [t]
        return []

    def get_xs_from_y(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        x0 = p0.x
        x1 = p1.x
        ts = self.get_ts_from_y(y)
        xs = []
        for t in ts:
            x = (1 - t) * x0 + t * x1
            xs.append(x)
        return xs


class quadratic_bezier_curve(line):
    def __init__(self, p0, p1, p2):
        line.__init__(self)
        self.num = 2
        self.positions = [p0, p1, p2]

    def __repr__(self):
        ans = '{Q'
        for i in range(3):
            ans = ans + ' ' + str(self.positions[i])
        ans = ans + '}'
        return ans

    def print_to_svg_path(self):
        ans = '<path d="M ' + str(self.positions[0].x) + ' ' + str(self.positions[0].y) + ' Q ' + str(
            self.positions[1].x) + ' ' + str(self.positions[1].y) + ' ' + str(self.positions[2].x) + ' ' + str(
            self.positions[2].y) + '"/>'
        return ans

    def print_to_svg_path_inner(self,isfirst=False):
        ans=''
        if isfirst:
            ans = 'M ' + str(self.positions[0].x) + ' ' + str(self.positions[0].y) + ' '
        ans=ans+'Q ' + str(
            self.positions[1].x) + ' ' + str(self.positions[1].y) + ' ' + str(self.positions[2].x) + ' ' + str(
            self.positions[2].y) + ' '
        return ans

    def print_to_svg_circle(self, radius=0.01, stroke_width=0.01):
        ans1 = '<circle cx="' + str(self.positions[0].x) + '" cy="' + str(self.positions[0].y) + '" r="' + str(
            radius) + '" stroke="Green" stroke-width="'+str(stroke_width)+'"/>'
        ans2 = '<circle cx="' + str(self.positions[1].x) + '" cy="' + str(self.positions[1].y) + '" r="' + str(
            radius) + '" stroke="Red" stroke-width="'+str(stroke_width)+'"/>'
        return ans1 + '\n' + ans2

    def getpos(self, t):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        return sqr(1 - t) * p0 + 2 * t * (1 - t) * p1 + sqr(t) * p2

    def speed(self, t):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        return 2 * (p0 - 2 * p1 + p2) * t + (-2 * p0 + 2 * p1)

    def partintegral(self, t):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        pa = p0 - 2 * p1 + p2
        xa = pa.x
        ya = pa.y
        pb = p1 - p0
        xb = pb.x
        yb = pb.y
        a = sqr(xa) + sqr(ya)
        b = 2 * (xa * xb + ya * yb)
        c = sqr(xb) + sqr(yb)
        if (abs((p1.y - p0.y) * (p2.x - p0.x) - (p2.y - p0.y) * (p1.x - p0.x)) < MINE):
            # it means that this is a straight line
            return straight_line(p0, p2).partintegral(t)
        integralresult = (2 * a * t + b) * math.sqrt(t * (a * t + b) + c) / (2 * a) - (sqr(b) - 4 * a * c) * math.log(
            2 * math.sqrt(a) * math.sqrt(t * (a * t + b) + c) + 2 * a * t + b) / (4 * cubic(math.sqrt(a)))
        return integralresult

    def length(self):
        return self.partintegral(t=1.0) - self.partintegral(t=0.0)

    def draw(self, interpolation_points=20, color=None, alpha=None, ax=None):
        t_vals = np.linspace(0.0, 1.0, interpolation_points)
        xy_vals = [self.getpos(i) for i in t_vals]
        x_vals = [i.x for i in xy_vals]
        y_vals = [i.y for i in xy_vals]
        if (ax is None):
            ax = new_axis()
        ax.plot(x_vals, y_vals, color=color, alpha=alpha)

    def argmindist(self, p, mint=0.0, maxt=1.0):
        x1 = self.positions[0].x
        y1 = self.positions[0].y
        x2 = self.positions[1].x
        y2 = self.positions[1].y
        x3 = self.positions[2].x
        y3 = self.positions[2].y
        x0 = p.x
        y0 = p.y
        a = (
                2.0 * (x1 - 2 * x2 + x3) * (2 * x1 - 4 * x2 + 2 * x3) + 2.0 * (y1 - 2 * y2 + y3) * (
                2 * y1 - 4 * y2 + 2 * y3))
        if (a < MINE):
            l = straight_line(self.positions[0], self.positions[2])
            t = l.argmindist(p, mint, maxt)
            return t
        b = (-2.0 * (2 * x1 - 2 * x2) * (2 * x1 - 4 * x2 + 2 * x3) - 2.0 * (2 * y1 - 2 * y2) * (
                2 * y1 - 4 * y2 + 2 * y3) - 2 * (2 * x1 - 2 * x2) * (x1 - 2 * x2 + x3) - 2 * (2 * y1 - 2 * y2) * (
                     y1 - 2 * y2 + y3))
        c = (
                2.0 * sqr(2 * x1 - 2 * x2) + 2.0 * sqr(2 * y1 - 2 * y2) - 2 * (x0 - x1) * (
                2 * x1 - 4 * x2 + 2 * x3) - 2 * (
                        y0 - y1) * (2 * y1 - 4 * y2 + 2 * y3))
        d = 2.0 * (2 * x1 - 2 * x2) * (x0 - x1) + 2.0 * (2 * y1 - 2 * y2) * (y0 - y1)
        t = cubic_equ(a, b, c, d)
        if (t < mint):
            return mint
        if (t > maxt):
            return maxt
        return t

    def minsqrdist(self, p, mint=0.0, maxt=1.0):
        t = self.argmindist(p, mint, maxt)
        q = self.getpos(t)
        d = p - q
        return sqr(d.x) + sqr(d.y)

    def upgrade(self):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        np0 = p0
        np1 = 1.0 / 3 * p0 + 2.0 / 3 * p1
        np2 = 2.0 / 3 * p1 + 1.0 / 3 * p2
        np3 = p2
        return cubic_bezier_curve(np0, np1, np2, np3)

    def get_ts_from_x_raw(self, x):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        pa = p0 - 2 * p1 + p2
        pb = 2 * (p1 - p0)
        # pc=p0-p
        xa = pa.x
        xb = pb.x
        xc = p0.x - x
        if (abs(xa) > MINE):
            delta = xb * xb - 4 * xa * xc
            if (abs(delta) < MINE):
                return [-xb / (2 * xa)]
            else:
                if (delta < 0):
                    return []
                else:
                    # delta>0
                    t1 = (-xb + math.sqrt(delta)) / (2 * xa)
                    t2 = (-xb - math.sqrt(delta)) / (2 * xa)
                    return [t1, t2]
        else:
            if (abs(xb) > MINE):
                return [-xc / xb]
            else:
                return []

    def get_ts_from_x(self, x):
        val = self.get_ts_from_x_raw(x)
        ts = []
        for t in val:
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                ts.append(t)
        return ts

    def get_ys_from_x(self, x):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        y0 = p0.y
        y1 = p1.y
        y2 = p2.y
        ts = self.get_ts_from_x(x)
        ys = []
        for t in ts:
            y = (1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2
            ys.append(y)
        return ys

    def get_ts_from_y_raw(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        pa = p0 - 2 * p1 + p2
        pb = 2 * (p1 - p0)
        # pc=p0-p
        ya = pa.y
        yb = pb.y
        yc = p0.y - y
        if (abs(ya) > MINE):
            delta = yb * yb - 4 * ya * yc
            if (abs(delta) < MINE):
                return [-yb / (2 * ya)]
            else:
                if (delta < 0):
                    return []
                else:
                    # delta>0
                    t1 = (-yb + math.sqrt(delta)) / (2 * ya)
                    t2 = (-yb - math.sqrt(delta)) / (2 * ya)
                    return [t1, t2]
        else:
            if (abs(yb) > MINE):
                return [-yc / yb]
            else:
                return []

    def get_ts_from_y(self, y):
        val = self.get_ts_from_y_raw(y)
        ts = []
        for t in val:
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                ts.append(t)
        return ts

    def get_xs_from_y(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        x0 = p0.x
        x1 = p1.x
        x2 = p2.x
        ts = self.get_ts_from_y(y)
        xs = []
        for t in ts:
            x = (1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2
            xs.append(x)
        return xs


class cubic_bezier_curve(line):
    def __init__(self, p0, p1, p2, p3):
        line.__init__(self)
        self.num = 3
        self.positions = [p0, p1, p2, p3]

    def getpos(self, t):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        return cubic(1 - t) * p0 + 3 * t * sqr(1 - t) * p1 + 3 * sqr(t) * (1 - t) * p2 + cubic(t) * p3

    def speed(self, t):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        return -3 * sqr(1 - t) * p0 + (9 * sqr(t) - 12 * t + 3) * p1 + (-9 * sqr(t) + 6 * t) * p2 + 3 * sqr(t) * p3

    def partintegral(self, t):  # integral(sqrt(v(x)^2+v(y)^2)dt,0,t)
        if (t == 0):
            return 0
        integral_innerfunc = 0
        slice = 100
        if (t > 0):
            lastpt = self.getpos(0)
            delta = t / slice
            for i in range(slice):
                j = i + 1
                nextpt = self.getpos(j * delta)
                dist = abs(nextpt - lastpt)
                integral_innerfunc += dist
        else:
            lastpt = self.getpos(0)
            delta = t / slice
            for i in range(slice):
                j = i + 1
                nextpt = self.getpos(j * delta)
                dist = abs(nextpt - lastpt)
                integral_innerfunc += dist
        return integral_innerfunc

    def downgradecheck(self, l2, epilson=downgrade_epilson):
        midp = self.getpos(t=0.5)
        l2p = l2.getpos(t=0.5)
        return (abs(midp - l2p) < epilson)

    def downgrade(self, epilson=downgrade_epilson):
        # epilson = 1e-8
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]

        # test straight line
        # we just need to check distance of p1 and p2 is smaller than line p0p3.
        s = straight_line(p0, p3)
        d1 = s.minsqrdist(p1)
        d2 = s.minsqrdist(p2)
        if ((d1 < epilson) and (d2 < epilson)):
            pm = 0.5 * (p0 + p3)
            midcurve = quadratic_bezier_curve(p0, pm, p3)
            return [midcurve]

        # delta = (p1.y - 2 * p0.y) * (2 * p3.x - p2.x) - (p1.x - 2 * p0.x) * (2 * p3.y - p2.y)
        delta = (p1.x - p0.x) * (p3.y - p2.y) - (p1.y - p0.y) * (p3.x - p2.x)
        if (abs(delta) > 1e-8):
            # t1 = ((p3.x - p0.x) * (2 * p3.y - p2.y) - (p3.y - p0.y) * (p3.x - p2.x)) / delta
            # t2 = ((p3.x - p0.x) * (p1.y - 2 * p0.y) - (p3.y - p0.y) * (p1.x - 2 * p0.x)) / delta
            t1 = ((p3.y - p2.y) * (p3.x - p0.x) - (p3.x - p2.x) * (p3.y - p0.y)) / delta
            t2 = ((p1.x - p0.x) * (p3.y - p0.y) - (p3.x - p0.x) * (p1.y - p0.y)) / delta
            if (t1 > 1) and (t2 > 1):
                pm = straight_line(p0, p1).getpos(t1)
                # pm is also straightline(p3,p2).getpos(t2)
                midcurve = quadratic_bezier_curve(p0, pm, p3)
                if (self.downgradecheck(midcurve, epilson=epilson)):
                    return [midcurve]

        p01 = (p0 + p1) * 0.5
        p12 = (p1 + p2) * 0.5
        p23 = (p2 + p3) * 0.5
        pa = (p01 + p12) * 0.5
        pb = (p12 + p23) * 0.5
        pc = (pa + pb) * 0.5
        # split them into 2 parts
        resultl = cubic_bezier_curve(p0, p01, pa, pc).downgrade()
        resultr = cubic_bezier_curve(pc, pb, p23, p3).downgrade()
        resultl.extend(resultr)
        return resultl

    def draw(self, interpolation_points=20, color=None, alpha=None, ax=None):
        t_vals = np.linspace(0.0, 1.0, interpolation_points)
        xy_vals = [self.getpos(i) for i in t_vals]
        x_vals = [i.x for i in xy_vals]
        y_vals = [i.y for i in xy_vals]
        if (ax is None):
            ax = new_axis()
        ax.plot(x_vals, y_vals, color=color, alpha=alpha)

    def argmindist(self, p, mint=0.0, maxt=1.0):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        x_0 = p0.x
        y_0 = p0.y
        x_1 = p1.x
        y_1 = p1.y
        x_2 = p2.x
        y_2 = p2.y
        x_3 = p3.x
        y_3 = p3.y
        x_p = p.x
        y_p = p.y
        # x_0^2+9*x_1^2+9*x_2^2+x_3^2+y_0^2+9*y_1^2+9*y_2^2+y_3^2-6*x_0*x_1+6*x_0*x_2-18*x_1*x_2-2*x_0*x_3+6*x_1*x_3-6*x_2*x_3-6*y_0*y_1+6*y_0*y_2-18*y_1*y_2-2*y_0*y_3+6*y_1*y_3-6*y_2*y_3
        '''
        f6 = sqr(x_0) - 6 * x_1 * x_0 + 6 * x_2 * x_0 + 2 * x_3 * x_0 + 9 * sqr(x_1) + 9 * sqr(x_2) + sqr(x_3) + sqr(
            y_0) + 9 * sqr(y_1) + 9 * sqr(y_2) + sqr(
            y_3) - 18 * x_1 * x_2 - 6 * x_1 * x_3 + 6 * x_2 * x_3 - 6 * y_0 * y_1 + 6 * y_0 * y_2 - 18 * y_1 * y_2 + 2 * y_0 * y_3 - 6 * y_1 * y_3 + 6 * y_2 * y_3
        f5 = -6 * sqr(x_0) + 30 * x_1 * x_0 - 24 * x_2 * x_0 - 12 * x_3 * x_0 - 36 * sqr(x_1) - 18 * sqr(x_2) - 6 * sqr(
            x_3) - 6 * sqr(y_0) - 36 * sqr(y_1) - 18 * sqr(y_2) - 6 * sqr(
            y_3) + 54 * x_1 * x_2 + 30 * x_1 * x_3 - 24 * x_2 * x_3 + 30 * y_0 * y_1 - 24 * y_0 * y_2 + 54 * y_1 * y_2 - 12 * y_0 * y_3 + 30 * y_1 * y_3 - 24 * y_2 * y_3
        f4 = 15 * sqr(x_0) - 60 * x_1 * x_0 + 36 * x_2 * x_0 + 30 * x_3 * x_0 + 54 * sqr(x_1) + 9 * sqr(x_2) + 15 * sqr(
            x_3) + 15 * sqr(y_0) + 54 * sqr(y_1) + 9 * sqr(y_2) + 15 * sqr(
            y_3) - 54 * x_1 * x_2 - 60 * x_1 * x_3 + 36 * x_2 * x_3 - 60 * y_0 * y_1 + 36 * y_0 * y_2 - 54 * y_1 * y_2 + 30 * y_0 * y_3 - 60 * y_1 * y_3 + 36 * y_2 * y_3
        f3 = -20 * sqr(x_0) + 2 * x_p * x_0 + 60 * x_1 * x_0 - 24 * x_2 * x_0 - 40 * x_3 * x_0 - 36 * sqr(
            x_1) - 20 * sqr(x_3) - 20 * sqr(y_0) - 36 * sqr(y_1) - 20 * sqr(
            y_3) - 6 * x_p * x_1 + 6 * x_p * x_2 + 18 * x_1 * x_2 + 2 * x_p * x_3 + 60 * x_1 * x_3 - 24 * x_2 * x_3 + 2 * y_p * y_0 - 6 * y_p * y_1 + 60 * y_0 * y_1 + 6 * y_p * y_2 - 24 * y_0 * y_2 + 18 * y_1 * y_2 + 2 * y_p * y_3 - 40 * y_0 * y_3 + 60 * y_1 * y_3 - 24 * y_2 * y_3
        f2 = 15 * sqr(x_0) - 6 * x_p * x_0 - 30 * x_1 * x_0 + 6 * x_2 * x_0 + 30 * x_3 * x_0 + 9 * sqr(x_1) + 15 * sqr(
            x_3) + 15 * sqr(y_0) + 9 * sqr(y_1) + 15 * sqr(
            y_3) + 12 * x_p * x_1 - 6 * x_p * x_2 - 6 * x_p * x_3 - 30 * x_1 * x_3 + 6 * x_2 * x_3 - 6 * y_p * y_0 + 12 * y_p * y_1 - 30 * y_0 * y_1 - 6 * y_p * y_2 + 6 * y_0 * y_2 - 6 * y_p * y_3 + 30 * y_0 * y_3 - 30 * y_1 * y_3 + 6 * y_2 * y_3
        f1 = -6 * sqr(x_0) + 6 * x_p * x_0 + 6 * x_1 * x_0 - 12 * x_3 * x_0 - 6 * sqr(x_3) - 6 * sqr(y_0) - 6 * sqr(
            y_3) - 6 * x_p * x_1 + 6 * x_p * x_3 + 6 * x_1 * x_3 + 6 * y_p * y_0 - 6 * y_p * y_1 + 6 * y_0 * y_1 + 6 * y_p * y_3 - 12 * y_0 * y_3 + 6 * y_1 * y_3
        f0 = sqr(x_p) - 2 * x_0 * x_p - 2 * x_3 * x_p + sqr(y_p) + sqr(x_0) + sqr(x_3) + sqr(y_0) + sqr(
            y_3) + 2 * x_0 * x_3 - 2 * y_p * y_0 - 2 * y_p * y_3 + 2 * y_0 * y_3
        '''

        f6 = sqr(x_0) + 9 * sqr(x_1) + 9 * sqr(x_2) + sqr(x_3) + sqr(y_0) + 9 * sqr(y_1) + 9 * sqr(y_2) + sqr(
            y_3) - 6 * x_0 * x_1 + 6 * x_0 * x_2 - 18 * x_1 * x_2 - 2 * x_0 * x_3 + 6 * x_1 * x_3 - 6 * x_2 * x_3 - 6 * y_0 * y_1 + 6 * y_0 * y_2 - 18 * y_1 * y_2 - 2 * y_0 * y_3 + 6 * y_1 * y_3 - 6 * y_2 * y_3

        # f6=sqr(x_0-3*x_1+3*x_2-x_3)+sqr(y_0-3*y_1+3*y_2-y_3)

        if (f6 < MINE):
            # that means, p0-3p1+3p2-p3=0
            midp = 0.5 * (3 * p1 - p0)
            # midp = 0.5*(3*p2-*p3)
            q = quadratic_bezier_curve(p0, midp, p2)
            return q.argmindist(p, mint, maxt)

        f5 = -6 * sqr(x_0) - 36 * sqr(x_1) - 18 * sqr(x_2) - 6 * sqr(y_0) - 36 * sqr(y_1) - 18 * sqr(
            y_2) + 30 * x_0 * x_1 - 24 * x_0 * x_2 + 54 * x_1 * x_2 + 6 * x_0 * x_3 - 12 * x_1 * x_3 + 6 * x_2 * x_3 + 30 * y_0 * y_1 - 24 * y_0 * y_2 + 54 * y_1 * y_2 + 6 * y_0 * y_3 - 12 * y_1 * y_3 + 6 * y_2 * y_3
        f4 = 15 * sqr(x_0) + 54 * sqr(x_1) + 9 * sqr(x_2) + 15 * sqr(y_0) + 54 * sqr(y_1) + 9 * sqr(
            y_2) - 60 * x_0 * x_1 + 36 * x_0 * x_2 - 54 * x_1 * x_2 - 6 * x_0 * x_3 + 6 * x_1 * x_3 - 60 * y_0 * y_1 + 36 * y_0 * y_2 - 54 * y_1 * y_2 - 6 * y_0 * y_3 + 6 * y_1 * y_3
        f3 = -20 * sqr(x_0) - 36 * sqr(x_1) - 20 * sqr(y_0) - 36 * sqr(
            y_1) + 2 * x_p * x_0 - 6 * x_p * x_1 + 60 * x_0 * x_1 + 6 * x_p * x_2 - 24 * x_0 * x_2 + 18 * x_1 * x_2 - 2 * x_p * x_3 + 2 * x_0 * x_3 + 2 * y_p * y_0 - 6 * y_p * y_1 + 60 * y_0 * y_1 + 6 * y_p * y_2 - 24 * y_0 * y_2 + 18 * y_1 * y_2 - 2 * y_p * y_3 + 2 * y_0 * y_3
        f2 = 15 * sqr(x_0) + 9 * sqr(x_1) + 15 * sqr(y_0) + 9 * sqr(
            y_1) - 6 * x_p * x_0 + 12 * x_p * x_1 - 30 * x_0 * x_1 - 6 * x_p * x_2 + 6 * x_0 * x_2 - 6 * y_p * y_0 + 12 * y_p * y_1 - 30 * y_0 * y_1 - 6 * y_p * y_2 + 6 * y_0 * y_2
        f1 = -6 * sqr(x_0) - 6 * sqr(
            y_0) + 6 * x_p * x_0 - 6 * x_p * x_1 + 6 * x_0 * x_1 + 6 * y_p * y_0 - 6 * y_p * y_1 + 6 * y_0 * y_1
        f0 = sqr(x_p) + sqr(y_p) + sqr(x_0) + sqr(y_0) - 2 * x_p * x_0 - 2 * y_p * y_0
        # f=f6t^6+f5t^5+f4t^4+f3t^3+f2t^2+f1t+f0
        # df=6f6t^5+5f5t^4+4f4t^3+3f3t^2+2f2t+f1=0
        # ddf=30f6t^4+20f5t^3+12f4t^2+6f3t+2f2=0
        # we need to solve the minimized value of f
        # f(t) is a 6-grade equation, so we need at most 3 times up and 3 times down to get the minimal.
        # that is , check the two side position and all points df=0
        # to solve df=0, we just need to find roots between two side and all ddf=0, each range has at most one root

        # step 1: find roots of ddf=0
        ddf0 = quadratic_equ(30 * f6, 20 * f5, 12 * f4, 6 * f3, 2 * f2)
        resddf = []  # roots that ddf=0, with mint,maxt
        for x in ddf0:
            # check if this root is real and between mint,maxt
            if (abs(x.imag) < MINE):
                if (x.real > mint) and (x.real < maxt):
                    resddf.append(x.real)
        resddf.append(mint)
        resddf.append(maxt)
        resddf.sort()
        lddf = len(resddf) - 1

        def df(t):
            return 6 * f6 * t * t * t * t * t + 5 * f5 * t * t * t * t + 4 * f4 * t * t * t + 3 * f3 * t * t + 2 * f2 * t + f1

        resdf = [mint]  # end points and solotions that df=0
        for i in range(lddf):
            xl = resddf[i]
            xr = resddf[i + 1]
            yl = df(xl)
            yr = df(xr)
            if (yl * yr < 0):
                while (True):
                    xm = 0.5 * (xl + xr)
                    ym = df(xm)
                    if (abs(ym) < MINE):
                        resdf.append(xm)
                        break
                    if (yl * ym < 0):
                        xr = xm
                    else:
                        xl = xm
        resdf.append(maxt)

        def f(t):
            return f6 * sqr(cubic(t)) + f5 * sqr(t) * cubic(t) + f4 * sqr(sqr(t)) + f3 * cubic(t) + f2 * sqr(
                t) + f1 * t + f0

        '''
        resf = [mint]
        ldf = len(resdf) - 1
        for i in range(ldf):
            xl = resdf[i]
            xr = resdf[i + 1]
            yl = f(xl)
            yr = f(xr)
            if (yl * yr < 0):
                while (True):
                    xm = 0.5 * (xl + xr)
                    ym = f(xm)
                    if (abs(ym) < MINE):
                        resf.append(xm)
                        break
                    if (yl * ym < 0):
                        xr = xm
                    else:
                        xl = xm
        resf.append(maxt)
        '''

        minx = mint
        miny = f(mint)
        for i in resdf:
            if (f(i) < miny):
                minx = i
                miny = f(i)
        return minx

    def minsqrdist(self, p, mint=0.0, maxt=1.0):
        t = self.argmindist(p, mint, maxt)
        q = self.getpos(t)
        d = p - q
        return sqr(d.x) + sqr(d.y)

    def get_ts_from_x_raw(self, x):
        # cubic(1 - t) * p0 + 3 * t * sqr(1 - t) * p1 + 3 * sqr(t) * (1 - t) * p2 + cubic(t) * p3
        # equal to
        # p_3t^3+3p_2t^2-3p_2t^3-6p_1t^2+3p_1t^3+p_0+3p_0t^2-p_0t^3+3tp_1-3tp_0-p=0
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        pa = p3 - 3 * p2 + 3 * p1 - p0
        pb = 3 * p2 - 6 * p1 + 3 * p0
        pc = 3 * (p1 - p0)
        # pd=p0-p
        xa = pa.x
        xb = pb.x
        xc = pc.x
        xd = p0.x - x
        if (abs(xa) > MINE):
            single_root = cubic_equ(xa, xb, xc, xd)
            da = xa
            db = xb + xa * single_root
            dc = xc + db * single_root
            delta = sqr(db) - 4 * da * dc
            if (abs(delta) < MINE):
                x2 = -db / (2 * da)
                return [single_root, x2]
            elif (delta < 0):
                return [single_root]
            else:
                # delta>0
                sqrtdelta = math.sqrt(delta)
                x2 = (-db + sqrtdelta) / (2 * da)
                x3 = (-db - sqrtdelta) / (2 * da)
                return [single_root, x2, x3]
        else:
            da = xb
            db = xc
            dc = xd
            if (abs(da) < MINE):
                za = db
                zb = dc
                if (abs(za) < MINE):
                    return []
                return [-zb / za]
            delta = sqr(db) - 4 * da * dc
            if (abs(delta) < MINE):
                x2 = -db / (2 * da)
                return [x2]
            elif (delta < 0):
                return []
            else:
                # delta>0
                sqrtdelta = math.sqrt(delta)
                x2 = (-db + sqrtdelta) / (2 * da)
                x3 = (-db - sqrtdelta) / (2 * da)
                return [x2, x3]

    def get_ts_from_x(self, x):
        val = self.get_ts_from_x_raw(x)
        ans = []
        for t in val:
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                ans.append(t)
        return ans

    def get_ys_from_x(self, x):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        y0 = p0.y
        y1 = p1.y
        y2 = p2.y
        y3 = p3.y
        ts = self.get_ts_from_x(x)
        ys = []
        for t in ts:
            # cubic(1 - t) * p0 + 3 * t * sqr(1 - t) * p1 + 3 * sqr(t) * (1 - t) * p2 + cubic(t) * p3
            y = cubic(1 - t) * y0 + 3 * t * sqr(1 - t) * y1 + 3 * sqr(t) * (1 - t) * y2 + cubic(t) * y3
            ys.append(y)
        return ys

    def get_ts_from_y_raw(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        pa = p3 - 3 * p2 + 3 * p1 - p0
        pb = 3 * p2 - 6 * p1 + 3 * p0
        pc = 3 * (p1 - p0)
        # pd=p0-p
        ya = pa.y
        yb = pb.y
        yc = pc.y
        yd = p0.y - y
        if (abs(ya) > MINE):
            single_root = cubic_equ(ya, yb, yc, yd)
            da = ya
            db = yb + ya * single_root
            dc = yc + db * single_root
            delta = sqr(db) - 4 * da * dc
            if (abs(delta) < MINE):
                y2 = -db / (2 * da)
                return [single_root, y2]
            elif (delta < 0):
                return [single_root]
            else:
                # delta>0
                sqrtdelta = math.sqrt(delta)
                y2 = (-db + sqrtdelta) / (2 * da)
                y3 = (-db - sqrtdelta) / (2 * da)
                return [single_root, y2, y3]
        else:
            da = yb
            if (abs(da) < MINE):
                za = yc
                zb = yd
                if (abs(za) < MINE):
                    return []
                return [-zb / za]
            db = yc
            dc = yd
            delta = sqr(db) - 4 * da * dc
            if (abs(delta) < MINE):
                y2 = -db / (2 * da)
                return [y2]
            elif (delta < 0):
                return []
            else:
                # delta>0
                sqrtdelta = math.sqrt(delta)
                y2 = (-db + sqrtdelta) / (2 * da)
                y3 = (-db - sqrtdelta) / (2 * da)
                return [y2, y3]

    def get_ts_from_y(self, y):
        val = self.get_ts_from_y_raw(y)
        ans = []
        for t in val:
            if (t >= -MINE) and (t < 1 - MINE):
                # t>=0 and t<1
                ans.append(t)
        return ans

    def get_xs_from_y(self, y):
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        p3 = self.positions[3]
        x0 = p0.x
        x1 = p1.x
        x2 = p2.x
        x3 = p3.x
        ts = self.get_ts_from_y(y)
        xs = []
        for t in ts:
            # cubic(1 - t) * p0 + 3 * t * sqr(1 - t) * p1 + 3 * sqr(t) * (1 - t) * p2 + cubic(t) * p3
            x = cubic(1 - t) * x0 + 3 * t * sqr(1 - t) * x1 + 3 * sqr(t) * (1 - t) * x2 + cubic(t) * x3
            xs.append(x)
        return xs
