"""2D point, straight line, and quadratic Bezier curve primitives.

Each curve exposes:
    length()                       arc length
    getpos(t)                      point at parameter t in [0, 1]
    minsqrdist(p)                  squared distance from point p to the curve
    get_ys_from_x(x), get_xs_from_y(y)   horizontal / vertical ray-curve
                                         intersections (used by the inside test)
"""

import math
from typing import List

EPS = 1e-7


def _sqr(x: float) -> float:
    return x * x


def _solve_quadratic(a: float, b: float, c: float) -> List[float]:
    """Real roots of a t^2 + b t + c = 0."""
    if abs(a) < EPS:
        if abs(b) < EPS:
            return []
        return [-c / b]
    delta = b * b - 4 * a * c
    if abs(delta) < EPS:
        return [-b / (2 * a)]
    if delta < 0:
        return []
    sd = math.sqrt(delta)
    return [(-b - sd) / (2 * a), (-b + sd) / (2 * a)]


def _solve_cubic(a: float, b: float, c: float, d: float) -> List[float]:
    """Real roots of a t^3 + b t^2 + c t + d = 0."""
    if abs(a) < EPS:
        return _solve_quadratic(b, c, d)
    # Depress to t^3 + p t + q = 0 via substitution t = u - b / (3a).
    p = c / a - b * b / (3 * a * a)
    q = (2 * b * b * b) / (27 * a * a * a) - (b * c) / (3 * a * a) + d / a
    disc = (q / 2) ** 2 + (p / 3) ** 3
    shift = -b / (3 * a)
    roots: List[float] = []
    if disc > EPS:
        sd = math.sqrt(disc)
        u = -q / 2 + sd
        v = -q / 2 - sd
        roots.append(math.copysign(abs(u) ** (1 / 3), u)
                     + math.copysign(abs(v) ** (1 / 3), v)
                     + shift)
    elif disc < -EPS:
        r = math.sqrt(-(p / 3) ** 3)
        phi = math.acos(max(-1.0, min(1.0, -q / (2 * r))))
        m = 2 * (-p / 3) ** 0.5
        for k in range(3):
            roots.append(m * math.cos((phi + 2 * math.pi * k) / 3) + shift)
    else:
        u = math.copysign(abs(q / 2) ** (1 / 3), -q / 2)
        roots.append(2 * u + shift)
        roots.append(-u + shift)
    return roots


class Pos:
    """2D point with arithmetic operator overloads."""

    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: 'Pos') -> 'Pos':
        return Pos(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Pos') -> 'Pos':
        return Pos(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> 'Pos':
        return Pos(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f'({self.x}, {self.y})'


class StraightLine:
    """Straight segment from p0 to p1."""

    def __init__(self, p0: Pos, p1: Pos):
        self.p0 = p0
        self.p1 = p1

    def length(self) -> float:
        dx = self.p1.x - self.p0.x
        dy = self.p1.y - self.p0.y
        return math.hypot(dx, dy)

    def getpos(self, t: float) -> Pos:
        return Pos((1 - t) * self.p0.x + t * self.p1.x,
                   (1 - t) * self.p0.y + t * self.p1.y)

    def minsqrdist(self, p: Pos) -> float:
        dx = self.p1.x - self.p0.x
        dy = self.p1.y - self.p0.y
        sqrlen = dx * dx + dy * dy
        if sqrlen < EPS:
            ex = p.x - self.p0.x
            ey = p.y - self.p0.y
            return ex * ex + ey * ey
        t = ((p.x - self.p0.x) * dx + (p.y - self.p0.y) * dy) / sqrlen
        t = max(0.0, min(1.0, t))
        qx = (1 - t) * self.p0.x + t * self.p1.x
        qy = (1 - t) * self.p0.y + t * self.p1.y
        return (p.x - qx) ** 2 + (p.y - qy) ** 2

    def get_ys_from_x(self, x: float) -> List[float]:
        x0, x1 = self.p0.x, self.p1.x
        if abs(x0 - x1) < EPS:
            return []
        t = (x - x0) / (x1 - x0)
        if t < -EPS or t >= 1 - EPS:
            return []
        return [(1 - t) * self.p0.y + t * self.p1.y]

    def get_xs_from_y(self, y: float) -> List[float]:
        y0, y1 = self.p0.y, self.p1.y
        if abs(y0 - y1) < EPS:
            return []
        t = (y - y0) / (y1 - y0)
        if t < -EPS or t >= 1 - EPS:
            return []
        return [(1 - t) * self.p0.x + t * self.p1.x]


class QuadraticBezier:
    """Quadratic Bezier: P(t) = (1-t)^2 p0 + 2 t (1-t) p1 + t^2 p2."""

    def __init__(self, p0: Pos, p1: Pos, p2: Pos):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def getpos(self, t: float) -> Pos:
        s = 1 - t
        return Pos(s * s * self.p0.x + 2 * t * s * self.p1.x + t * t * self.p2.x,
                   s * s * self.p0.y + 2 * t * s * self.p1.y + t * t * self.p2.y)

    def length(self) -> float:
        # P(t) = A t^2 + B t + p0; |P'(t)|^2 = a t^2 + b t + c
        ax = self.p0.x - 2 * self.p1.x + self.p2.x
        ay = self.p0.y - 2 * self.p1.y + self.p2.y
        bx = self.p1.x - self.p0.x
        by = self.p1.y - self.p0.y
        a = ax * ax + ay * ay
        b = 2 * (ax * bx + ay * by)
        c = bx * bx + by * by
        if a < EPS:
            return StraightLine(self.p0, self.p2).length()

        def F(t: float) -> float:
            inner = t * (a * t + b) + c
            inner = max(inner, 0.0)
            term1 = (2 * a * t + b) * math.sqrt(inner) / (2 * a)
            denom = 2 * math.sqrt(a) * math.sqrt(inner) + 2 * a * t + b
            term2 = (b * b - 4 * a * c) * math.log(max(denom, EPS)) / (4 * (math.sqrt(a) ** 3))
            return term1 - term2

        return F(1.0) - F(0.0)

    def minsqrdist(self, p: Pos) -> float:
        # Minimize |P(t) - p|^2; derivative is cubic in t.
        x1, y1 = self.p0.x, self.p0.y
        x2, y2 = self.p1.x, self.p1.y
        x3, y3 = self.p2.x, self.p2.y
        ax = x1 - 2 * x2 + x3
        ay = y1 - 2 * y2 + y3
        if ax * ax + ay * ay < EPS:
            return StraightLine(self.p0, self.p2).minsqrdist(p)
        bx = 2 * (x2 - x1)
        by = 2 * (y2 - y1)
        cx = x1 - p.x
        cy = y1 - p.y
        a = 2 * (ax * ax + ay * ay)
        b = 3 * (ax * bx + ay * by)
        c = bx * bx + by * by + 2 * (ax * cx + ay * cy)
        d = bx * cx + by * cy
        candidates = [0.0, 1.0]
        for t in _solve_cubic(a, b, c, d):
            if 0.0 < t < 1.0:
                candidates.append(t)
        best = float('inf')
        for t in candidates:
            q = self.getpos(t)
            sd = (p.x - q.x) ** 2 + (p.y - q.y) ** 2
            if sd < best:
                best = sd
        return best

    def get_ys_from_x(self, x: float) -> List[float]:
        ax = self.p0.x - 2 * self.p1.x + self.p2.x
        bx = 2 * (self.p1.x - self.p0.x)
        cx = self.p0.x - x
        ys: List[float] = []
        for t in _solve_quadratic(ax, bx, cx):
            if -EPS <= t < 1 - EPS:
                s = 1 - t
                ys.append(s * s * self.p0.y + 2 * t * s * self.p1.y + t * t * self.p2.y)
        return ys

    def get_xs_from_y(self, y: float) -> List[float]:
        ay = self.p0.y - 2 * self.p1.y + self.p2.y
        by = 2 * (self.p1.y - self.p0.y)
        cy = self.p0.y - y
        xs: List[float] = []
        for t in _solve_quadratic(ay, by, cy):
            if -EPS <= t < 1 - EPS:
                s = 1 - t
                xs.append(s * s * self.p0.x + 2 * t * s * self.p1.x + t * t * self.p2.x)
        return xs
