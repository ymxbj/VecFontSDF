"""Glyph parsing from SVG + signed distance queries.

Each glyph is parsed from an SVG `<path d="...">` made of `M`, `L`, and `Q`
commands (move, line, quadratic Bezier). The glyph holds one or more
contours; each contour is a closed polyline of `StraightLine` and
`QuadraticBezier` segments.

The signed distance to a query point is the unsigned distance to the
nearest segment multiplied by a sign returned by a four-direction ray test
on the contours.
"""

import math
from typing import List, Optional

from geometry import EPS, Pos, QuadraticBezier, StraightLine


class Contour:
    """A closed contour made of straight and quadratic-Bezier segments."""

    def __init__(self) -> None:
        self.segments: List = []

    def add_line(self, p0: Pos, p1: Pos) -> None:
        self.segments.append(StraightLine(p0, p1))

    def add_quadratic(self, p0: Pos, p1: Pos, p2: Pos) -> None:
        self.segments.append(QuadraticBezier(p0, p1, p2))

    def cumulative_lengths(self) -> List[float]:
        out: List[float] = []
        total = 0.0
        for seg in self.segments:
            total += seg.length()
            out.append(total)
        return out

    def sample(self, n: int) -> List[Pos]:
        """Sample n points uniformly by arc length along the contour."""
        if n <= 0 or not self.segments:
            return []
        lens = self.cumulative_lengths()
        total = lens[-1]
        if total < EPS:
            return [self.segments[0].getpos(0.0)] * n
        out: List[Pos] = []
        step = 1.0 / n
        seg_idx = 0
        for i in range(n):
            target = (i + 0.5) * step * total
            while seg_idx < len(lens) - 1 and lens[seg_idx] < target:
                seg_idx += 1
            prev = lens[seg_idx - 1] if seg_idx > 0 else 0.0
            local_len = lens[seg_idx] - prev
            t = (target - prev) / local_len if local_len > EPS else 0.0
            out.append(self.segments[seg_idx].getpos(t))
        return out

    def min_sqr_dist(self, p: Pos) -> float:
        return min(seg.minsqrdist(p) for seg in self.segments)

    def _ray_count(self, p: Pos):
        """Count contour intersections along horizontal and vertical rays.

        Returns (left, right, down, up) where each is the number of
        intersections strictly to that side of p. If any intersection lies
        exactly on p the result is None (caller should treat as on-boundary).
        """
        left = right = down = up = 0
        for seg in self.segments:
            for y in seg.get_ys_from_x(p.x):
                if abs(y - p.y) < EPS:
                    return None
                if y < p.y:
                    down += 1
                else:
                    up += 1
            for x in seg.get_xs_from_y(p.y):
                if abs(x - p.x) < EPS:
                    return None
                if x < p.x:
                    left += 1
                else:
                    right += 1
        return left, right, down, up

    def contains(self, p: Pos) -> int:
        """+1 if p is strictly inside the contour, -1 if outside, 0 on boundary.

        Uses parity on both horizontal and vertical rays so a single
        degenerate ray (one that grazes a tangent or hits a control point)
        does not corrupt the answer.
        """
        counts = self._ray_count(p)
        if counts is None:
            return 0
        left, right, down, up = counts
        h_parity = (left + right) % 2  # ideally 0
        v_parity = (down + up) % 2     # ideally 0
        if h_parity == 0 and v_parity == 0:
            inside = (left % 2 == 1) and (down % 2 == 1)
            return 1 if inside else -1
        if h_parity == 0:
            return 1 if left % 2 == 1 else -1
        if v_parity == 0:
            return 1 if down % 2 == 1 else -1
        return 0


class Glyph:
    """A glyph = list of contours parsed from a single SVG path."""

    def __init__(self) -> None:
        self.contours: List[Contour] = []

    @classmethod
    def from_svg_file(cls, path: str) -> 'Glyph':
        with open(path) as f:
            return cls.from_svg_text(f.read())

    @classmethod
    def from_svg_text(cls, text: str) -> 'Glyph':
        glyph = cls()
        for chunk in text.split('<path d="')[1:]:
            inner, _ = chunk.split('"', maxsplit=1)
            glyph._parse_path(inner)
        return glyph

    def _parse_path(self, d: str) -> None:
        tokens = [t for t in d.strip().split(' ') if t != '']
        i = 0
        cur: Optional[Pos] = None
        contour: Optional[Contour] = None
        start: Optional[Pos] = None
        while i < len(tokens):
            cmd = tokens[i]
            if cmd == 'M':
                if contour is not None and contour.segments:
                    self.contours.append(contour)
                contour = Contour()
                x = float(tokens[i + 1])
                y = float(tokens[i + 2])
                cur = Pos(x, y)
                start = cur
                i += 3
            elif cmd == 'L':
                x = float(tokens[i + 1])
                y = float(tokens[i + 2])
                nxt = Pos(x, y)
                contour.add_line(cur, nxt)
                cur = nxt
                i += 3
            elif cmd == 'Q':
                x1 = float(tokens[i + 1])
                y1 = float(tokens[i + 2])
                x2 = float(tokens[i + 3])
                y2 = float(tokens[i + 4])
                ctrl = Pos(x1, y1)
                nxt = Pos(x2, y2)
                contour.add_quadratic(cur, ctrl, nxt)
                cur = nxt
                i += 5
            elif cmd in ('Z', 'z'):
                if cur is not None and start is not None:
                    contour.add_line(cur, start)
                    cur = start
                i += 1
            else:
                raise ValueError(f'unsupported SVG path command: {cmd!r}')
        if contour is not None and contour.segments:
            # Close the contour with a straight segment if the path was not
            # explicitly closed with Z.
            if start is not None and cur is not None and (
                    abs(start.x - cur.x) > EPS or abs(start.y - cur.y) > EPS):
                contour.add_line(cur, start)
            self.contours.append(contour)

    def signed_distance(self, p: Pos) -> float:
        """Signed distance to the glyph outline. Positive = outside, negative = inside."""
        unsigned = math.sqrt(min(c.min_sqr_dist(p) for c in self.contours))
        # Multiply contour signs: a point that is "inside" an even number
        # of contours is outside (counting holes correctly).
        sign = 1
        for c in self.contours:
            s = c.contains(p)
            if s == 0:
                return 0.0
            sign *= -s  # contains returns +1 for inside; flip so inside -> negative SDF
        return sign * unsigned
