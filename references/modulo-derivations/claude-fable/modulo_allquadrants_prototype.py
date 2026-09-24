"""
All-quadrant prototype of interval-mod-interval (extends modulo_v3_prototype.py).

    mod(A, B)  for single pieces A, B of ANY sign, including zero-crossing and infinite ones.

Theory: proof-all-quadrants.md (this directory). Summary of the pipeline:
  1. canonicalise: finite endpoints -> Fraction; +-inf stays float; a dividend endpoint at
     +-inf is dropped (flag forced open), a divisor piece has 0 removed.
  2. split A at 0 (0 in both halves if 0 in A), B at 0 (0 removed) -> <= 4 sign-pure boxes.
  3. per box: hull pieces from the two far edges (Q1: A mod y1 U x1 mod B; Q2: A mod y1 U
     x0 mod B; Q3/Q4 by negation), UN-merged; then the exact result is
        union of open interiors of the hull pieces  U  {every hull endpoint that is attained}
     with attainment decided by an O(1) test (proof-all-quadrants.md section 2.5).
  4. union over boxes, normalise.
  Step 3's "un-merged" is the fix for the one-point-hole bug of the old prototype
  (proof-all-quadrants.md section 1.2(a)).

Representation: a piece is (lo, lo_closed, hi, hi_closed); a multi-interval is a sorted
list of pieces. Values: int / Fraction / float(+-inf). Run this file to execute the harness.
"""
import math
import random
import sys
from fractions import Fraction

INF = float("inf")
NINF = float("-inf")


# ----------------------------------------------------------------------------- helpers

def is_inf(v):
    return isinstance(v, float) and math.isinf(v)


def canon(v):
    """Exact representation: +-inf stay float, everything else becomes a Fraction."""
    return v if is_inf(v) else Fraction(v)


def canon_piece(p):
    lo, lc, hi, hc = p
    return (canon(lo), bool(lc), canon(hi), bool(hc))


def nonempty(p):
    lo, lc, hi, hc = p
    return lo < hi or (lo == hi and lc and hc)


def in_piece(v, p):
    lo, lc, hi, hc = p
    return (lo < v < hi) or (v == lo and lc) or (v == hi and hc)


def isect(p, q):
    """Exact intersection of two pieces (may be empty)."""
    (l1, c1, h1, d1), (l2, c2, h2, d2) = p, q
    lo, lc = (l1, c1) if l1 > l2 else (l2, c2) if l2 > l1 else (l1, c1 and c2)
    hi, hc = (h1, d1) if h1 < h2 else (h2, d2) if h2 < h1 else (h1, d1 and d2)
    return (lo, lc, hi, hc)


def isect_nonempty(p, q):
    return nonempty(isect(p, q))


def neg_piece(p):
    lo, lc, hi, hc = p
    return (-hi, hc, -lo, lc)


def neg_pieces(ps):
    return sorted((neg_piece(p) for p in ps), key=lambda t: (t[0], not t[1]))


def norm(ivs):
    """Drop empty pieces, sort, merge overlapping or touching (with a closed side) pieces."""
    ivs = [iv for iv in ivs if nonempty(iv)]
    ivs.sort(key=lambda t: (t[0], not t[1]))
    out = []
    for lo, lc, hi, hc in ivs:
        if out and (lo < out[-1][2] or (lo == out[-1][2] and (lc or out[-1][3]))):
            p = out[-1]
            if p[2] > hi:
                continue
            out[-1] = (p[0], p[1], hi, hc if hi > p[2] else (hc or p[3]))
        else:
            out.append((lo, lc, hi, hc))
    return out


def fdiv(a, b):
    """a / b on the extended reals for the cases this file needs (b != 0, never inf/inf)."""
    if is_inf(a):
        assert not is_inf(b)
        return a if b > 0 else -a
    if is_inf(b):
        return Fraction(0)
    return Fraction(a) / Fraction(b)


def ceil_ext(t):
    return t if is_inf(t) else math.ceil(t)


def floor_ext(t):
    return t if is_inf(t) else math.floor(t)


def mod_ext(x, y):
    """Scalar floor-mod on the extended reals, matching Python: x finite, y != 0."""
    assert not is_inf(x) and y != 0
    if is_inf(y):
        if x == 0:
            return Fraction(0)
        return x if (x > 0) == (y > 0) else y
    return Fraction(x) % Fraction(y)


# ----------------------------------------------------------------------------- shapes
# Every function below returns UN-MERGED closed-hull pieces as (lo, hi) pairs.

def p1_hull(x0, x1, m):
    """[x0, x1] mod m for a finite scalar m > 0 and real x0 <= x1 (any sign; x0 may be
    -inf, x1 may be +inf)."""
    if is_inf(x0) or is_inf(x1):
        return [(Fraction(0), m)]
    n = (x1 // m) - (x0 // m)
    if n == 0:
        return [(x0 % m, x1 % m)]
    if n == 1:
        return [(Fraction(0), x1 % m), (x0 % m, m)]
    return [(Fraction(0), m)]


def p2_hull_q1(c, y0, y1):
    """c mod [y0, y1] for finite c >= 0, finite y0 >= 0 (y0 == 0 only as an OPEN endpoint,
    meaning b = floor(c/y0) = +inf), y1 in [y0, +inf]."""
    if c == 0:
        return [(Fraction(0), Fraction(0))]
    a = 0 if is_inf(y1) else c // y1
    b = INF if y0 == 0 else c // y0
    cy1 = c if is_inf(y1) else c % y1
    if a == b:
        return [(cy1, c % y0)]
    right = (c, c) if a == 0 else (cy1, c / (a + 1))
    left = (Fraction(0), c % y0) if b == a + 1 else (Fraction(0), c / (a + 2))
    return [left, right]


def p2_hull_q2(c, y0, y1):
    """c mod [y0, y1] for finite c <= 0, finite y0 >= 0 (y0 == 0 only as an OPEN endpoint,
    meaning m_hi = +inf), y1 in [y0, +inf]."""
    if c == 0:
        return [(Fraction(0), Fraction(0))]
    ac = -c
    m_lo = 1 if is_inf(y1) else math.ceil(ac / y1)
    m_hi = INF if y0 == 0 else math.ceil(ac / y0)
    cy1 = INF if is_inf(y1) else c % y1
    if m_hi == m_lo:
        return [(c % y0, cy1)]
    if m_hi == m_lo + 1:
        return [(Fraction(0), cy1), (c % y0, ac / m_lo)]
    return [(Fraction(0), cy1), (Fraction(0), ac / m_lo)]


def shape_q1(A, B):
    """Hull pieces for a Q1 box: 0 <= x0 <= x1 <= +inf, 0 < y0 <= y1 <= +inf."""
    x0, _, x1, _ = A
    y0, _, y1, _ = B
    if is_inf(y0):                          # B = {+inf}: x % inf = x for x >= 0
        return [(x0, x1)]
    if is_inf(x1):
        return [(Fraction(0), INF if is_inf(y1) else y1)]
    pieces = [(x0, x1)] if is_inf(y1) else p1_hull(x0, x1, y1)
    return pieces + p2_hull_q1(x1, y0, y1)


def shape_q2(A, B):
    """Hull pieces for a Q2 box: -inf <= x0 <= x1 <= 0, 0 < y0 <= y1 <= +inf."""
    x0, _, x1, _ = A
    y0, _, y1, _ = B
    if is_inf(y0):                          # B = {+inf}: x % inf = inf (x < 0), 0 (x = 0)
        return ([(Fraction(0), Fraction(0))] if x1 == 0 else []) + \
               ([(INF, INF)] if x0 < 0 else [])
    if is_inf(x0):
        return [(Fraction(0), INF if is_inf(y1) else y1)]
    if is_inf(y1):
        pieces = [(Fraction(0), Fraction(0))] if x1 == 0 else []
    else:
        pieces = p1_hull(x0, x1, y1)
    return pieces + p2_hull_q2(x0, y0, y1)


# ----------------------------------------------------------------------------- O(1) attainment

def _bv(v, B):
    """B intersect (v, +inf]  (the divisors that can produce residue v >= 0), and the same
    restricted to finite y (needed for k != 0 witnesses)."""
    y0, y0c, y1, y1c = B
    yl, ylc = (y0, y0c) if y0 > v else (v, False)
    Bv = (yl, ylc, y1, y1c)
    Bvf = (yl, ylc, y1, y1c and not is_inf(y1))
    return Bv, Bvf


def attained_q1(v, A, B):
    """O(1) test, Q1 box (x0 >= 0 finite, x1 <= +inf, y0 > 0 finite, y1 <= +inf)."""
    x0, x0c, x1, x1c = A
    if is_inf(v) or v < 0:
        return False
    Bv, Bvf = _bv(v, B)
    if not nonempty(Bv):
        return False
    if in_piece(v, A):                      # k = 0 (also covers y = +inf: x % inf = x)
        return True
    if not nonempty(Bvf):
        return False
    yl = Bvf[0]
    K_lo = max(1, ceil_ext(fdiv(x0 - v, B[2])))
    # yl == 0 only when v == 0 and B is open at 0: arbitrarily small y => unbounded k
    K_hi = (INF if x1 > v else 0) if yl == 0 else floor_ext(fdiv(x1 - v, yl))
    if K_hi < K_lo:
        return False
    if K_hi - K_lo >= 2:                    # an interior k always has a flag-free witness
        return True
    for k in {K_lo, K_hi}:
        I = (fdiv(x0 - v, k), x0c, fdiv(x1 - v, k), x1c)
        if isect_nonempty(I, Bvf):
            return True
    return False


def attained_q2(v, A, B):
    """O(1) test, Q2 box (x0 >= -inf, x1 <= 0 finite, y0 > 0 finite, y1 <= +inf)."""
    x0, x0c, x1, x1c = A
    y0, y0c, y1, y1c = B
    if v == INF:                            # x % inf = inf for x < 0
        return is_inf(y1) and y1c and x0 < 0 and nonempty(A)
    if v == NINF or v < 0:
        return False
    Bv, Bvf = _bv(v, B)
    if not nonempty(Bv):
        return False
    if v == 0 and in_piece(0, A):           # m = 0
        return True
    if not nonempty(Bvf):
        return False
    yl = Bvf[0]
    M_lo = max(1, ceil_ext(fdiv(v - x1, y1)))
    M_hi = (INF if x0 < v else 0) if yl == 0 else floor_ext(fdiv(v - x0, yl))
    if M_hi < M_lo:
        return False
    if M_hi - M_lo >= 2:
        return True
    for m in {M_lo, M_hi}:
        I = (fdiv(v - x1, m), x1c, fdiv(v - x0, m), x0c)
        if isect_nonempty(I, Bvf):
            return True
    return False


def attained_box(v, A, B):
    """O(1) attainment for one sign-pure box (any quadrant)."""
    if B[0] < 0:
        return attained_box(-v, neg_piece(A), neg_piece(B))
    if A[0] >= 0:
        return attained_q1(v, A, B)
    return attained_q2(v, A, B)


# ----------------------------------------------------------------------------- mod

def mod_box(A, B):
    """Exact result for one sign-pure box."""
    if B[0] < 0:
        return neg_pieces(mod_box(neg_piece(A), neg_piece(B)))
    hull = shape_q1(A, B) if A[0] >= 0 else shape_q2(A, B)
    out = []
    ends = set()
    for lo, hi in hull:
        if lo < hi:
            out.append((lo, False, hi, False))
        ends.add(lo)
        ends.add(hi)
    for v in ends:
        if attained_box(v, A, B):
            out.append((v, True, v, True))
    return norm(out)


def split_dividend(A):
    """A intersect [-inf, 0] and A intersect [0, +inf]; 0 is in both iff 0 in A."""
    parts = [isect(A, (NINF, False, Fraction(0), True)),
             isect(A, (Fraction(0), True, INF, False))]
    parts = [p for p in parts if nonempty(p)]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts = parts[:1]
    return parts


def split_divisor(B):
    """B with 0 removed: B intersect [-inf, 0) and B intersect (0, +inf]."""
    parts = [isect(B, (NINF, True, Fraction(0), False)),
             isect(B, (Fraction(0), False, INF, True))]
    return [p for p in parts if nonempty(p)]


def prepare(A, B):
    """Canonicalise operands and apply policy D8: infinite dividend endpoints are dropped."""
    A = canon_piece(A)
    B = canon_piece(B)
    x0, x0c, x1, x1c = A
    A = (x0, x0c and not is_inf(x0), x1, x1c and not is_inf(x1))
    return A, B


def mod(A, B):
    """A mod B for single pieces of any sign; exact Fraction arithmetic."""
    A, B = prepare(A, B)
    if not nonempty(A):
        return []
    out = []
    for a in split_dividend(A):
        for b in split_divisor(B):
            out += mod_box(a, b)
    return norm(out)


def attained(v, A, B):
    """O(1) attainment against full (prepared) operands = OR over the sign-pure boxes."""
    A, B = prepare(A, B)
    v = canon(v)
    if not nonempty(A):
        return False
    return any(attained_box(v, a, b)
               for a in split_dividend(A) for b in split_divisor(B))


def contains(ivs, v):
    return any(in_piece(v, p) for p in ivs)


# ----------------------------------------------------------------------------- brute oracle

def attained_brute(v, A, B, K=None):
    """Independent oracle: does some x in A, y in B give x mod y == v?

    Loops over every integer quotient k in [-K, K] and, for each, intersects the y-interval
    {y : v + k*y in A} with B and with the residue condition (0 <= v < y or y < v <= 0).
    Uses no quadrant reasoning and no k-range derivation. Closed infinite divisor
    endpoints are handled by the scalar rule x % (+-inf). K defaults to an exhaustive bound
    for finite operands; infinite operands use K = 64 (ample for the harness grids)."""
    A, B = prepare(A, B)
    v = canon(v)
    if not nonempty(A):
        return False
    x0, x0c, x1, x1c = A
    y0, y0c, y1, y1c = B
    # y = +-inf as a divisor point
    for yinf, flag in ((y1, y1c), (y0, y0c)):
        if is_inf(yinf) and flag and in_piece(yinf, B):
            # residues: x for sign(x) == sign(yinf) or x == 0; yinf for the other sign
            if v == yinf:
                if (yinf > 0 and x0 < 0) or (yinf < 0 and x1 > 0):
                    return True
            elif not is_inf(v) and in_piece(v, A) and (v == 0 or (v > 0) == (yinf > 0)):
                return True
    if is_inf(v):
        return False
    Bfin = (y0, y0c and not is_inf(y0), y1, y1c and not is_inf(y1))
    if v > 0:
        Y = (v, False, INF, False)
    elif v < 0:
        Y = (NINF, False, v, False)
    else:
        Y = (NINF, False, INF, False)
    Ydom = isect(Bfin, Y)
    if v == 0:  # exclude y == 0 (already excluded from B but be safe)
        pass
    if not nonempty(Ydom):
        return False
    if K is None:
        # Elementary bound: a witness has |k| = |x - v| / |y| <= (|x| + |v|) / |y|.  With
        # every divisor part bounded away from 0 (near end y_near != 0) and A finite this
        # is exhaustive.  Otherwise (B reaches 0, or A unbounded) use the far ends plus a
        # margin -- generous for the grids and fuzz ranges used in __main__.
        parts = split_divisor(B)
        xs = [abs(t) for t in (x0, x1) if not is_inf(t)]
        xmax = max(xs) if xs else Fraction(0)
        near = [abs(p[0]) if p[2] > 0 else abs(p[2]) for p in parts]   # positive part iff hi > 0
        far = [abs(p[2]) if p[2] > 0 else abs(p[0]) for p in parts]
        far_fin = [t for t in far if not is_inf(t)]
        if all(t > 0 for t in near) and not (is_inf(x0) or is_inf(x1)):
            K = math.floor((xmax + abs(v)) / min(near)) + 2
        else:
            K = math.floor((xmax + abs(v)) / min(far_fin)) + 8 if far_fin else 8
        K = min(K, 2000)
    for k in range(-K, K + 1):
        if k == 0:
            if in_piece(v, A):
                return True
            continue
        lo, lc = fdiv(x0 - v, k), x0c
        hi, hc = fdiv(x1 - v, k), x1c
        if k < 0:
            lo, lc, hi, hc = hi, hc, lo, lc
        if isect_nonempty((lo, lc, hi, hc), Ydom):
            return True
    return False


# ----------------------------------------------------------------------------- harness

def _fin_proxy(lo, hi):
    flo = lo if not is_inf(lo) else (hi - 7 if not is_inf(hi) else Fraction(-7))
    fhi = hi if not is_inf(hi) else (lo + 7 if not is_inf(lo) else Fraction(7))
    return flo, fhi


def sample_point(p, rng):
    """A random point of piece p (Fraction), sometimes an endpoint if closed (incl. a closed
    +-inf divisor endpoint)."""
    lo, lc, hi, hc = p
    flo, fhi = _fin_proxy(lo, hi)
    if lo == hi:
        return lo
    r = rng.random()
    if r < 0.15 and lc:
        return lo
    if r < 0.30 and hc:
        return hi
    return flo + (fhi - flo) * Fraction(rng.randint(1, 63), 64)


def sample_interior(p, rng):
    lo, lc, hi, hc = p
    if lo == hi:
        return lo
    flo, fhi = _fin_proxy(lo, hi)
    return flo + (fhi - flo) * Fraction(rng.randint(1, 63), 64)


def lattice_points(p, step, reach=4):
    """Points of piece p on the lattice step*Z (plus closed endpoints); infinite ends are
    replaced by the finite end +- reach (or +-reach if both are infinite). A closed
    infinite endpoint is included as itself."""
    lo, lc, hi, hc = p
    flo, fhi = lo, hi
    if is_inf(lo):
        flo = (hi - reach) if not is_inf(hi) else -reach
    if is_inf(hi):
        fhi = (lo + reach) if not is_inf(lo) else reach
    pts = set()
    t = math.ceil(flo / step)
    while t * step <= fhi:
        v = Fraction(t) * step
        if in_piece(v, p):
            pts.add(v)
        t += 1
    for v, f in ((lo, lc), (hi, hc)):
        if f:
            pts.add(v)
    return pts


def lattice_soundness(A, B, R, step=Fraction(1, 2)):
    """Deterministic: every lattice (x, y) in A x B must map into R. Returns a failing
    (x, y, x mod y) or None."""
    Ap, Bp = prepare(A, B)
    if not nonempty(Ap):
        return None
    xs = lattice_points(Ap, step)
    for b in split_divisor(Bp):
        for y in lattice_points(b, step):
            for x in xs:
                r = mod_ext(x, y)
                if not contains(R, r):
                    return (x, y, r)
    return None


def check_pair(A, B, rng, n_samples=10, stats=None, lattice=None):
    """All checks for one operand pair; increments stats[...] on failure."""
    R = mod(A, B)
    Ap, Bp = prepare(A, B)
    fails = []
    # (0) deterministic lattice soundness
    if lattice is not None:
        bad = lattice_soundness(A, B, R, lattice)
        if bad is not None:
            fails.append(("lattice",) + bad)
    # (1) soundness: sampled operand points map into R
    if nonempty(Ap):
        for b in split_divisor(Bp):
            for _ in range(n_samples):
                x = sample_point(Ap, rng)
                y = sample_point(b, rng)
                r = mod_ext(x, y)
                if not contains(R, r):
                    fails.append(("sound", x, y, r))
                    break
    # (2) closure: closed endpoint <=> attained (brute), plus O(1) == brute
    for lo, lc, hi, hc in R:
        for v, flag in ((lo, lc), (hi, hc)):
            br = attained_brute(v, A, B)
            if br != flag:
                fails.append(("closure", v, flag, br))
            if attained(v, A, B) != br:
                fails.append(("o1_vs_brute", v))
    # (3) sharpness: sampled interior result points are attained
    for p in R:
        for _ in range(2):
            v = sample_interior(p, rng)
            if not attained_brute(v, A, B):
                fails.append(("sharp", v))
                break
    # (4) gaps and outside: not attained
    probes = []
    for p, q in zip(R, R[1:]):
        if p[2] < q[0]:
            # a finite point strictly inside the gap (a neighbour may be the point {+-inf})
            probes.append(p[2] + 1 if is_inf(q[0]) else q[0] - 1 if is_inf(p[2])
                          else (p[2] + q[0]) / 2)
    if R:
        if not is_inf(R[0][0]):
            probes.append(R[0][0] - 1)
        if not is_inf(R[-1][2]):
            probes.append(R[-1][2] + 1)
    for v in probes:
        if attained_brute(v, A, B):
            fails.append(("gap", v))
        if attained(v, A, B):
            fails.append(("o1_vs_brute", v))
    if stats is not None:
        for f in fails:
            stats[f[0]] = stats.get(f[0], 0) + 1
    return R, fails


def old_q1_suite():
    """The 112-case Q1 corner / zero-touch suite from modulo_v3_prototype.py, verbatim."""
    GEOMS = [
        ("G1", (3, 4.5, 2.75, 3),
         lambda f: [(0, f[0] and f[3], 1.75, f[1] and f[2])]),
        ("G2", (3, 4, 2, 3),
         lambda f: [(0, (f[0] and f[3]) or (f[1] and f[2]), 2, False)]),
        ("G3", (3, 6, 2, 3),
         lambda f: [(0, True, 3, False)]),
        ("G4", (1, 2, 3, 4),
         lambda f: [(1, f[0], 2, f[1])]),
        ("G5", (1, 3, 3, 4),
         lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(1, f[0], 3, f[1])]),
        ("G6", (3.5, 4, 2, 3),
         lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(0.5, f[0] and f[3], 2, False)]),
        ("G7", (3, 4, 2.75, 4),
         lambda f: [(0, True, 1.25, f[1] and f[2]), (3, f[0], 4, False)]),
    ]
    bad = total = 0
    for name, (x0, x1, y0, y1), expect in GEOMS:
        for m in range(16):
            f = (bool(m & 1), bool(m & 2), bool(m & 4), bool(m & 8))
            got = mod((x0, f[0], x1, f[1]), (y0, f[2], y1, f[3]))
            exp = [canon_piece(p) for p in expect(f)]
            total += 1
            if got != exp:
                bad += 1
                print(f"  MISMATCH {name} flags={f}: got {got}, expected {exp}")
    return total, bad


def grid_pieces(values, dividend):
    """Every piece lo <= hi over the grid with all flag combos, canonicalised, deduplicated
    (a dividend's infinite endpoints are dropped so their flags do not matter)."""
    seen = set()
    out = []
    for i, lo in enumerate(values):
        for hi in values[i:]:
            for lc in (False, True):
                for hc in (False, True):
                    p = canon_piece((lo, lc, hi, hc))
                    if dividend:
                        p, _ = prepare(p, (1, True, 1, True))
                    if not nonempty(p) or p in seen:
                        continue
                    seen.add(p)
                    out.append(p)
    return out


def fmt(ivs):
    def s(v):
        return "inf" if v == INF else "-inf" if v == NINF else str(v)
    return " U ".join(("[" if lc else "(") + s(lo) + ", " + s(hi) + ("]" if hc else ")")
                      for lo, lc, hi, hc in ivs) or "{}"


def random_piece(rng, allow_inf=True):
    def val():
        if allow_inf and rng.random() < 0.08:
            return rng.choice([NINF, INF])
        return Fraction(rng.randint(-24, 24), rng.randint(1, 4))
    a, b = val(), val()
    if a > b:
        a, b = b, a
    return (a, rng.random() < 0.5, b, rng.random() < 0.5)


if __name__ == "__main__":
    # (a) the old 112-case Q1 suite
    total, bad = old_q1_suite()
    print(f"(a) old Q1 corner/zero-touch suite: {total} combos, {bad} mismatches")

    # regression for the merged-hull hole (section 1.2(a) of proof-all-quadrants.md)
    hole_cases = [
        (((2, False, Fraction(5, 2), False), (1, False, Fraction(3, 2), False)),
         "(0, 0.5) U (0.5, 1.25) plus 0"),
        (((Fraction(5, 2), False, Fraction(7, 2), False), (1, True, 1, True)), None),
        (((Fraction(5, 2), True, Fraction(5, 2), True), (1, False, 2, False)), None),
    ]
    for (A, B), _ in hole_cases:
        print(f"    hole regression: {fmt([A])} mod {fmt([B])} = {fmt(mod(A, B))}")

    # (b) exhaustive grid
    GRID = [NINF, -3, -2, Fraction(-3, 2), -1, Fraction(-1, 2), 0,
            Fraction(1, 2), 1, Fraction(3, 2), 2, 3, INF]
    As = grid_pieces(GRID, dividend=True)
    Bs = grid_pieces(GRID, dividend=False)
    rng = random.Random(2026)
    stats = {}
    pairs = 0
    shown = 0
    quick = "--quick" in sys.argv
    for A in As:
        for B in Bs:
            if quick and rng.random() > 0.05:
                continue
            pairs += 1
            R, fails = check_pair(A, B, rng, n_samples=6, stats=stats, lattice=Fraction(1, 2))
            if fails and shown < 15:
                shown += 1
                print(f"    FAIL {fmt([A])} mod {fmt([B])} = {fmt(R)}: {fails[:3]}")
    print(f"(b) grid: {len(As)} dividend pieces x {len(Bs)} divisor pieces = {pairs} pairs; "
          f"failures: lattice-soundness={stats.get('lattice', 0)}, "
          f"sampled-soundness={stats.get('sound', 0)}, closure={stats.get('closure', 0)}, "
          f"sharpness={stats.get('sharp', 0)}, gap={stats.get('gap', 0)}, "
          f"o1_vs_brute={stats.get('o1_vs_brute', 0)}")

    # (c) random fuzz, all quadrants, Fraction operands (some infinite endpoints)
    rng = random.Random(7)
    stats = {}
    n = 0
    shown = 0
    N = 600 if quick else 6000
    while n < N:
        A = random_piece(rng)
        B = random_piece(rng)
        Ap, Bp = prepare(A, B)
        if not nonempty(Ap) or not split_divisor(Bp):
            continue
        n += 1
        R, fails = check_pair(A, B, rng, n_samples=12, stats=stats, lattice=Fraction(1, 4))
        if fails and shown < 15:
            shown += 1
            print(f"    FAIL {fmt([A])} mod {fmt([B])} = {fmt(R)}: {fails[:3]}")
    print(f"(c) fuzz: {n} cases; failures: lattice-soundness={stats.get('lattice', 0)}, "
          f"sampled-soundness={stats.get('sound', 0)}, closure={stats.get('closure', 0)}, "
          f"sharpness={stats.get('sharp', 0)}, gap={stats.get('gap', 0)}, "
          f"o1_vs_brute={stats.get('o1_vs_brute', 0)}")

    # worked examples from proof-all-quadrants.md section 2.6
    ex = [((0, True, INF, False), (2, True, 3, True)),
          ((2, True, 3, True), (1, True, INF, False)),
          ((2, True, 3, True), (1, True, INF, True)),
          ((NINF, False, -1, True), (2, True, INF, True)),
          ((NINF, False, -1, True), (2, True, INF, False)),
          ((-3, True, -2, True), (1, True, INF, True)),
          ((-3, True, -2, True), (INF, True, INF, True)),
          ((2, True, 3, True), (INF, True, INF, True)),
          ((1, True, INF, True), (2, True, 3, True)),
          ((INF, True, INF, True), (2, True, 3, True)),
          ((1, True, INF, False), (-3, True, -2, True)),
          ((-6, True, -3, True), (4, True, 5, True)),
          ((-7, True, -3, True), (2, True, 5, True)),
          ((3, True, 7, True), (-5, True, -2, True)),
          ((3, True, 7, True), (-2, True, 5, True)),
          ((-3, True, 7, True), (2, True, 5, True)),
          ((-3, True, 3, True), (-2, True, 2, True))]
    print("worked examples:")
    for A, B in ex:
        print(f"    {fmt([canon_piece(A)])} mod {fmt([canon_piece(B)])} = {fmt(mod(A, B))}")
