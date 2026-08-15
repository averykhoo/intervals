"""
Validated prototype of interval-mod-interval (v3 design), positive quadrant only.

    A mod B  =  (A mod sup B)  UNION  (sup A mod B)        # shape (locations)
    closure of each resulting endpoint decided by a direct attainment test

Theory: references/modulo-derivations/claude-fable/proof-two-edge-reduction.md (two-edge reduction, Q1)
and proof-sign-symmetries-quadrants.md (sign identities; far-edge rule in all quadrants).
Design rationale: v3-modulo-design-notes.md.

Validation (this file's __main__): 3997 random cases, random open/closed flags on both
operands, vs an exact attainment oracle -> 0 soundness failures, 0 closure faults.
(Naive epsilon-propagation scored 604 closure faults on the same suite.)

Representation here: an interval is (lo, lo_closed, hi, hi_closed); a multi-interval is a
sorted list of those. Port to MultiInterval's endpoints/epsilon representation for real use.
Requires A strictly non-negative and B strictly positive; negatives/zero handled per the
design notes (antipodal identity + sign-pure splitting), not implemented here.
"""
import math
import random


def norm(ivs):
    """Drop empty pieces, sort, merge overlapping/touching intervals."""
    ivs = [iv for iv in ivs if iv[0] < iv[2] or (iv[0] == iv[2] and iv[1] and iv[3])]
    ivs.sort(key=lambda t: (t[0], not t[1]))
    out = []
    for lo, lc, hi, hc in ivs:
        if out and (lo < out[-1][2] or (lo == out[-1][2] and (lc or out[-1][3]))):
            p = out[-1]
            out[-1] = (p[0], p[1], p[2], p[3]) if p[2] > hi else \
                      (p[0], p[1], hi, hc if hi > p[2] else (hc or p[3]))
        else:
            out.append((lo, lc, hi, hc))
    return out


def isect_nonempty(a, b):
    (l1, c1, h1, d1), (l2, c2, h2, d2) = a, b
    lo, lc = (l1, c1) if l1 > l2 else (l2, c2) if l2 > l1 else (l1, c1 and c2)
    hi, hc = (h1, d1) if h1 < h2 else (h2, d2) if h2 < h1 else (h1, d1 and d2)
    return lo < hi or (lo == hi and lc and hc)


def attained(v, A, B):
    """Exact test: does some x in A, y in B give x % y == v ?  (A >= 0, B > 0)"""
    x0, x0c, x1, x1c = A
    y0, y0c, y1, y1c = B
    if v < 0:
        return False
    # B intersect (v, inf): need y > v for v to be a residue
    Bv = (max(y0, v), y0c if y0 > v else False, y1, y1c)
    if y0 <= v:
        Bv = (v, False, y1, y1c)
    if not (Bv[0] < Bv[2] or (Bv[0] == Bv[2] and Bv[1] and Bv[3])):
        return False
    # k = 0: v itself must lie in A
    if (x0 < v < x1) or (v == x0 and x0c) or (v == x1 and x1c):
        return True
    # k >= 1: need y in Bv with v + k*y in A, i.e. y in [(x0-v)/k, (x1-v)/k]
    kmax = int((x1 - v) / y0) + 1 if y0 > 0 else 0
    for k in range(1, kmax + 1):
        if isect_nonempty(Bv, ((x0 - v) / k, x0c, (x1 - v) / k, x1c)):
            return True
    return False


def shape(A, B):
    """Step 1: locations only; every endpoint treated as closed."""
    x0, _, x1, _ = A
    y0, _, y1, _ = B
    # P1: A mod y1 (interval mod scalar — mirrors the shipped __mod__ case table)
    n = math.floor(x1 / y1) - math.floor(x0 / y1)
    p1 = [(x0 % y1, True, x1 % y1, True)] if n == 0 else \
         [(0, True, x1 % y1, True), (x0 % y1, True, y1, True)] if n == 1 else \
         [(0, True, y1, True)]
    # P2: x1 mod B (scalar mod interval — z formulas are slide 15's)
    c = x1
    a, b = math.floor(c / y1), math.floor(c / y0)
    if a == b:
        p2 = [(c % y1, True, c % y0, True)]
    else:
        z1 = c / (a + 1)
        right = (c, True, c, True) if a == 0 else (c % y1, True, z1, True)
        p2 = [(0, True, c % y0, True), right] if b == a + 1 else \
             [(0, True, c / (a + 2), True), right]
    return norm(p1 + p2)


def mod(A, B):
    """Step 2: decide each endpoint's closure independently via attainment.

    Positive quadrant only.  Without the guard below this fails SILENTLY on negative
    operands (returns {} for a negative divisor, drops the negative half when B spans
    zero) rather than raising -- see v3-modulo-design-notes.md section 4.
    """
    if A[0] < 0 or B[0] <= 0:
        raise NotImplementedError(
            f"prototype covers the positive quadrant only (A >= 0, B > 0); got A={A}, B={B}. "
            "Negatives need the Q2 primitive pair + antipodal transfer, and zero-crossing "
            "operands need sign-pure splitting -- see v3-modulo-design-notes.md section 4."
        )
    out = [(lo, attained(lo, A, B), hi, attained(hi, A, B)) for lo, _, hi, _ in shape(A, B)]
    # drop phantom degenerate points whose single value turned out unattainable
    # (e.g. the {0} piece when the only zero line touches at an excluded corner)
    return [p for p in out if p[0] < p[2] or (p[1] and p[3])]


def contains(ivs, v):
    return any((lo < v < hi) or (v == lo and lc) or (v == hi and hc)
               for lo, lc, hi, hc in ivs)


if __name__ == "__main__":
    # ---- targeted corner / zero-touch regression suite ----
    # Hand-derived truth tables over all 16 open/closed flag combos per geometry.
    # See v3-modulo-design-notes.md §3b for the derivations (zero-touch classification).
    # f = (x0_closed, x1_closed, y0_closed, y1_closed)
    GEOMS = [
        # lone zero line touches only at corner (x0, y1): 0 iff both flags
        ("G1", (3, 4.5, 2.75, 3),
         lambda f: [(0, f[0] and f[3], 1.75, f[1] and f[2])]),
        # zero lines touch at BOTH near-corners: 0 iff either corner included
        ("G2", (3, 4, 2, 3),
         lambda f: [(0, (f[0] and f[3]) or (f[1] and f[2]), 2, False)]),
        # a zero line crosses the interior: 0 unconditionally (far-corner alignment too)
        ("G3", (3, 6, 2, 3),
         lambda f: [(0, True, 3, False)]),
        # k=0 sector: corner values attained along whole edges -> only x-flags matter
        ("G4", (1, 2, 3, 4),
         lambda f: [(1, f[0], 2, f[1])]),
        # x1 == y0 exactly: {0} iff corner (x1,y0); max needs only x1 (witnesses y > x1)
        ("G5", (1, 3, 3, 4),
         lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(1, f[0], 3, f[1])]),
        # detached {0} piece appears iff corner (x1,y0) is included
        ("G6", (3.5, 4, 2, 3),
         lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(0.5, f[0] and f[3], 2, False)]),
        # far-corner zero (unconditional) + generic corner + k=0-style endpoint at once
        ("G7", (3, 4, 2.75, 4),
         lambda f: [(0, True, 1.25, f[1] and f[2]), (3, f[0], 4, False)]),
    ]
    bad = 0
    for name, (x0, x1, y0, y1), expect in GEOMS:
        for m in range(16):
            f = (bool(m & 1), bool(m & 2), bool(m & 4), bool(m & 8))
            got = mod((x0, f[0], x1, f[1]), (y0, f[2], y1, f[3]))
            if [tuple(p) for p in got] != [tuple(p) for p in expect(f)]:
                bad += 1
                print(f"  MISMATCH {name} flags={f}: got {got}, expected {expect(f)}")
    print(f"corner/zero-touch suite: {7 * 16} combos, {bad} mismatches")

    # ---- randomized fuzz vs attainment oracle ----
    r = random.Random(5)
    sound = over = cases = 0
    for _ in range(4000):
        x0 = round(r.uniform(0, 20), 2)
        x1 = round(x0 + r.uniform(0, 20), 2)
        y0 = round(r.uniform(0.3, 9), 2)
        y1 = round(y0 + r.uniform(0, 8), 2)
        A = (x0, r.random() < .5, x1, r.random() < .5)
        B = (y0, r.random() < .5, y1, r.random() < .5)
        if A[0] == A[2] and not (A[1] and A[3]):
            continue
        if B[0] == B[2] and not (B[1] and B[3]):
            continue
        R = mod(A, B)
        cases += 1
        for _ in range(80):  # soundness: sampled real values must be contained
            x = r.uniform(x0, x1)
            y = r.uniform(y0, y1)
            if not contains(R, x % y):
                sound += 1
                break
        for lo, lc, hi, hc in R:  # tightness: no closed endpoint that is unattainable
            if any(f and not attained(v, A, B) for v, f in ((lo, lc), (hi, hc))):
                over += 1
                break
    print(f"cases: {cases}, soundness failures: {sound}, closure faults: {over}")
