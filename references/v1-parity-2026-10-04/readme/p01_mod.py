# README: "__mod__ is only partly implemented, __rmod__ is not yet implemented"
# differential: v1 % (in v1's supported region) vs v2 %, with an exact brute-force membership oracle
from common import *
import itertools
warnings.simplefilter('ignore')

def pieces_of(s):
    return [(Fraction(p.inf), p.inf_closed, Fraction(p.sup), p.sup_closed) for p in s.pieces]

def nonempty(a, b):
    """two pieces (lo,lc,hi,hc) intersect?"""
    lo, lc = max((a[0], not a[1]), (b[0], not b[1]))  # larger lower end; open wins at ties
    hi, hc = min((a[2], a[3]), (b[2], b[3]))         # smaller upper end; open (False) wins at ties
    lc = not lc
    return lo < hi or (lo == hi and lc and hc)

def in_mod(y, X, D):
    """exact: y in {x mod d : x in X, d in D}, X >= 0 finite, D > 0 (python floor mod)"""
    y = Fraction(y)
    if y < 0: return False
    xs, ds = pieces_of(X), pieces_of(D)
    dpos = [(d[0], d[1], d[2], d[3]) for d in ds]
    # k = 0: y in X and some d > y
    gt = (y, False, Fraction(10**9), True)
    if any(nonempty(x, (y, True, y, True)) for x in xs) and any(nonempty(d, gt) for d in dpos):
        return True
    supx = max(x[2] for x in xs)
    infd = min(d[0] for d in dpos)
    kmax = int(supx / infd) + 2 if infd > 0 else 10**4
    for k in range(1, kmax + 1):
        for x in xs:
            cand = ((x[0] - y) / k, x[1], (x[2] - y) / k, x[3])
            if cand[0] > cand[2]: continue
            for d in dpos:
                if nonempty(cand, d) and nonempty(cand, gt) and nonempty(d, gt):
                    # need a single d in all three
                    a = cand; b = d
                    lo, lc = max((a[0], not a[1]), (b[0], not b[1]), (y, True)); lc = not lc
                    hi, hc = min((a[2], a[3]), (b[2], b[3]))
                    if lo < hi or (lo == hi and lc and hc): return True
    return False

def rand_piece(rng, lo, hi, den=2):
    a, b = sorted(Fraction(rng.randint(lo * den, hi * den), den) for _ in range(2))
    if a == b: return (a, a, True, True)
    return (a, b, rng.random() < .5, rng.random() < .5)

def mk(ps):
    v1x = V1()
    for (a, b, ac, bc) in ps:
        v1x.update(V1(a, b, start_closed=ac, end_closed=bc) if a != b else V1(a))
    return v1x, MI.from_pieces(ps)

rng = random.Random(20261004)
n = agree = v1_wrong = v2_wrong = v1_raise = 0
examples = []
cases = []
# hand cases
hand = [([(0, 1, True, True)], 1), ([(Fraction(1,2), 1, True, True)], 1), ([(Fraction(1,2), 1, True, False)], 1),
        ([(0, 10, True, True)], 3), ([(1, 2, True, True)], [(3, 4, True, True)]),
        ([(3, Fraction(79,10), True, True)], [(Fraction(79,10), Fraction(126,10), True, True)]),
        ([(1, 5, False, False)], [(1, 2, False, True)]), ([(2, 2, True, True)], 2), ([(4, 6, True, False)], 2)]
for X, m in hand: cases.append((X, m))
for _ in range(400):
    X = [rand_piece(rng, 0, 12) for _ in range(rng.randint(1, 2))]
    if rng.random() < .4:
        m = rng.choice([1, 2, 3, Fraction(3, 2), Fraction(5, 2)])
    else:
        m = [rand_piece(rng, 1, 5) for _ in range(rng.randint(1, 2))]
        # v1 needs strictly positive other: keep lo >= 1 (from rand_piece)
    cases.append((X, m))
for X, m in cases:
    a1, a2 = mk(X)
    if isinstance(m, list): m1, m2 = mk(m)
    else: m1, m2 = m, m
    n += 1
    try:
        r1 = a1 % m1
        if r1 is NotImplemented: raise TypeError('NotImplemented')
        r1 = conv(r1)
    except Exception as e:
        v1_raise += 1; examples.append(('v1 raised', str(a2), str(m2), repr(e)[:80])); r1 = None
    r2 = a2 % m2
    D = m2 if isinstance(m2, MI) else MI(m2)
    if r1 is not None and r1 == r2:
        agree += 1
        # spot-check v2 against the oracle anyway
        for y in probes_of(r2, a2, D):
            if math.isfinite(y) and (y in r2) != in_mod(y, a2, D):
                v2_wrong += 1; examples.append(('v2 wrong (agreeing)', str(a2), str(D), y)); break
        continue
    for y in probes_of(r2, a2, D, *( [r1] if r1 is not None else [])):
        if not math.isfinite(y): continue
        truth = in_mod(y, a2, D)
        if r1 is not None and (y in r1) != truth:
            v1_wrong += 1; examples.append(('v1 wrong', str(a2), str(D), str(r1), str(r2), y, truth)); break
    for y in probes_of(r2, a2, D, *( [r1] if r1 is not None else [])):
        if not math.isfinite(y): continue
        if (y in r2) != in_mod(y, a2, D):
            v2_wrong += 1; examples.append(('v2 wrong', str(a2), str(D), str(r2), y)); break
print(f'cases {n}  agree {agree}  v1_raised {v1_raise}  v1_wrong {v1_wrong}  v2_wrong {v2_wrong}')
for e in examples[:12]: print('  ', e)
# sabotage: the oracle must catch a wrong expectation
assert in_mod(Fraction(1, 2), MI(Fraction(1,2), 1), MI(1)) and not in_mod(Fraction(1, 4), MI(Fraction(1,2), 1), MI(1))
assert not in_mod(1, MI(Fraction(1,2), 1), MI(1)) and in_mod(0, MI(Fraction(1,2), 1), MI(1))
# v1 unsupported regions (README: partly implemented) and __rmod__
for desc, f1, f2 in [
    ('neg self % 3', lambda: V1(-5, -1) % 3, lambda: MI(-5, -1) % 3),
    ('self % neg', lambda: V1(1, 5) % -3, lambda: MI(1, 5) % -3),
    ('zero-crossing self % [1,2]', lambda: V1(-1, 5) % V1(1, 2), lambda: MI(-1, 5) % MI(1, 2)),
    ('[0,5] % [1,2] (self touches 0)', lambda: V1(0, 5) % V1(1, 2), lambda: MI(0, 5) % MI(1, 2)),
    ('infinite self % 3', lambda: V1(1, math.inf) % 3, lambda: MI(1, math.inf) % 3),
    ('7 % [2,5] (__rmod__)', lambda: 7 % V1(2, 5), lambda: 7 % MI(2, 5)),
    ('[1,5] % 0', lambda: V1(1, 5) % 0, lambda: MI(1, 5) % 0),
]:
    out = []
    for f in (f1, f2):
        try: out.append(str(f()))
        except Exception as e: out.append(f'{type(e).__name__}: {e}'[:70])
    print(f'{desc:35s} v1: {out[0]:45s} v2: {out[1]}')
