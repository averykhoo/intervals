"""seeded sweep: multi-piece bases mixing [0, b] / (0, b] / [0] pieces (and sometimes a negative or
sign-straddling piece) raised to interval exponents holding negatives. v1 vs v2 vs an exact oracle.
usage: probe_zero_base_neg_exp_sweep.py [frac|float] [seed] [n]"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random, warnings
from fractions import Fraction as F
from common import *
import intervals

MODE = sys.argv[1] if len(sys.argv) > 1 else 'frac'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261004
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400
rng = random.Random(SEED)
INF = math.inf
if MODE == 'frac':
    POS = [F(1, 16), F(1, 9), F(1, 4), F(1), F(4), F(9), F(16)]
else:  # every value and every corner power is an exact float
    POS = [F(1, 16), F(1, 4), F(1), F(4), F(16), F(64)]
EXPS = [F(-2), F(-3, 2), F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(3, 2), F(2)]
NEG_EXPS = [e for e in EXPS if e < 0]


def conv(v):
    return float(v) if MODE == 'float' else v


def ipow(x, y):
    if y.denominator == 1:
        return x ** int(y)
    r = F(math.isqrt(x.numerator), math.isqrt(x.denominator))
    assert r * r == x
    return r ** int(2 * y)


def cval(x, y):
    """corner value or limit; x may be 0 (limit from the right)"""
    if x == 0:
        return F(0) if y > 0 else (INF if y < 0 else F(1))
    return ipow(x, y)


def contains(piece, v):
    lo, hi, lc, hc = piece
    return (lo < v or (lc and lo == v)) and (v < hi or (hc and v == hi))


def truth(bs, es):
    out, clipped = [], False
    for X in bs:
        lo, hi, lc, hc = X
        if lo < 0:
            clipped = True
            if hi > 0 or (hi == 0 and hc):
                lo, lc = F(0), True
            else:
                continue
        for Y in es:
            yl, yh, ylc, yhc = Y
            if lo == 0 and lc:
                if yh > 0:
                    out.append((F(0), F(0), True, True))
                if yl < 0 or (yl == 0 and ylc):
                    clipped = True  # 0 ** y, y <= 0, dropped
            # positive part
            plo, plc = (lo, lc) if lo > 0 else (F(0), False)
            if hi < plo or (hi == plo and not (plc and hc)):
                continue  # no positive part ([0] handled above)
            corners = [(cval(x, y), xc and yc and x != 0) for x, xc in ((plo, plc), (hi, hc)) for y, yc in ((yl, ylc), (yh, yhc))]
            vals = [v for v, _ in corners]
            mn, mx = min(vals), max(vals)
            one = contains((plo, hi, plc, hc), 1) or contains(Y, 0)

            def att(v):
                if v == 0 or v == INF:
                    return False
                if v == 1:
                    return one
                return any(c == v and ok for c, ok in corners)
            out.append((mn, mx, att(mn), att(mx)))
    return out, clipped


def rand_base():
    k = rng.randint(1, 3)
    vals = sorted(rng.sample(POS, 2 * k - 1))
    pieces = []
    zero_kind = rng.choice(['closed', 'open', 'point', 'closed'])
    if zero_kind == 'point':
        pieces.append((F(0), F(0), True, True))
        rest = vals
    else:
        pieces.append((F(0), vals[0], zero_kind == 'closed', rng.random() < 0.6))
        rest = vals[1:]
    for i in range(0, len(rest) - 1, 2):
        lo, hi = rest[i], rest[i + 1]
        if rng.random() < 0.2:
            pieces.append((lo, lo, True, True))
        else:
            pieces.append((lo, hi, rng.random() < 0.5, rng.random() < 0.5))
    r = rng.random()
    if r < 0.12:  # a negative piece too
        pieces.insert(0, (F(-4), F(-1), True, rng.random() < 0.5))
    elif r < 0.2 and zero_kind != 'point':  # the [0, b] piece straddles 0
        lo, hi, lc, hc = pieces[0]
        pieces[0] = (F(-1), hi, rng.random() < 0.5, hc)
    return pieces


def rand_exp():
    k = rng.randint(1, 2)
    while True:
        vals = sorted(rng.sample(EXPS, 2 * k))
        if vals[0] < 0:
            break
    out = []
    for i in range(k):
        lo, hi = vals[2 * i], vals[2 * i + 1]
        if rng.random() < 0.15:
            out.append((lo, lo, True, True))
        else:
            out.append((lo, hi, rng.random() < 0.5, rng.random() < 0.5))
    return out


def conv_pieces(ps):
    return [(conv(a), conv(b), c, d) for a, b, c, d in ps]


EXTRA = [F(1, 10 ** 40), F(1, 10 ** 6), F(10 ** 6), F(10 ** 40), F(-1, 10 ** 6), F(-10 ** 6)]
stats, examples = {}, {}
wrong_warn = 0
for trial in range(N):
    bs, es = rand_base(), rand_exp()
    t, clipped = truth(bs, es)
    b1, e1 = mk1(conv_pieces(bs)), mk1(conv_pieces(es))
    b2, e2 = mk2(conv_pieces(bs)), mk2(conv_pieces(es))
    r1, x1 = run(lambda: b1 ** e1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            r2, x2 = b2 ** e2, None
        except Exception as e:  # noqa: BLE001
            r2, x2 = None, f'{type(e).__name__}: {e}'
    got_clip = any(issubclass(m.category, intervals.DomainClippedWarning) for m in w)
    if x2 is None and got_clip != clipped:
        wrong_warn += 1
        examples.setdefault('WARN', (bs, es, str(r2), clipped, [str(m.message) for m in w]))
    if x1:
        k1 = 'v1 ' + x1.split(':')[0]
    elif not well_formed1(r1):
        k1 = 'v1 MALFORMED'
    else:
        d = compare(pieces1(r1), t, EXTRA)
        k1 = 'v1 ok' if not d else ('v1 (-inf,inf)' if s1(r1) == '(-inf, inf)' else 'v1 DIFF')
    if x2:
        k2 = 'v2 ' + x2.split(':')[0]
    else:
        d2 = compare(pieces2(r2), t, EXTRA)
        k2 = 'v2 ok' if not d2 else 'v2 DIFF'
    key = f'{k1:22s} | {k2}'
    stats[key] = stats.get(key, 0) + 1
    if key not in examples:
        examples[key] = (bs, es, x1 or s1(r1), x2 or str(r2), [(str(a), str(b), c, d) for a, b, c, d in t])
    if 'DIFF' in k2 and ('DIFF2', trial) not in examples and sum(1 for k in examples if isinstance(k, tuple)) < 5:
        examples[('DIFF2', trial)] = (bs, es, x2 or str(r2), [(str(a), str(b), c, d) for a, b, c, d in t], compare(pieces2(r2), t, EXTRA)[:4])

print(f'mode {MODE} seed {SEED} cases {N}; DomainClippedWarning mismatches {wrong_warn}')
for k, v in sorted(stats.items()):
    print(f'{v:5d}  {k}')
print('--- one example per outcome')
for k, v in examples.items():
    print(k)
    for item in v:
        print('     ', item)

# sabotage: a wrong oracle (flip the closed flag on the first finite end) must be caught on a v2 result
bs, es = [(F(0), F(4), True, True), (F(9), F(16), True, False)], [(F(-1, 2), F(-1, 2), True, True)]
t, _ = truth(bs, es)
r = quiet(lambda: mk2(bs) ** mk2(es))
assert not compare(pieces2(r), t, EXTRA), (str(r), t)
bad = [(a, b, c, not d) if b != INF else (a, b, c, d) for a, b, c, d in t]
assert compare(pieces2(r), bad, EXTRA), 'sabotage NOT caught'
print('sabotage caught;', str(r), t)
