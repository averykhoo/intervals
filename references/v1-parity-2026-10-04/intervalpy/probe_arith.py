from common import *
from collections import Counter
sys.path.insert(0, 'tests')
import oracles  # noqa: E402

def run(f):
    try:
        return 'ok', f()
    except Exception as e:
        return type(e).__name__, str(e)[:70]

def exact_val(rng):
    r = rng.random()
    if r < 0.5:
        return rng.randint(-5, 5)
    return F(rng.randint(-20, 20), rng.randint(1, 5))

def exact_interval(rng):
    while True:
        a, b = sorted([exact_val(rng), exact_val(rng)])
        so, ec = rng.random() < 0.5, rng.random() < 0.5
        if rng.random() < 0.12:
            a, so = -INF, True
        if rng.random() < 0.12:
            b, ec = INF, False
        if a == b:
            so, ec = False, True
        try:
            return I(a, so, b, ec)
        except ValueError:
            pass

def judge(name, op, a_cuts, b, v1res, R, extra=()):
    """compare v1 (if it answered) and v2 against the exact oracle at finite probe points"""
    pts = probe_points(*[c.value for c in a_cuts], *([c.value for c in b] if isinstance(b, tuple) else []), *ends_of(R), *extra,
                       *(ends_of(v1res[1]) if v1res[0] == 'ok' else []))
    truth = {p: oracles.attained(op, p, a_cuts, b) for p in pts}
    v2_ok = all((p in R) == t for p, t in truth.items())
    check(name + ' v2', v2_ok, f'{a_cuts} {b} -> {R}')
    if v1res[0] != 'ok':
        return 'v1raise:' + v1res[0]
    bad = [p for p, t in truth.items() if (p in v1res[1]) != t]
    if not bad:
        return 'v1ok'
    loose = all((p in v1res[1]) and not truth[p] for p in bad)
    return 'v1loose' if loose else 'v1WRONG'

rng = random.Random(11)
stats = Counter(); ex = {}
ops = [('add', '+'), ('sub', '-'), ('mul', '*'), ('div', '/')]
for _ in range(700):
    a, b = exact_interval(rng), exact_interval(rng)
    A, B = to_v2(a), to_v2(b)
    s = exact_val(rng)
    for op, sym in ops:
        f = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}[sym]
        for kind, x1, y1, x2, y2 in [('II', a, b, A, B), ('IR', a, s, A, s), ('RI', s, a, s, A)]:
            v1r = run(lambda: f(x1, y1))
            R = f(x2, y2)
            xc = x2.cuts if isinstance(x2, M) else M(x2).cuts
            yc = y2.cuts if isinstance(y2, M) else M(y2).cuts
            o = judge(f'{op} {kind}', op, xc, yc, v1r, R)
            stats[(op, kind, o)] += 1
            ex.setdefault((op, kind, o), (x1, sym, y1, v1r, R))
    # unary
    for op, f1, f2 in [('neg', lambda: -a, lambda: -A), ('pos', lambda: +a, lambda: +A), ('abs', lambda: abs(a), lambda: abs(A)),
                       ('reciprocal', lambda: a.reciprocal(), lambda: A.reciprocal())]:
        v1r = run(f1); R = f2()
        o = judge(op, op, A.cuts, None, v1r, R)
        stats[(op, '', o)] += 1
        ex.setdefault((op, '', o), (a, v1r, R))
for k in sorted(stats, key=str):
    print(k, stats[k])
for k, e in sorted(ex.items(), key=str):
    if k[2] != 'v1ok':
        print('  example', k, e)

# floats: v1 python corner arithmetic vs v2 round-to-nearest
rng = random.Random(3)
fl = Counter()
for _ in range(2000):
    x = sorted([rng.uniform(-10, 10) for _ in range(2)]); y = sorted([rng.uniform(-10, 10) for _ in range(2)])
    a, b = I(x[0], False, x[1], True), I(y[0], False, y[1], True)
    A, B = to_v2(a), to_v2(b)
    for sym, f in [('+', lambda p, q: p + q), ('-', lambda p, q: p - q), ('*', lambda p, q: p * q)]:
        r1, R = f(a, b), f(A, B)
        same = (r1.start == R.inf and r1.end == R.sup)
        fl[(sym, same)] += 1
        check('float ' + sym, same, (a, b, r1, R))
    # Fraction + float mix: python double-rounds, v2 rounds once
    fr = F(rng.randint(1, 10**6), 3)
    r1, R = a + fr, A + fr
    fl[('mixed Fraction+float', r1.start == R.inf and r1.end == R.sup)] += 1
print('floats', dict(fl))

# hand-picked
print('A / 0:', run(lambda: I(1, False, 2, True) / 0), M(1, 2) / 0)
print('A / [0]:', run(lambda: I(1, False, 2, True) / I(0, False, 0, True)), M(1, 2) / M(0))
print('0 / A:', run(lambda: 0 / I(1, False, 2, True)), 0 / M(1, 2))
print('0 / [-1,1]:', run(lambda: 0 / I(-1, False, 1, True)), 0 / M(-1, 1))
print('inf / A:', run(lambda: INF / I(1, False, 2, True)), INF / M(1, 2))
print('[0,1] / [-1,1]:', run(lambda: I(0, False, 1, True) / I(-1, False, 1, True)), M(0, 1) / M(-1, 1))
print('[1,2] / [-1,1]:', run(lambda: I(1, False, 2, True) / I(-1, False, 1, True)), M(1, 2) / M(-1, 1))
print('(0,inf) * 0:', run(lambda: I(0, True, INF, False) * 0), M(0, INF, start_closed=False, end_closed=False) * 0)
print('(0,inf) * [0,1]:', run(lambda: I(0, True, INF, False) * I(0, False, 1, True)), M(0, INF, start_closed=False, end_closed=False) * M(0, 1))
print('(-inf,0) + (0,inf):', run(lambda: I(-INF, True, 0, False) + I(0, True, INF, False)), M(-INF, 0, start_closed=False, end_closed=False) + M(0, INF, start_closed=False, end_closed=False))
print('[0].reciprocal:', run(lambda: I(0, False, 0, True).reciprocal()), M(0).reciprocal())
print('[-1,1].reciprocal:', run(lambda: I(-1, False, 1, True).reciprocal()), M(-1, 1).reciprocal())
print('(0,1].reciprocal:', run(lambda: I(0, True, 1, True).reciprocal()), M(0, 1, start_closed=False).reciprocal())
print('[1,inf).reciprocal:', run(lambda: I(1, False, INF, False).reciprocal()), M(1, INF, end_closed=False).reciprocal())
print('degenerate + Interval-as-Real:', run(lambda: I(1, False, 1, True) + I(2, False, 2, True)), M(1) + M(2))
print('Interval + str:', run(lambda: I(1, False, 2, True) + 'x'), run(lambda: M(1, 2) + 'x'))
print('int/int exactness:', run(lambda: I(1, False, 2, True) / 3), M(1, 2) / 3)
# SELFTEST: wrong expectation (oracle of add vs a mul result)
judge('SELFTEST expected mismatch', 'add', M(1, 2).cuts, M(3, 4).cuts, ('ok', I(3, False, 8, True)), M(3, 8))
report_end(__file__)
