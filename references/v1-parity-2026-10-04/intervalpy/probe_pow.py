from common import *
from collections import Counter
sys.path.insert(0, 'tests')
import oracles  # noqa: E402
from probe_arith_helpers import exact_interval, exact_val, run

rng = random.Random(5)
st = Counter(); ex = {}
# 1. integer exponents (Real), exact oracle
for _ in range(1500):
    a = exact_interval(rng); A = to_v2(a)
    n = rng.choice([0, 1, 2, 3, 4, -1, -2, -3, 2.0, 3.0, F(4, 2)])
    v1r = run(lambda: a ** n); R = A ** n
    pts = probe_points(*ends_of(a), *ends_of(R), *(ends_of(v1r[1]) if v1r[0] == 'ok' else []))
    truth = {p: oracles.attained('pow', p, A.cuts, int(n)) for p in pts}
    check('pow int v2', all((p in R) == t for p, t in truth.items()), (a, n, R))
    if v1r[0] != 'ok':
        o = 'v1raise:' + v1r[0] + ':' + v1r[1][:40]
    else:
        bad = [p for p, t in truth.items() if (p in v1r[1]) != t]
        o = 'v1ok' if not bad else ('v1loose' if all(p in v1r[1] for p in bad) else 'v1WRONG')
    key = ('int-exp', 'neg-start' if a.start < 0 else ('zero-start' if a.start == 0 else 'pos-start'), 'n<0' if n < 0 else ('n=0' if n == 0 else ('odd' if int(n) % 2 else 'even')), o)
    st[key] += 1; ex.setdefault(key, (a, n, v1r, R))

def approx_same(i1, R, tol=1e-9):
    """v1 Interval vs a contiguous v2 set, endpoints within tol (relative), flags equal"""
    if len(R) != 1:
        return False
    def close(x, y):
        if math.isinf(x) or math.isinf(y):
            return x == y
        return abs(x - y) <= tol * max(1, abs(x), abs(y))
    return close(float(i1.start), float(R.inf)) and close(float(i1.end), float(R.sup)) and \
        i1.start_closed == R.inf_closed and (i1.end_closed == R.sup_closed or math.isinf(i1.end))

# 2. non-integer real exponents and interval exponents
for _ in range(1500):
    a = exact_interval(rng); A = to_v2(a)
    kind = rng.choice(['real', 'interval', 'point-interval'])
    if kind == 'real':
        p = rng.choice([F(1, 2), 0.5, F(3, 2), -0.5, F(-1, 3), 2.5])
        v1r = run(lambda: a ** p); R = A ** p
    elif kind == 'interval':
        b = I(*sorted([F(rng.randint(-6, 6), 2), F(rng.randint(-6, 6), 2)])[:1], False, 0, True) if False else None
        lo, hi = sorted([F(rng.randint(-6, 6), 2), F(rng.randint(-6, 6), 2)])
        if lo == hi:
            hi += 1
        b = I(lo, rng.random() < 0.5, hi, rng.random() < 0.5); B = to_v2(b); p = b
        v1r = run(lambda: a ** b); R = A ** B
    else:
        n = rng.choice([2, 3, -1]); b = I(n, False, n, True); p = b
        v1r = run(lambda: a ** b); R = A ** M(n)
    sign = 'neg-start' if a.start < 0 else ('zero-start' if a.start == 0 else 'pos-start')
    if v1r[0] != 'ok':
        o = 'v1raise:' + v1r[0] + ':' + v1r[1][:40]
    else:
        o = 'agree~' if approx_same(v1r[1], R) else 'DIFFER'
    key = (kind, sign, o); st[key] += 1; ex.setdefault(key, (a, p, v1r, R))

for k in sorted(st, key=str):
    print(k, st[k])
for k, e in sorted(ex.items(), key=str):
    if k[-1] not in ('v1ok', 'agree~'):
        print('  example', k, e)
check('SELFTEST expected mismatch', approx_same(I(1, False, 2, True), M(1, 3)))
report_end(__file__)
