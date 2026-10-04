"""every v1/v2 disagreement on a positive base (b ** A and A ** p with A > 0) is v2 opening an irrational float end"""
from common import *
from collections import Counter
from probe_arith_helpers import run, exact_interval

def close(x, y, tol=1e-9):
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= tol * max(1, abs(x), abs(y))

def explain(i1, R):
    if len(R) != 1 or not close(float(i1.start), float(R.inf)) or not close(float(i1.end), float(R.sup)):
        return 'value-differ'
    reasons = []
    if i1.start_closed != R.inf_closed:
        reasons.append('lo:' + ('v2-open-float' if (not R.inf_closed and isinstance(R.inf, float)) else 'OTHER'))
    if i1.end_closed != R.sup_closed and not math.isinf(i1.end):
        reasons.append('hi:' + ('v2-open-float' if (not R.sup_closed and isinstance(R.sup, float)) else 'OTHER'))
    return ','.join(reasons) or 'same'

st = Counter(); ex = {}
rng = random.Random(9)
for _ in range(3000):
    a = exact_interval(rng); A = to_v2(a)
    which = rng.random()
    if which < 0.5:
        b = rng.choice([2, 3, F(1, 2), 0.5, 10, F(7, 3)])
        v1r = run(lambda: b ** a); R = b ** A; tag = 'b**A'
    else:
        if a.start <= 0:
            continue
        p = rng.choice([F(1, 2), 0.5, F(3, 2), -0.5, F(-1, 3), 2.5])
        v1r = run(lambda: a ** p); R = A ** p; tag = 'A**p'
    if v1r[0] != 'ok':
        k = (tag, 'v1raise ' + v1r[1][:30])
    else:
        k = (tag, explain(v1r[1], R))
    st[k] += 1; ex.setdefault(k, (a, v1r, R))
for k in sorted(st, key=str):
    print(k, st[k])
for k, e in ex.items():
    if 'OTHER' in k[1] or 'value' in k[1] or 'raise' in k[1]:
        print('  example', k, e)
check('no unexplained', not any('OTHER' in k[1] or 'value' in k[1] for k in st))
check('SELFTEST expected mismatch', explain(I(1, False, 2, True), M(1, 3)) == 'same')
report_end(__file__)
