"""exp: v1 (math.exp endpoint-wise) vs v2 vs gmpy2 at 400 bits rounded to nearest. log: v1 cannot run (TypeError);
v2 log(base) checked against gmpy2 alone, plus the interval-base composition A.log() / B.log()."""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random
from fractions import Fraction as F
import gmpy2
from common import *

ctx = gmpy2.get_context(); ctx.precision = 400

def cr(v):
    """mpfr (400 bits) -> nearest double"""
    return float(v)

def ref_exp(x):
    return cr(gmpy2.exp(gmpy2.mpfr(x)))

def ref_log(x, b=None):
    lx = gmpy2.log(gmpy2.mpfr(x))
    return cr(lx if b is None else lx / gmpy2.log(gmpy2.mpfr(b)))

rng = random.Random(99)
xs = [rng.uniform(-700, 700) for _ in range(150)] + [rng.uniform(-2, 2) for _ in range(150)] + [rng.uniform(-1e-5, 1e-5) for _ in range(50)]
st = {'v1==cr': 0, 'v1!=cr': 0, 'v2==[cr]': 0, 'v2!=[cr]': 0}
bad1 = []
for x in xs:
    want = ref_exp(x)
    r1 = V1(x).exp(); r2 = quiet(lambda: V2(x).exp())
    st['v1==cr' if pieces1(r1) == [(want, want, True, True)] else 'v1!=cr'] += 1
    if pieces1(r1) != [(want, want, True, True)]:
        bad1.append((x, pieces1(r1)[0][0], want))
    st['v2==[cr]' if pieces2(r2) == [(want, want, True, True)] else 'v2!=[cr]'] += 1
print('exp of a float point', st)
for b in bad1[:4]:
    print('  v1 off: x=%r v1=%r correctly rounded=%r' % b)

# exp of intervals with float ends: compare as sets (pieces of 1-2 sorted float pairs)
agree = differ = 0
for _ in range(300):
    a, b = sorted(rng.uniform(-50, 50) for _ in range(2))
    lc, hc = rng.random() < .5, rng.random() < .5
    r1 = V1(start=a, end=b, start_closed=lc, end_closed=hc).exp()
    r2 = quiet(lambda: V2(a, b, start_closed=lc, end_closed=hc).exp())
    want = [(ref_exp(a), ref_exp(b), lc, hc)]
    d1, d2 = compare(pieces1(r1), want), compare(pieces2(r2), want)
    if not d2 and not d1:
        agree += 1
    else:
        differ += 1
        if differ < 4:
            print('  exp interval differ', a, b, lc, hc, pieces1(r1), pieces2(r2), want)
print(f'exp float intervals vs correctly rounded ends: both right {agree}, some wrong {differ}')

# exact operands: v2's open enclosure must contain the true value; v1's closed rounded point may not be it
cnt = {'v2 encloses': 0, 'v2 misses': 0}
for _ in range(100):
    q = F(rng.randint(-3000, 3000), rng.randint(1, 97))
    r2 = quiet(lambda: V2(q).exp())
    t = gmpy2.exp(gmpy2.mpfr(q.numerator) / q.denominator)
    lo, hi, lc, hc = pieces2(r2)[0]
    ok = (gmpy2.mpfr(lo) < t or (lc and gmpy2.mpfr(lo) == t)) and (t < gmpy2.mpfr(hi) or (hc and t == gmpy2.mpfr(hi)))
    cnt['v2 encloses' if ok else 'v2 misses'] += 1
print('exp of a Fraction point, v2 enclosure holds the true value:', cnt)

# log: v2 against gmpy2 (v1 raises TypeError for every non-empty input)
cnt = {'ok': 0, 'bad': 0}
for _ in range(300):
    x = rng.choice([rng.uniform(1e-300, 1e-200), rng.uniform(0.001, 5), rng.uniform(1, 1e300)])
    b = rng.choice([None, 2, 10, 0.5, 3.0, F(1, 3), 7])
    r2 = quiet(lambda: V2(x).log(b))
    want = ref_log(x, b)
    lo, hi, lc, hc = pieces2(r2)[0]
    # nearest class on a float operand: the point itself, or (for an exact base) an enclosure holding it
    ok = (lo == hi == want) or (lo <= want <= hi)
    cnt['ok' if ok else 'bad'] += 1
    if not ok:
        print('  log bad', x, b, pieces2(r2), want)
print('v2 log(x, base) of a float point vs gmpy2:', cnt)
r1, e1 = run(lambda: V1(2.0).log())
print('v1 V1(2.0).log():', e1)

# interval base: v2 composition A.log() / B.log() vs brute force over a grid
A, B = V2(1, 8), V2(2, 4)
comp = quiet(lambda: A.log() / B.log())
print('A.log() / B.log() for A=[1,8], B=[2,4]:', comp, '(exact image [0, 3])')
grid = [(F(a, 4), F(b, 4)) for a in range(4, 33) for b in range(8, 17)]
missing = [(a, b) for a, b in grid if ref_log(a, b) not in comp]
print('grid points log_b(a) outside the composition:', len(missing), 'of', len(grid))
# sabotage: a wrong reference is caught
assert pieces1(V1(1.0).exp()) != [(ref_exp(2.0),) * 2 + (True, True)]
print('sabotage caught')
