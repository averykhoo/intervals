"""A op s and s op A (+ - * /) with s a gmpy2 mpq / mpz / mpfr(53 bits) / mpfr(200 bits) or np.float32, A a single
Fraction-ended piece (non-dyadic ends). oracle: exact corners in Fraction, then either the exact value or it rounded
once to nearest. each side's ends are classified exact / nearest / other (end values taken exactly: mpq by
numer/denom, mpfr by as_integer_ratio). flags compared with the exact flags. `sab`: v2's expectation for mpq is
'nearest', must fail."""
import sys, random, warnings, operator
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import gmpy2, numpy as np
import multi_interval as v1m
import intervals as v2
SAB = len(sys.argv) > 1
MPQ, MPZ, MPFR = type(gmpy2.mpq(1, 2)), type(gmpy2.mpz(1)), type(gmpy2.mpfr(1))
def exact(x):
    if isinstance(x, (MPQ, MPZ)): return F(int(gmpy2.numer(x)), int(gmpy2.denom(x)))
    if isinstance(x, MPFR): return F(*x.as_integer_ratio())
    if isinstance(x, np.floating): return F(float(x))
    return F(x)
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
def scalar(kind, rng):
    q = F(rng.randint(-50, 50), rng.randint(1, 13))
    if kind == 'mpq': return gmpy2.mpq(q.numerator, q.denominator)
    if kind == 'mpz': return gmpy2.mpz(rng.randint(-50, 50))
    if kind == 'mpfr53': return gmpy2.mpfr(str(float(q)))
    if kind == 'mpfr200': return gmpy2.mpfr(gmpy2.mpq(q.numerator, q.denominator), 200)
    if kind == 'float32': return np.float32(float(q))
    if kind == 'float': return float(q)
    if kind == 'np.float64': return np.float64(float(q))
EXACT_KINDS = {'mpq', 'mpz', 'mpfr200'}    # v2-plan.md ~1117: a Rational exact by type; a non-double real exact by value
rng = random.Random(1123)
tally = {}
for it in range(400):
    lo = F(rng.randint(-30, 30), rng.choice([3, 7, 9, 11])); hi = lo + F(rng.randint(1, 40), rng.choice([3, 5, 7]))
    lc, hc = rng.random() < .5, rng.random() < .5
    A1 = v1m.MultiInterval(lo, hi, start_closed=lc, end_closed=hc); A2 = v2.MultiInterval(lo, hi, start_closed=lc, end_closed=hc)
    for kind in ['mpq', 'mpz', 'mpfr53', 'mpfr200', 'float32', 'float', 'np.float64']:
        s = scalar(kind, rng); X = exact(s)
        for on, op in OPS.items():
            for side in ('A op s', 's op A'):
                if on == '/' and side == 'A op s' and X == 0: continue
                if on == '/' and side == 's op A' and lo <= 0 <= hi: continue
                cs = [(op(F(e), X) if side == 'A op s' else op(X, F(e)), c) for e, c in ((lo, lc), (hi, hc))]
                if X == 0 and on == '*': want = [(F(0), True, F(0), True)]
                else:
                    (u, uc), (w, wc) = sorted(cs, key=lambda t: t[0]); want = [(u, uc, w, wc)]
                try:
                    r1 = op(A1, s) if side == 'A op s' else op(s, A1)
                    g1 = [(exact(r1.endpoints[i][0]), r1.endpoints[i][1] == 0, exact(r1.endpoints[i + 1][0]), r1.endpoints[i + 1][1] == 0) for i in range(0, len(r1.endpoints), 2)]
                except Exception as e: g1 = f'RAISES {type(e).__name__}'
                r2 = op(A2, s) if side == 'A op s' else op(s, A2)
                g2 = [(exact(p.inf), bool(p.inf_closed), exact(p.sup), bool(p.sup_closed)) for p in r2]
                def cls(g):
                    if isinstance(g, str): return g
                    if len(g) != 1: return 'pieces'
                    if (g[0][1], g[0][3]) != (want[0][1], want[0][3]) and not (want[0][0] == want[0][2]): return 'flags'
                    if (g[0][0], g[0][2]) == (want[0][0], want[0][2]): return 'exact'
                    if (g[0][0], g[0][2]) == (F(float(want[0][0])), F(float(want[0][2]))): return 'nearest'
                    return 'other'
                k = (kind, on, side); t = tally.setdefault(k, {})
                c1, c2 = 'v1 ' + cls(g1), 'v2 ' + cls(g2)
                t[c1] = t.get(c1, 0) + 1; t[c2] = t.get(c2, 0) + 1
                is_exact = kind in ('mpq', 'mpz') or (kind == 'mpfr200' and F(float(X)) != X)   # a wide mpfr equal to a double is that double
                exp2 = 'v2 exact' if is_exact != (SAB and kind == 'mpq') else 'v2 nearest'
                if c2 != exp2 and not (c2 == 'v2 exact' and exp2 == 'v2 nearest' and F(float(want[0][0])) == want[0][0] and F(float(want[0][2])) == want[0][2]):
                    t['V2 UNEXPECTED'] = t.get('V2 UNEXPECTED', 0) + 1; t.setdefault('ex', (lo, hi, lc, hc, s, g1, g2, want))
bad = 0
for k, t in tally.items():
    print(k, {a: b for a, b in t.items() if a != 'ex'}, '' if 'ex' not in t else t['ex'])
    bad += t.get('V2 UNEXPECTED', 0)
print('v2 unexpected', bad)
assert bad == 0
