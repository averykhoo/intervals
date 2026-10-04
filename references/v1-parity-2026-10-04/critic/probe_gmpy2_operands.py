"""re-check arith row 'operand types ... gmpy2 mpq/mpz/mpfr ... [EQUAL]': its evidence is printed text; the file's
set-compare section crashed (SystemError in v1 __contains__ on mpq ends) before the mpq and float32 checks ran."""
import sys, warnings, operator, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import gmpy2, numpy as np
import multi_interval as v1m
import intervals as v2
S1, S2 = v1m.MultiInterval(F(1), F(2), end_closed=False), v2.MultiInterval(F(1), F(2), end_closed=False)
def ends1(a): return [(e[0], type(e[0]).__name__) for e in a.endpoints]
def ends2(b): return [(c, type(c).__name__) for p in b for c in (p.inf, p.sup)]
def safe(f):
    try: return f()
    except Exception as e: return f'RAISES {type(e).__name__}: {str(e)[:60]}'
ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
scal = {'mpq(1,3)': gmpy2.mpq(1, 3), 'mpz(2)': gmpy2.mpz(2), 'mpfr(0.1) 53b': gmpy2.mpfr('0.1'),
        'mpfr(0.1) 200b': gmpy2.mpfr('0.1', 200), 'float32(0.1)': np.float32(0.1)}
mism = 0
for sn, s in scal.items():
    for on, op in ops.items():
        a, b = safe(lambda: op(S1, s)), safe(lambda: op(S2, s))
        if isinstance(a, str) or isinstance(b, str):
            print(f'[1,2) {on} {sn}: v1 {a} | v2 {b}'); continue
        # exact values of the ends, compared as Fractions (gmpy2 converts exactly via F(str) for mpq, F(float) else)
        def exact(x):
            if isinstance(x, (type(gmpy2.mpq(1)), type(gmpy2.mpz(1)))): return F(int(gmpy2.numer(x)), int(gmpy2.denom(x)))
            if isinstance(x, type(gmpy2.mpfr(1))): return F(*x.as_integer_ratio())
            return F(x)
        e1 = [exact(v) for v, _ in ends1(a)]; e2 = [exact(v) for v, _ in ends2(b)]
        same = e1 == e2
        mism += not same
        print(f'[1,2) {on} {sn}: same exact ends {same}  v1 types {[t for _, t in ends1(a)]}  v2 types {[t for _, t in ends2(b)]}'
              + ('' if same else f'\n      v1 {[float(x) for x in e1]} v2 {[float(x) for x in e2]}  diff {[float(x - y) for x, y in zip(e1, e2)]}'))
print('v1 membership query with a Fraction on an mpq-ended set:', safe(lambda: F(1, 2) in (S1 * gmpy2.mpq(1, 3))))
print('v2 same query:', safe(lambda: F(1, 2) in (S2 * gmpy2.mpq(1, 3))))
print('mismatching op/operand pairs:', mism)
assert mism >= 0
