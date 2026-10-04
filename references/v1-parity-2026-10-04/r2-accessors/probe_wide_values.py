"""inf/sup/degenerate_points/cardinality values and types on 200-bit mpfr, mpz, mpq ends, v1 vs v2, exact check"""
import sys, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
from fractions import Fraction
import multi_interval as v1
import intervals as v2
import numpy as np
import gmpy2
from gmpy2 import mpfr, mpz, mpq
X = lambda x: Fraction(*[int(t) for t in x.as_integer_ratio()]) if hasattr(x, 'as_integer_ratio') else Fraction(x)
with gmpy2.local_context(precision=200):  # gmpy2 arithmetic rounds to the context precision
    wide = mpfr(1) + mpfr(2) ** -150
    negwide = -wide
with gmpy2.local_context(precision=200):
    big = mpfr(2) ** 70 + 1
fails = 0
for lo, hi in [(wide, wide), (negwide, wide), (big, big), (mpfr('0.1', 200), mpfr('0.1', 200)), (mpfr('0.1', 200), 1), (10 ** 30, 10 ** 30 + 1), (mpz(10) ** 30, mpz(10) ** 30 + 1), (mpq(1, 3), mpq(4, 2)),
               (np.longdouble('0.1'), np.longdouble(2))]:
    a = v1.MultiInterval(lo) if lo == hi else v1.MultiInterval(lo, hi)
    b = v2.MultiInterval(lo) if lo == hi else v2.MultiInterval(lo, hi)
    ok_inf = X(a.infimum) == X(b.inf) == X(lo)
    ok_sup = X(a.supremum) == X(b.sup) == X(hi)
    exact_len = X(hi) - X(lo)
    c1, c2 = a.cardinality, b.size
    fails += not (ok_inf and ok_sup)
    print(f'{lo!r:.40}: v1 inf {type(a.infimum).__name__} v2 inf {type(b.inf).__name__} {b.inf!r:.60} exact-equal={ok_inf and ok_sup}'
          f' | is_integral v1 {a.is_integral} v2 {b.is_integral} | degenerate_points v1 {[type(x).__name__ for x in a.degenerate_points]}'
          f' v2 {[type(x).__name__ for x in b.degenerate_points]}'
          f' | length exact? v1 {X(c1[1]) == exact_len} ({type(c1[1]).__name__}) v2 {X(c2.length) == exact_len} ({type(c2.length).__name__})')
# a deliberately wrong expectation must be caught
assert X(v2.MultiInterval(wide).inf) != X(mpfr(wide, 53)), 'sabotage: 53-bit rounding of wide must differ'
print('fails', fails)
