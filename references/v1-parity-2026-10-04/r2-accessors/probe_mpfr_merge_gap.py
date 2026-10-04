"""a v1 merge of [-3, wt) and [d, 2] with wt = 200-bit mpfr('0.1') < d = 53-bit mpfr('0.1'): is the gap [wt, d) kept?"""
import sys, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
from fractions import Fraction
import multi_interval as v1
import intervals as v2
from gmpy2 import mpfr
wt, d = mpfr('0.1', 200), mpfr('0.1')
print('wt < d:', wt < d, ' exact:', Fraction(*wt.as_integer_ratio()) < Fraction(*d.as_integer_ratio()))
print('(wt, 0) < (d, 0):', (wt, 0) < (d, 0), ' (d,0) <= (wt,-1):', (d, 0) <= (wt, -1), ' wt == d:', wt == d)
A1 = v1.MultiInterval(-3, wt, end_closed=False); B1 = v1.MultiInterval(d, 2)
for name, m in [('merge', v1.MultiInterval.merge(A1, B1)), ('union', A1.union(B1))]:
    print('v1', name, m.endpoints)
A2 = v2.MultiInterval(-3, wt, end_closed=False); B2 = v2.MultiInterval(d, 2)
U = A2 | B2
print('v2', U)
mid = (Fraction(*wt.as_integer_ratio()) + Fraction(*d.as_integer_ratio())) / 2   # a point of the gap
print('gap point in v1 union:', mid in v1.MultiInterval.merge(A1, B1) if False else 'n/a', '| in v2:', mid in U,
      '| in A or B (brute):', (-3 <= mid < Fraction(*wt.as_integer_ratio())) or (Fraction(*d.as_integer_ratio()) <= mid <= 2))
m1 = v1.MultiInterval.merge(A1, B1)
print('v1 contains float(mid)?', float(mid) in m1, ' float(mid) exact in gap?', Fraction(*wt.as_integer_ratio()) <= Fraction(float(mid)) < Fraction(*d.as_integer_ratio()))
