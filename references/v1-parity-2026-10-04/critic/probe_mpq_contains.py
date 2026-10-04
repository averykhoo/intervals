"""v1 __contains__ on a set whose ends are gmpy2 mpq (from S * mpq): which queries crash? (the arith probe hit SystemError)"""
import sys, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import gmpy2
import multi_interval as v1m
import intervals as v2
a1 = v1m.MultiInterval(F(1), F(2), end_closed=False) * gmpy2.mpq(1, 3)
a2 = v2.MultiInterval(F(1), F(2), end_closed=False) * gmpy2.mpq(1, 3)
crash = 0
for x in [F(1, 3), F(1, 2), F(2, 3), F(1, 3) - F(1, 10**6), 0.5, 1, 0, gmpy2.mpq(1, 2), F(5, 7)]:
    def go(a):
        try: return x in a
        except Exception as e: return f'RAISES {type(e).__name__}: {str(e)[:50]}'
    r1, r2 = go(a1), go(a2)
    crash += isinstance(r1, str)
    print(repr(x), 'v1', r1, '| v2', r2)
print('v1 crashes', crash)
assert crash >= 0
