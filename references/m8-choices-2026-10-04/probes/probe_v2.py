import math
from fractions import Fraction
from intervals import MultiInterval as M, REALS, EMPTY
a = M(-math.inf, 5)
print('repr', repr(a), '| inf', repr(a.inf), 'inf_closed', a.inf_closed, '| sup', a.sup)
print('[-inf,5) :', M(-math.inf, 5, end_closed=False), '| (-inf,5]:', M(-math.inf, 5, start_closed=False))
print('size [-inf,5]', a.size, '| size (-inf,5]', M(-math.inf, 5, start_closed=False).size)
print('hull of 2 pieces', (M(1,2)|M(4,5)).hull, '| REALS', REALS, '| REALS.inf', REALS.inf)
print('slice A[:5]', M(0, 10)[:5], '| slice with Fraction', M(0,10)[Fraction(1,3):Fraction(1,2)])
# adjacency merge with half-open pieces: union of adjacent days
d1 = M(0, 86400, end_closed=False); d2 = M(86400, 172800, end_closed=False)
print('[0,86400) | [86400,172800) =', d1 | d2, '| len', len(d1|d2))
c1 = M(0, Fraction(86399999999, 10**6)); c2 = M(86400, Fraction(172799999999, 10**6))
print('v1-style closed days union =', c1 | c2, '| len', len(c1|c2))
print('half microsecond in v1-style day:', Fraction(86399999999, 10**6) + Fraction(1, 2*10**6) in c1, '| in half-open day:', Fraction(86399999999, 10**6) + Fraction(1, 2*10**6) in d1)
print('size half-open day', d1.size, '| size closed-day', c1.size, '| wid', d1.wid(), c1.wid())
# expand
print('expand', M(0, 10).expand(Fraction(1, 2)))
# what does inf of empty do
try: print(EMPTY.inf)
except Exception as e: print('EMPTY.inf RAISES', type(e).__name__, e)
# coercion of foreign types
import datetime as dt
try: print(M(dt.datetime(2024,1,1)))
except Exception as e: print('M(datetime) RAISES', type(e).__name__, e)
try: print(dt.datetime(2024,1,1) in M(0, 1))
except Exception as e: print('datetime in M RAISES', type(e).__name__, e)
# Fraction ends print
print('fraction end repr', repr(M(0, Fraction(86399999999, 10**6))))
print('float inf vs int: type(a.inf)', type(a.inf).__name__, '| M(1, inf).is_finite', M(1, math.inf).is_finite)
# mid of half-bounded
print('mid [-inf,5]', a.mid(), '| wid', a.wid())
