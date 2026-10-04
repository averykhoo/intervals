import sys, warnings; sys.path[:0] = ['.', 'archive/v1']
from fractions import Fraction as F
import multi_interval as v1
import intervals as v2
MI = v2.MultiInterval
ok = True
def check(name, cond):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name); ok &= bool(cond)
# 1. reciprocal split: v1 whole line, v2 gap (-0.5, 0.5)
r1 = v1.MultiInterval(-2, 2).reciprocal() if hasattr(v1.MultiInterval(-2,2),'reciprocal') else None
print('v1 1/[-2,2] =', r1)
r2 = 1 / MI(-2, 2)
print('v2 1/[-2,2] =', repr(r2))
check('v2 excludes 0 and 0.25', 0 not in r2 and F(1,4) not in r2)
check('v2 includes -0.5, 0.5, 100', -0.5 in r2 and 0.5 in r2 and 100 in r2)
check('v1 includes 0.25 (lost the gap)', r1 is not None and 0.25 in r1)
# deliberately wrong expectation, must FAIL
check('SABOTAGE expect v2 contains 0.25 (should FAIL)', F(1,4) in r2)
# 2. modulo: [12,18.7] mod 7.5 = [0,3.7] U [4.5,7.5)
m = MI(12, 18.7) % 7.5
print('v2 [12,18.7] % 7.5 =', repr(m))
check('mod gap 4 excluded', 4 not in m and 4.5 in m and 7.5 not in m and 0 in m)
# 3. regex compiled at module level
import intervals.fmt as fmt, re
check('fmt._TOKEN is compiled at module level', isinstance(fmt._TOKEN, re.Pattern))
# 4. negative base, non-integral exponent: real-only, clipped
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    p = MI(-4, 4) ** 0.5
print('v2 [-4,4] ** 0.5 =', repr(p), [type(x.message).__name__ for x in w])
check('pow on negative base clipped to real domain', p == MI(0, 2))
print('ALL (excluding sabotage) OK' if not ok else 'unexpected: sabotage passed')
