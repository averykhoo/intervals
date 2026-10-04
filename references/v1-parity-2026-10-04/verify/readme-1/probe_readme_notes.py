import sys, random
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
fails = 0
def check(name, ok):
    global fails
    print(('OK  ' if ok else 'FAIL'), name)
    if not ok: fails += 1

# 1. reciprocal of [-2,2]: gemini says (-inf,-0.5] U [0.5,inf)
a1 = v1.MultiInterval(-2, 2); a2 = v2.MultiInterval(-2, 2)
r1 = a1.reciprocal(); r2 = 1 / a2
print('v1 1/[-2,2] =', r1, '| v2 =', r2)
for x in [F(-1,2), F(1,2), F(1,4), 0, F(-1,4), 10, -10, F(51,100), F(49,100)]:
    truth = abs(x) >= F(1,2)   # exists y in [-2,2]\{0} with 1/y == x
    check(f'v2 1/[-2,2] contains {x} == {truth}', (x in r2) == truth)
    print('    v1 says', x in r1)
# deliberately wrong expectation must be caught
check('SABOTAGE (expect FAIL): v2 1/[-2,2] contains 0', 0 in r2)

# 2. regex at module level
import intervals.fmt as fmt, re
check('fmt._TOKEN is a module-level compiled regex', isinstance(fmt._TOKEN, re.Pattern))
print('parse "[1,2]" ->', v2.MultiInterval.parse('[1,2]'))

# 3. modulo by scalar: gemini's slice-and-shift construction vs v2 %, and v1 %
random.seed(1)
def brute_mod(lo, hi, m, lc, hc):
    # membership oracle: y in A mod m iff exists x in A, x - m*floor(x/m) == y; check sample y
    pass
dis = 0; n = 0
for _ in range(300):
    lo = F(random.randint(-20, 20), random.choice([1,2,4]))
    hi = lo + F(random.randint(0, 30), random.choice([1,2,4]))
    m = F(random.randint(1, 8), random.choice([1,2]))
    A2 = v2.MultiInterval(lo, hi)
    R2 = A2 % m
    # slice-and-shift construction (gemini), computed with v2 set ops
    import math
    k0 = math.floor(lo / m); k1 = math.floor(hi / m)
    S = v2.MultiInterval()
    for k in range(k0, k1 + 1):
        chunk = A2 & v2.MultiInterval(k*m, (k+1)*m, end_closed=False) if 'end_closed' in v2.MultiInterval.__init__.__code__.co_varnames else None
        if chunk is None: break
        S = S | (chunk - k*m)
    try:
        R1 = v1.MultiInterval(float(lo), float(hi)) % float(m)
    except Exception as e:
        R1 = e
    # sample points
    for y in [F(i, 8) * m for i in range(-1, 9)]:
        n += 1
        x_in = any(lo <= y + k*m <= hi for k in range(k0 - 1, k1 + 2)) and 0 <= y < m
        if (y in R2) != x_in: dis += 1; print('v2 mod mismatch', lo, hi, m, y)
        if S is not None and (y in S) != x_in: dis += 1; print('slice mismatch', lo, hi, m, y)
        if not isinstance(R1, Exception) and ((float(y) in R1) != x_in): print('  v1 mod differs', lo, hi, m, y, R1)
check(f'v2 % and slice-and-shift agree with brute membership on {n} points', dis == 0)

# 4. or-trick unsoundness, v2-plan 'v1 epsilon propagation UNSOUND': [0,1]*(2,3)
p1 = v1.MultiInterval(0, 1) * v1.MultiInterval(2, 3, start_closed=False, end_closed=False)
p2 = v2.MultiInterval(0, 1) * v2.MultiInterval(2, 3, start_closed=False, end_closed=False)
print('v1 [0,1]*(2,3) =', p1, ' v2 =', p2)
check('v2 [0,1]*(2,3) contains 0 (0*2.5 = 0)', 0 in p2)
print('    v1 contains 0:', 0 in p1)

# 5. complex: v1 pow of negative base
try: print('v1 [-2,-1]**0.5 ->', v1.MultiInterval(-2,-1) ** 0.5)
except Exception as e: print('v1 [-2,-1]**0.5 raises', type(e).__name__, e)
try: print('v2 [-2,-1]**0.5 ->', v2.MultiInterval(-2,-1) ** F(1,2))
except Exception as e: print('v2 [-2,-1]**0.5 raises', type(e).__name__, e)
print('fails', fails, '(1 expected: the sabotage line)')
