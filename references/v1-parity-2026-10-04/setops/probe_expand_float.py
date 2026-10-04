"""expand with float rounding: v1 endpoint +- d in python floats vs v2 expand, nearest and outward classes"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random
rng = random.Random(5)
t = Tally('expand float vs v1'); t3 = Tally('expand Fraction/float mix vs v1')
for _ in range(400):
    a = round(rng.uniform(-5, 5), rng.randint(1, 3)); b = a + round(rng.uniform(0, 3), 2); d = round(rng.uniform(0, 2), rng.randint(1, 3))
    r1 = M1(a, b).expand(d); r2 = M2(a, b).expand(d)
    t.check((a, b, d), same_set_v1_v2(r1, r2))
    fa = Fraction(rng.randint(-9, 9), rng.randint(1, 7))
    r1 = M1(fa, fa + 1).expand(d); r2 = M2(fa, fa + 1).expand(d)
    t3.check((fa, d), same_set_v1_v2(r1, r2))
t.report(); t3.report()
print('v1', M1(0.1).expand(0.2), ' v2', M2(0.1).expand(0.2), ' v2 outward', v2.OutwardMultiInterval(0.1).expand(0.2))
s = Tally('sabotage'); s.check('x', same_set_v1_v2(M1(0.1).expand(0.2), M2(0.1).expand(0.25))); assert s.report(0) == 1
