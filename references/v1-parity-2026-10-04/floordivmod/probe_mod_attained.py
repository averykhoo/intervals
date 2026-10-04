"""v1 _mod_attained(value, A, B) vs v2 any(modulo._attained(value, x, y) for boxes) vs the exact oracle"""
from common import *
import sys
from intervals import modulo
SAB = '--sabotage' in sys.argv
rng = random.Random(55)
n = same = 0; bad = []
for _ in range(600):
    a = rand_pieces(rng, lo=0, hi=12)
    b = rand_pieces(rng, lo=0, hi=8)
    if b[0][0] == 0 and b[0][1]: b[0] = (b[0][0], False) + b[0][2:]
    b = canon([p for p in b if nonempty(*p)])
    if not b: continue
    vals = [Fraction(rng.randint(0, 40), 4) for _ in range(6)] + [lo for lo, *_ in a] + [p[2] for p in b]
    for v in vals:
        n += 1
        o1 = v1._mod_attained(v, a, b)
        o2 = any(modulo._attained(v, x, y) for x in a for y in b)
        if SAB: o2 = any(modulo._attained(v + Fraction(1, 4), x, y) for x in a for y in b)
        orc = attained_mod(v, a, b)
        if o1 == o2 == orc: same += 1
        else: bad.append((show(a), show(b), v, o1, o2, orc))
print(f'values {n}, all three agree {same}, disagree {len(bad)}')
for x in bad[:8]: print(x)
