"""for each `/` difference in the float sweep: which side's end equals the correctly rounded exact quotient of a corner?"""
import sys, math, random, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/arith', '.scratch/v1-parity/critic']
warnings.simplefilter('ignore')
from _fs_defs import *
rng = random.Random(4242)
tally = {'v2 nearest, v1 not': 0, 'v1 nearest, v2 not': 0, 'neither/other': 0}
ex = []
ops = ['+', '-', '*', '/']
for i in range(700):
    pa = rpieces(rng)
    for name in ops:
        pb = rpieces(rng, positive_only=(name == '/'))
        if name != '/': continue
        r1 = struct1(mk1(pa) / mk1(pb)); r2 = struct2(mk2(pa) / mk2(pb))
        if r1 == r2: continue
        corners = {float(F(x) / F(y)) for (a, b, _, _) in pa for x in (a, b) for (c, d, _, _) in pb for y in (c, d)}
        for t1, t2 in zip(r1, r2):
            for k in (0, 2):
                if t1[k] != t2[k]:
                    if t2[k] in corners and t1[k] not in corners: tally['v2 nearest, v1 not'] += 1
                    elif t1[k] in corners and t2[k] not in corners: tally['v1 nearest, v2 not'] += 1
                    else: tally['neither/other'] += 1; ex.append((t1[k], t2[k]))
print(tally, ex[:3])
assert tally['v2 nearest, v1 not'] > 0 and tally['v1 nearest, v2 not'] == 0
