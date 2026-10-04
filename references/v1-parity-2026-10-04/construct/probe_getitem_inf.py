"""classify the nonzero-bound slice mismatches of probe_getitem.py: all of them an inf start or -inf stop?"""
from common import *
import random
exec(open(ROOT + '/.scratch/v1-parity/construct/probe_getitem.py', encoding='utf-8').read().split("random.seed(2024)")[0].split('from common import *')[1])
random.seed(77)
NZ = [None, -inf, inf] + [v for v in VALS if v != 0] + [10, -10]
kinds = {}
for _ in range(800):
    A1 = rand_v1(); A2 = v1_to_v2(A1)
    lo, hi = random.choice(NZ), random.choice(NZ)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    o1 = outcome(lambda: A1[lo:hi]); o2 = outcome(lambda: A2[lo:hi])
    if not (o1[0] == o2[0] == 'ok' and v1_to_v2(o1[1]) == o2[1]):
        k = ('lo=inf' if lo == inf else '') + ('hi=-inf' if hi == -inf else '') or 'OTHER'
        kinds[k] = kinds.get(k, 0) + 1
        if k == 'OTHER': print('OTHER', str(A1), lo, hi, o1, o2)
print(kinds)
