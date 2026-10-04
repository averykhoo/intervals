from common import *
from probe_arith_helpers import run
rng = random.Random(21)
n = 0
for _ in range(20000):
    ivs = [rand_interval(rng) for _ in range(rng.randint(1, 3))]
    mi = MI(*ivs)
    r = run(lambda: mi.reciprocal())
    if r[0] != 'ok':
        print(mi, r, '| v2', mi_to_v2(mi).reciprocal())
        n += 1
        if n > 4:
            break
for i in [I(0, False, 0, True), I(0, True, 1, True), I(-1, False, 0, False), I(-INF, True, 0, False)]:
    print(i, run(lambda: i.reciprocal()), run(lambda: MI(i).reciprocal()))
