"""the v2 workaround for 3-arg pow, checked against python's pow on every point triple"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import random
from common import *
from probe_pow3 import points2, workaround  # noqa (re-runs probe_pow3's prints)
rng = random.Random(33)
ok = bad = 0
for _ in range(300):
    bs = rng.sample(range(-20, 40), rng.randint(1, 4))
    es = rng.sample(range(0, 12), rng.randint(1, 3))
    ms = rng.sample([m for m in range(-9, 15) if m != 0], rng.randint(1, 3))
    truth = sorted({pow(b, e, m) for b in bs for e in es for m in ms})
    w = workaround(points2(bs), points2(es), points2(ms))
    if pieces2(w) == [(t, t, True, True) for t in truth]:
        ok += 1
    else:
        bad += 1; print('BAD', bs, es, ms, w)
print('workaround ok', ok, 'bad', bad)
# sabotage
assert pieces2(workaround(points2([2]), points2([2]), points2([5]))) == [(4, 4, True, True)]
assert pieces2(workaround(points2([2]), points2([2]), points2([5]))) != [(3, 3, True, True)]
