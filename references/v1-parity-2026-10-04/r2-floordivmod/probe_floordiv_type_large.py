"""large-magnitude float //: v1's int vs the exact floor of the doubles' quotient vs v2's float vs python.
run: timeout 120 <python> .scratch/v1-parity/r2-floordivmod/probe_floordiv_type_large.py"""
import sys, os, math, random, warnings
from fractions import Fraction
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT); sys.path[:0] = [ROOT, os.path.join(ROOT, 'archive', 'v1')]
import multi_interval as v1, intervals as v2
from intervals import kernel
warnings.simplefilter('ignore')
def ends2(m): return [e for lo, _, hi, _ in kernel.pieces(m.cuts) for e in (lo, hi)]
for x, y in [(1e17, 3), (1e300, 3), (1e300, 7.0), (2.0**60, 3), (9007199254740993, 3.0), (1e17, 0.3)]:
    exact = math.floor(Fraction(x) / Fraction(y))
    r1 = (v1.MultiInterval(x) // y).endpoints[0][0]
    r2 = ends2(v2.MultiInterval(x) // y)
    py = x // y
    print(f'{x!r} // {y!r}: exact {exact}, v1 {r1!r} ({type(r1).__name__}, v1==exact {r1 == exact}), '
          f'v2 {r2[0]!r} ({type(r2[0]).__name__}, v2==nearest(exact) {r2[0] == float(exact)}), python {py!r} (v2==python {r2[0] == py})')
rng = random.Random(7); n = 0; v1_wrong = 0; v2_not_nearest = 0; v2_ne_py = 0
for _ in range(300):
    x = rng.uniform(1, 10) * 10.0 ** rng.randint(15, 300); y = rng.choice([3, 7, 0.3, 1.7, 11.0])
    exact = math.floor(Fraction(x) / Fraction(y))
    r1 = (v1.MultiInterval(x) // y).endpoints[0][0]; r2 = ends2(v2.MultiInterval(x) // y)[0]
    n += 1; v1_wrong += r1 != exact; v2_not_nearest += r2 != float(exact); v2_ne_py += r2 != x // y
print(f'random large: {n} cases; v1 int != exact floor: {v1_wrong}; v2 != nearest double of exact floor: {v2_not_nearest}; v2 != python x//y: {v2_ne_py}')
