"""find the random large case where v2 // differs from python's float //; decide exactly. same seed as probe_floordiv_type_large.py"""
import sys, os, math, random, warnings
from fractions import Fraction
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT); sys.path[:0] = [ROOT]
import intervals as v2
from intervals import kernel
warnings.simplefilter('ignore')
rng = random.Random(7)
for _ in range(300):
    x = rng.uniform(1, 10) * 10.0 ** rng.randint(15, 300); y = rng.choice([3, 7, 0.3, 1.7, 11.0])
    r2 = next(iter(kernel.pieces((v2.MultiInterval(x) // y).cuts)))[0]
    if r2 != x // y:
        ex = math.floor(Fraction(x) / Fraction(y))
        print(repr(x), repr(y), 'v2', repr(r2), 'python', repr(x // y), 'exact', ex,
              '|v2-exact|', abs(Fraction(r2) - ex), '|py-exact|', abs(Fraction(x // y) - ex))
