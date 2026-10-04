"""stand-in for linux x86-64 np.longdouble (64-bit significand): exact value, float() rounds,
as_integer_ratio exact, arithmetic/comparison exact (as a long double +0 / compare with a double is exact).
runs v1 accessors and v2 accessors on the same ends. --sabotage flips an expectation."""
import sys, numbers, random
from fractions import Fraction
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2

class LD:
    def __init__(self, q): self.q = Fraction(q)
    def _o(self, o): return o.q if isinstance(o, LD) else o
    def __float__(self): return float(self.q)
    def as_integer_ratio(self): return self.q.numerator, self.q.denominator
    def __lt__(s, o): return s.q < s._o(o)
    def __le__(s, o): return s.q <= s._o(o)
    def __gt__(s, o): return s.q > s._o(o)
    def __ge__(s, o): return s.q >= s._o(o)
    def __eq__(s, o): return s.q == s._o(o)
    def __hash__(s): return hash(s.q)
    def __add__(s, o): return LD(s.q + s._o(o))
    __radd__ = __add__
    def __sub__(s, o): return LD(s.q - s._o(o))
    def __rsub__(s, o): return LD(s._o(o) - s.q)
    def __neg__(s): return LD(-s.q)
    def __repr__(s): return f'LD({s.q})'
numbers.Real.register(LD)

def val(x):
    return x.q if isinstance(x, LD) else Fraction(x) if not (isinstance(x, float) and x in (float('inf'), -float('inf'))) else x

sab = '--sabotage' in sys.argv
rng = random.Random(17)
same = diff = 0
cases = [(LD(1 + Fraction(1, 2**60)), LD(2 - Fraction(1, 2**62))), (LD(Fraction(-1, 3)*0+Fraction(-6148914691236517205, 2**64)), 0.5)]
for _ in range(300):
    a = Fraction(rng.randint(-2**20, 2**20), 2**rng.randint(0, 63))
    b = a + Fraction(rng.randint(0, 2**20), 2**rng.randint(0, 63))
    cases.append((LD(a), LD(b)))
for lo, hi in cases:
    A1 = v1.MultiInterval(lo, hi)
    A2 = v2.MultiInterval(lo, hi)
    got1 = (val(A1.infimum), val(A1.supremum), A1.is_degenerate)
    got2 = (val(A2.inf), val(A2.sup), A2.is_degenerate)
    want = (val(lo), val(hi), val(lo) == val(hi))
    if sab: want = (want[0] + Fraction(1, 2**64),) + want[1:]
    ok = got1 == want and got2 == want
    same += ok; diff += (not ok)
    if not ok and diff <= 3: print('DIFF', lo, hi, got1, got2, want)
print('types v1', type(A1.infimum).__name__, 'v2', type(A2.inf).__name__)
print(f'same={same} diff={diff}')
