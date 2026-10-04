"""workaround for v1's int-typed // results in v2: lift float operands to the Fraction they denote first.
run: timeout 120 <python> .scratch/v1-parity/r2-floordivmod/probe_floordiv_int_workaround.py"""
import sys, os, math, random, warnings
from fractions import Fraction
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT); sys.path[:0] = [ROOT, os.path.join(ROOT, 'archive', 'v1')]
import multi_interval as v1, intervals as v2
from intervals import kernel
warnings.simplefilter('ignore')
def ends2(m): return [e for lo, _, hi, _ in kernel.pieces(m.cuts) for e in (lo, hi)]
def lift(m): return v2.MultiInterval.from_pieces([(Fraction(lo), Fraction(hi), lc, hc) for lo, lc, hi, hc in kernel.pieces(m.cuts)])
rng = random.Random(11); n = ok_type = ok_val = 0; bad = []
for _ in range(300):
    a = rng.randint(-160, 160) / 8 + rng.choice([0, 0.1]); b = a + rng.randint(0, 80) / 8
    m = rng.choice([0.5, 0.3, 2.0, 1.7, -0.25, 3]) 
    A = v2.MultiInterval(a, b)
    plain = A // m; w = lift(A) // Fraction(m)
    n += 1
    ok_type += all(type(e) is int for e in ends2(w))
    ok_val += plain == w and set(ends2(plain)) == set(ends2(w))   # same set, only the type differs
    if not (plain == w): bad.append((a, b, m, plain, w))
print(f'{n} cases: workaround all-int {ok_type}; same set as plain v2 // {ok_val}; differing {len(bad)}', bad[:3])
x = 1e17; print('1e17 // 3 via lift:', ends2(lift(v2.MultiInterval(x)) // 3), 'exact floor', math.floor(Fraction(x) / 3))
print('sanity (must be False):', (v2.MultiInterval(1.5, 3.5) // 1) == v2.MultiInterval(1, 4))
