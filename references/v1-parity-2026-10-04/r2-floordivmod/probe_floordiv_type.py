"""TYPE (int vs float) of the endpoints of `//` results, v1 vs v2, against python's scalar rule.

v1 floors each endpoint with math.floor -> always int (finite). v2: exact quotient, integers made float
iff an operand has a finite float (modulo.floordiv `as_float`). python scalars: x // y is float iff an
operand is float. run: timeout 120 <python> .scratch/v1-parity/r2-floordivmod/probe_floordiv_type.py [--sabotage]
--sabotage: expect v2's types to equal v1's (a deliberately wrong expectation; must be caught).
"""
import sys, os, math, random, warnings
from fractions import Fraction
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'archive', 'v1')]
import multi_interval as v1
import intervals as v2
from intervals import kernel
warnings.simplefilter('ignore')
SAB = '--sabotage' in sys.argv


def v1_ends(m):
    return [p for p, _ in m.endpoints]


def v2_ends(m):
    out = []
    for lo, _, hi, _ in kernel.pieces(m.cuts):
        out += [lo, hi]
    return out


def tname(v):
    return type(v).__name__


def finite(vs):
    return [v for v in vs if not (isinstance(v, float) and math.isinf(v))]


def run(f):
    try:
        return 'ok', f()
    except Exception as e:  # noqa
        return 'err', f'{type(e).__name__}: {e}'


def is_float_op(*xs):
    return any(isinstance(x, float) and math.isfinite(x) for x in xs)


stats = dict(cases=0, v1_all_int=0, v1_err=0, v2_float_iff_float_op=0, v2_type_mismatch=0,
             v2_nonintegral=0, value_mismatch=0, python_point_type_agree=0, python_point_cases=0,
             sabotage_caught=0)
examples = []


def check(desc, a1, b1, a2, b2, ops_vals, scalar_pair=None):
    """a1 // b1 in v1, a2 // b2 in v2; ops_vals: raw operand numbers (for the float rule)"""
    stats['cases'] += 1
    s1, r1 = run(lambda: a1 // b1)
    s2, r2 = run(lambda: a2 // b2)
    if s2 != 'ok':
        examples.append(('V2 ERR', desc, r2))
        return
    e2 = finite(v2_ends(r2))
    want_float = is_float_op(*ops_vals)
    types2 = {tname(v) for v in e2}
    if e2:
        if types2 == ({'float'} if want_float else {'int'}):
            stats['v2_float_iff_float_op'] += 1
        else:
            stats['v2_type_mismatch'] += 1
            examples.append(('V2 TYPE', desc, types2, want_float))
        if any(v != math.floor(v) for v in e2):
            stats['v2_nonintegral'] += 1
            examples.append(('V2 NONINT', desc, e2))
    if s1 != 'ok':
        stats['v1_err'] += 1
        return
    e1 = finite(v1_ends(r1))
    types1 = {tname(v) for v in e1}
    if types1 <= {'int'}:
        stats['v1_all_int'] += 1
    else:
        examples.append(('V1 TYPE', desc, types1))
    if SAB and e1 and e2 and types1 != types2:
        stats['sabotage_caught'] += 1
    # values: v2's integer set must sit inside v1's hull range when both computed (types aside)
    if e1 and e2 and (min(e2) < min(e1) - 1 or max(e2) > max(e1) + 1):
        stats['value_mismatch'] += 1
        examples.append(('VALUE', desc, e1, e2))
    if scalar_pair is not None:
        x, y = scalar_pair
        stats['python_point_cases'] += 1
        py = x // y
        if e2 and len(set(e2)) == 1 and type(e2[0]) is type(py) and e2[0] == py:
            stats['python_point_type_agree'] += 1
        else:
            examples.append(('PY POINT', desc, py, tname(py), e2))


# hand-picked
F = Fraction
hand = [
    ((1.5, 3.5), 1), ((1, 7), 2.0), ((1.0, 7.0), 2), ((1, 7), 2), ((F(3, 2), F(7, 2)), 1), ((1, 7), F(1, 2)),
    ((2.0, 2.0), 1), ((-0.5, -0.5), 1), ((0.3, 0.3), 1), ((-0.0, 0.0), 1), ((1, 1), 0.001), ((-3.5, 2.5), 0.5),
    ((1e300, 1e300), 3), ((0.5, 2000.5), 1), ((1.0, math.inf), 2), ((1, math.inf), 2), ((1, 2), -0.5),
]
for (lo, hi), m in hand:
    ec = not (isinstance(hi, float) and math.isinf(hi))
    check(f'[{lo!r}, {hi!r}] // {m!r}', v1.MultiInterval(lo, hi, end_closed=ec), m,
          v2.MultiInterval.from_pieces([(lo, hi, True, ec)]), m, (lo, hi, m),
          scalar_pair=(lo, m) if lo == hi else None)
# set divisor and scalar on the left
for (a, b) in [((1, 7), (2.0, 3.0)), ((1.0, 7.0), (2, 3)), ((1, 7), (2, 3)), ((F(1, 3), 7), (F(1, 2), 3))]:
    check(f'{a} // {b}', v1.MultiInterval(*a), v1.MultiInterval(*b), v2.MultiInterval(*a), v2.MultiInterval(*b),
          a + b)
for m, (lo, hi) in [(7.0, (2, 3)), (7, (2.0, 3.0)), (7, (2, 3)), (F(7, 2), (1, 2))]:
    stats['cases'] += 0
    check(f'{m!r} // [{lo!r}, {hi!r}]', m, v1.MultiInterval(lo, hi), m, v2.MultiInterval(lo, hi), (m, lo, hi))
# divmod first part, Outward class
for cls in (v2.MultiInterval, v2.OutwardMultiInterval):
    q, _ = divmod(cls(1.5, 7.5), 2)
    print(cls.__name__, 'divmod([1.5,7.5], 2)[0] end types:', {tname(v) for v in v2_ends(q)}, q)
    q, _ = divmod(cls(1, 7), 2)
    print(cls.__name__, 'divmod([1,7], 2)[0] end types:', {tname(v) for v in v2_ends(q)}, q)
print('v1 divmod:', run(lambda: divmod(v1.MultiInterval(1.5, 7.5), 2)))

# seeded sweep
rng = random.Random(20261004)
kinds = ['int', 'float', 'frac']


def num(kind, lo, hi):
    if kind == 'int':
        return rng.randint(lo, hi)
    if kind == 'float':
        return rng.randint(lo * 8, hi * 8) / 8 + (rng.choice([0, 0, 0.1]))
    return Fraction(rng.randint(lo * 6, hi * 6), rng.choice([1, 2, 3, 6]))


for _ in range(400):
    ka, kb, km = rng.choice(kinds), rng.choice(kinds), rng.choice(kinds)
    a, b = sorted([num(ka, -20, 20), num(kb, -20, 20)])
    m = num(km, 1, 9) * rng.choice([1, -1])
    if m == 0:
        continue
    point = rng.random() < 0.2
    if point:
        b = a
    check(f'[{a!r}, {b!r}] // {m!r}', v1.MultiInterval(a, b), m, v2.MultiInterval(a, b), m, (a, b, m),
          scalar_pair=(a, m) if point else None)
    # set divisor of one sign
    c, d = sorted([num(kb, 1, 9), num(km, 1, 9)])
    check(f'[{a!r}, {b!r}] // [{c!r}, {d!r}]', v1.MultiInterval(a, b), v1.MultiInterval(c, d),
          v2.MultiInterval(a, b), v2.MultiInterval(c, d), (a, b, c, d))

for k, v in stats.items():
    print(f'{k}: {v}')
seen = {}
for e in examples:
    seen.setdefault(e[0], []).append(e)
for k, es in seen.items():
    print(f'--- {k}: {len(es)}')
    for e in es[:6]:
        print('   ', e[1:])
if SAB:
    print('SABOTAGE', 'CAUGHT' if stats['sabotage_caught'] else 'NOT CAUGHT')
