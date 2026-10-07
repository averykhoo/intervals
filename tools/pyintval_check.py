"""
pyintval against `multiinterval.ieee1788` on random binary64 operands, a manual tool (never run by CI)

pyintval (https://github.com/marciogameiro/pyintval, MIT) is an independent 1788 set-based library over
binary64: C++, `+ - * / sqrt fma` correctly rounded, the elementary functions on CORE-MATH widened one ulp per
end on purpose, its own decorations (`references/python-1788-libraries-2026-10-07.md`). it is not installed by
any extra: `pip install pyintval==0.3.0` into the env first.

    $PY tools/pyintval_check.py [--n 2000] [--seed 0] [--only sin,add] [--flavour bare|decorated|both]

every op both libraries have (`OPS`, 1788's names) runs on the same random operands in each (`gen_bounds`:
specials and their neighbours, small dyadics, log-uniform, any bit pattern; empty, entire, half-bounded,
points, narrow boxes), and each pair of answers is classified. a set: `equal`, `ours tighter` (inside theirs,
split by the ulps theirs is wider on the worse side: `<=2` is their one-ulp widening of a correctly rounded
end), `theirs tighter`, `crossed`. a decoration: `dec equal`, `dec ours stronger`, `dec theirs stronger`. a
number or a boolean: `equal` or `differ`. `ours raised`, `theirs raised`: one side raising.

pyintval is loose by design and departs from 1788 in places (the survey's findings), so a difference alone
says little. the referee, independent of both libraries, decides what it can:
* `referee`: MPFR (gmpy2, rounded down and up) at points of the box: its ends, infinite ones as limits, 0
  inside an operand, the domain's edges (`EDGES`, `OPEN_EDGES`), sin's and cos's extrema (`extrema`, pi to
  `PI_BITS`). every value must be inside ours (else `UNSOUND`); where the op is monotone on the box
  (`explainable`), each of our ends must be theirs or the extreme value outward (`ours the tightest`).
  cancelMinus and cancelPlus exactly by 1788.1 4.5.3 (`cancel_exact`); an empty mulRev exactly
  (`mul_rev_meets`). a result that is not 1788's exact answer is `set NOT 1788's exact answer`
* `expected_decoration`: 1788's decoration where it is decidable here (step functions off their jumps, tan
  off its poles, atan2 touching its cut, pow from a base 0 or below 0): ours must be it (else `dec NOT
  1788's`), and a difference from theirs is then theirs (`ours 1788's`)

the leads: `UNSOUND`, any `NOT 1788's`, `theirs tighter`, `crossed`, `differ`, a raise, a decoration that
differs undecided, and `ours tighter` undecided by more than two ulps or on an op in `TIGHT` (pyintval
rounds correctly there, so the two should be equal). each lead's operands are printed as a repro (at most
`LEAD_LIMIT` per op and class) and all are written to `.scratch/pyintval-check/leads-<stamp>.tsv`; the exit
status is 1 if there is any. the seconds each side took are printed too. what a run found, and the families
decided by hand: `references/python-1788-libraries-2026-10-07.md`.
"""
import argparse
import math
import random
import struct
import sys
import time
import warnings
from collections import Counter
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyintval as iv  # noqa: E402

from multiinterval import ieee1788 as I  # noqa: E402

INF = math.inf
RANK = {'trv': 0, 'def': 1, 'dac': 2, 'com': 3}
LEAD_LIMIT = 5  # repros printed per (op, flavour, class)

# 1788 name: (operand kinds, ours, theirs, result kind). kinds: 'I' interval, 'n' an int exponent, 'r' a real
OPS = {}


def _op(name, kinds, theirs, result='set', ours=None):
    OPS[name] = (kinds, ours or I.NAMES[name], theirs, result)


for _name, _theirs in [
        ('pos', lambda x: +x), ('neg', lambda x: -x), ('recip', iv.recip), ('sqr', iv.sqr), ('sqrt', iv.sqrt),
        ('abs', iv.abs), ('exp', iv.exp), ('exp2', iv.exp2), ('exp10', iv.exp10), ('expm1', iv.expm1),
        ('log', iv.log), ('log2', iv.log2), ('log10', iv.log10), ('logp1', iv.log1p), ('sin', iv.sin),
        ('cos', iv.cos), ('tan', iv.tan), ('asin', iv.asin), ('acos', iv.acos), ('atan', iv.atan),
        ('sinh', iv.sinh), ('cosh', iv.cosh), ('tanh', iv.tanh), ('asinh', iv.asinh), ('acosh', iv.acosh),
        ('atanh', iv.atanh), ('cbrt', iv.cbrt), ('sign', iv.sign), ('ceil', iv.ceil), ('floor', iv.floor),
        ('trunc', iv.trunc), ('roundTiesToEven', iv.round), ('roundTiesToAway', iv.round_ties_to_away),
        ('sqrRev', iv.sqr_rev), ('absRev', iv.abs_rev)]:
    _op(_name, 'I', _theirs)
for _name, _theirs in [
        ('add', lambda x, y: x + y), ('sub', lambda x, y: x - y), ('mul', lambda x, y: x * y),
        ('div', lambda x, y: x / y), ('min', iv.min), ('max', iv.max), ('atan2', iv.atan2), ('hypot', iv.hypot),
        ('pow', iv.pow), ('intersection', iv.intersection), ('convexHull', iv.hull),
        ('cancelMinus', iv.cancel_minus), ('cancelPlus', iv.cancel_plus), ('mulRev', iv.mul_rev)]:
    _op(_name, 'II', _theirs)
_op('sqrRevBin', 'II', iv.sqr_rev, ours=I.sqr_rev)
_op('absRevBin', 'II', iv.abs_rev, ours=I.abs_rev)
_op('mulRevTen', 'III', iv.mul_rev, ours=I.mul_rev)
_op('fma', 'III', iv.fma)
_op('pown', 'In', iv.pown)


def _bare(x):
    return x.interval if isinstance(x, iv.DecoratedInterval) else x


for _name, _attr in [('inf', 'lo'), ('sup', 'hi'), ('mid', 'mid'), ('wid', 'wid'), ('rad', 'rad'),
                     ('mag', 'mag'), ('mig', 'mig')]:
    _op(_name, 'I', lambda x, a=_attr: getattr(_bare(x), a), 'num')
for _name, _theirs in [
        ('equal', lambda x, y: x == y), ('subset', lambda x, y: x.subset(y)),
        ('interior', lambda x, y: x.is_interior_to(y)), ('disjoint', lambda x, y: x.is_disjoint(y)),
        ('less', lambda x, y: x.less(y)), ('strictLess', lambda x, y: x.strict_less(y)),
        ('precedes', lambda x, y: x.precedes(y)), ('strictPrecedes', lambda x, y: x.strict_precedes(y))]:
    _op(_name, 'II', _theirs, 'bool')
for _name, _attr in [('isEmpty', 'is_empty'), ('isEntire', 'is_entire'), ('isCommonInterval', 'is_common'),
                     ('isSingleton', 'is_singleton')]:
    _op(_name, 'I', lambda x, a=_attr: getattr(x, a), 'bool')
_op('isMember', 'rI', lambda m, x: x.contains(m), 'bool')

# pyintval's correctly rounded ops (its README), and the ones exact or decided by those: ours should equal theirs
TIGHT = {'pos', 'neg', 'add', 'sub', 'mul', 'div', 'recip', 'sqr', 'sqrt', 'fma', 'abs', 'min', 'max', 'sign',
         'ceil', 'floor', 'trunc', 'roundTiesToEven', 'roundTiesToAway', 'intersection', 'convexHull',
         'cancelMinus', 'cancelPlus', 'sqrRev', 'sqrRevBin', 'absRev', 'absRevBin', 'mulRev', 'mulRevTen'}


# -- operands --

def _specials():
    out = [0.0, 1.0, 2.0, 0.5, 3.0, 10.0, 0.1, 1e-300, 1e300, 2.0 ** 53, 2.0 ** 52 + 0.5, 709.782712893384, 710.0,
           -745.1332191019411, 1.7976931348623157e308, 5e-324, 2.2250738585072014e-308, 1e22, 2.0 ** 1023]
    out += [k * math.pi / 4 for k in range(1, 9)] + [math.pi * 2 ** 20, math.pi * 1e15]
    return out + [-v for v in out]


SPECIALS = _specials()


def _bits_double(rng):
    while True:
        x = struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64)))[0]
        if math.isfinite(x):
            return x


def gen_double(rng):
    """a finite double: a special or its neighbours, a small dyadic, log-uniform, or any bit pattern"""
    u = rng.random()
    if u < 0.3:
        x = rng.choice(SPECIALS)
        for _ in range(rng.choice([0, 0, 1, 2, 3])):
            x = math.nextafter(x, rng.choice([INF, -INF]))
        return x if math.isfinite(x) else 0.0
    if u < 0.55:
        return rng.randint(-40, 40) / 2.0 ** rng.randint(0, 3)
    if u < 0.95:
        return rng.choice([1, -1]) * rng.random() * 2.0 ** rng.randint(-60, 60)
    return _bits_double(rng)


def gen_bounds(rng):
    """(lo, hi) of a 1788 interval, None for the empty set"""
    u = rng.random()
    if u < 0.05:
        return None
    if u < 0.08:
        return -INF, INF
    if u < 0.16:
        a = gen_double(rng)
        return (-INF, a) if rng.random() < 0.5 else (a, INF)
    if u < 0.26:
        a = gen_double(rng)
        return a, a
    a, b = gen_double(rng), gen_double(rng)
    if rng.random() < 0.3:  # a narrow one, so sin, tan and the steps see more than "the whole range"
        b = a + (abs(a) * 2.0 ** -rng.randint(1, 50) if a else 2.0 ** -rng.randint(1, 50))
    return min(a, b), max(a, b)


def make_pair(bounds, decoration):
    """the same interval in each library; `decoration` None is bare, else it is capped by newDec"""
    ours = I.empty() if bounds is None else I.Interval(*bounds)
    theirs = iv.Interval.empty() if bounds is None else iv.Interval(*bounds)
    if decoration is None:
        return ours, theirs
    best = I.decoration_part(I.new_dec(ours)).value
    d = decoration if RANK[decoration] <= RANK[best] else best
    return I.set_dec(ours, d), iv.DecoratedInterval.from_parts(theirs, d)


def gen_args(rng, kinds, decorated):
    ours, theirs, shown = [], [], []
    for k in kinds:
        if k == 'I':
            bounds = gen_bounds(rng)
            d = rng.choice(['com', 'com', 'dac', 'def', 'trv']) if decorated else None
            a, b = make_pair(bounds, d)
            ours.append(a), theirs.append(b), shown.append(str(a))
        elif k == 'n':
            n = rng.choice([-3, -2, -1, 0, 1, 2, 3, 4, 5, rng.randint(-40, 40)])
            ours.append(n), theirs.append(n), shown.append(str(n))
        else:
            m = gen_double(rng)
            ours.append(m), theirs.append(m), shown.append(repr(m))
    return ours, theirs, ' '.join(shown)


# -- comparison --

def _ordinal(x):
    """the double's place in the total order of doubles (consecutive doubles differ by 1)"""
    n = struct.unpack('<q', struct.pack('<d', x))[0]
    return n if n >= 0 else -(n & 0x7FFFFFFFFFFFFFFF)


def ulps(a, b):
    if a == b:
        return 0
    if math.isinf(a) or math.isinf(b):
        return math.inf
    return abs(_ordinal(a) - _ordinal(b))


def ours_set(r):
    dec = r.decoration
    return (None if I.is_empty(r) else (I.inf(r), I.sup(r))), (None if dec is None else dec.value)


def theirs_set(r):
    if isinstance(r, iv.DecoratedInterval):
        if r.is_nai:
            return 'nai', 'ill'
        return theirs_set(r.interval)[0], r.decoration
    return (None if r.is_empty else (r.lo, r.hi)), None


def compare_sets(o, t):
    """the class of ours `o` against theirs `t`, each (lo, hi) or None for empty, and the wider side's ulps"""
    if o == t or (o and t and o[0] == t[0] and o[1] == t[1]):
        return 'equal', 0
    if o is None:
        return 'ours tighter', math.inf
    if t is None:
        return 'theirs tighter', 0
    if t[0] <= o[0] and o[1] <= t[1]:
        return 'ours tighter', max(ulps(o[0], t[0]), ulps(o[1], t[1]))
    if o[0] <= t[0] and t[1] <= o[1]:
        return 'theirs tighter', 0
    return 'crossed', 0


# -- the referee: MPFR (gmpy2) at points of the box, independent of both libraries --

def _gmpy2_oracle():
    import gmpy2
    contexts = [gmpy2.context(precision=53, emin=-1073, emax=1024, subnormalize=True, round=r)
                for r in (gmpy2.RoundDown, gmpy2.RoundUp)]
    unary = {n: getattr(gmpy2, n) for n in ('exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'sin', 'cos',
                                            'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh',
                                            'acosh', 'atanh', 'cbrt', 'sqrt')}
    unary.update(logp1=gmpy2.log1p, sqr=lambda x: x * x, recip=lambda x: 1 / x)
    fs = {n: (f, 1) for n, f in unary.items()}
    fs.update(add=(lambda x, y: x + y, 2), sub=(lambda x, y: x - y, 2), mul=(lambda x, y: x * y, 2),
              div=(lambda x, y: x / y, 2), atan2=(gmpy2.atan2, 2), hypot=(gmpy2.hypot, 2),
              pow=(lambda x, y: x ** y, 2), pown=(lambda x, n: x ** n, 2), fma=(gmpy2.fma, 3))

    def value(name, point):
        """the real value at a point of the domain, rounded down and up to doubles (MPFR rounds correctly)"""
        f = fs[name][0]
        out = []
        for c in contexts:
            with c:
                out.append(float(f(*(gmpy2.mpfr(p) if isinstance(p, float) else p for p in point))))
        return out
    return value, gmpy2


ORACLE, _gmpy2 = _gmpy2_oracle()
DOMAIN = {  # the real domain; an op not here is defined on every real
    'log': lambda x: x > 0, 'log2': lambda x: x > 0, 'log10': lambda x: x > 0, 'logp1': lambda x: x > -1,
    'sqrt': lambda x: x >= 0, 'asin': lambda x: -1 <= x <= 1, 'acos': lambda x: -1 <= x <= 1,
    'acosh': lambda x: x >= 1, 'atanh': lambda x: -1 < x < 1, 'recip': lambda x: x != 0,
    'div': lambda x, y: y != 0, 'atan2': lambda y, x: x != 0 or y != 0,
    'pow': lambda x, y: x > 0 or (x == 0 and y > 0), 'pown': lambda x, n: x != 0 or n >= 0}
# where MPFR's value at a point outside the domain is the one-sided limit from inside it (log(+0) is -inf,
# pow(+0, -1) is +inf); at an infinite end MPFR's value is the limit too
LIMIT = dict(DOMAIN, log=lambda x: x >= 0, log2=lambda x: x >= 0, log10=lambda x: x >= 0,
             logp1=lambda x: x >= -1, atanh=lambda x: -1 <= x <= 1, pow=lambda x, y: x > 0 or (x == 0 and y != 0))
# a domain's ends, a point of the box when inside an operand: {op: {operand index: [edges]}}
EDGES = {'log': {0: [0.0]}, 'log2': {0: [0.0]}, 'log10': {0: [0.0]}, 'logp1': {0: [-1.0]}, 'sqrt': {0: [0.0]},
         'asin': {0: [-1.0, 1.0]}, 'acos': {0: [-1.0, 1.0]}, 'acosh': {0: [1.0]}, 'atanh': {0: [-1.0, 1.0]},
         'pow': {0: [0.0]}}
# the open edges `LIMIT` adds, and the side the domain is on: a limit there only if the operand reaches past it
OPEN_EDGES = {'log': {0: {0.0: 1}}, 'log2': {0: {0.0: 1}}, 'log10': {0: {0.0: 1}}, 'logp1': {0: {-1.0: 1}},
              'atanh': {0: {-1.0: 1, 1.0: -1}}, 'pow': {0: {0.0: 1}}}
ORACLE_OPS = {'exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'logp1', 'sin', 'cos', 'tan', 'asin', 'acos',
              'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'cbrt', 'sqrt', 'sqr', 'recip', 'add', 'sub',
              'mul', 'div', 'atan2', 'hypot', 'pow', 'pown', 'fma'}
# monotone in each operand on either side of 0 and of each domain edge, so the image's ends are values or
# limits at the points; sin and cos on a bounded box with their extrema inside (`extrema`), tan on one
# with no pole, atan2 off its cut (`explainable`)
MONOTONE = ORACLE_OPS - {'sin', 'cos', 'tan', 'atan2'}


def points(name, oargs, kinds):
    """the box's points: per operand its ends (an infinite one included), 0 where it straddles 0, and the
    op's domain edges inside it; None if an operand is empty"""
    axes = []
    for i, (a, k) in enumerate(zip(oargs, kinds)):
        if k == 'n':
            axes.append([a])
            continue
        if I.is_empty(a):
            return None
        lo, hi = I.inf(a) + 0.0, I.sup(a) + 0.0
        inside = [0.0] + EDGES.get(name, {}).get(i, [])
        axes.append(sorted({lo, hi} | {e for e in inside if lo < e < hi}))
    out = [()]
    for axis in axes:
        out = [p + (v,) for p in out for v in axis]
    return out


def explainable(name, oargs):
    """whether the image's ends are among the values and limits at `points`"""
    if name in MONOTONE:
        return True
    if name in ('sin', 'cos', 'tan'):
        lo, hi = I.inf(oargs[0]), I.sup(oargs[0])
        return math.isfinite(lo) and math.isfinite(hi) and (name != 'tan' or pole_free(lo, hi))
    if name == 'atan2':  # the angles of a box: at its corners, or on the axes through the origin when the
        # box holds it (`points` has the 0s); not when it crosses the cut (x < 0, y from below 0 to 0 or above)
        (ylo, yhi), (xlo, xhi) = [(I.inf(a), I.sup(a)) for a in oargs]
        return not (xlo < 0 and ylo < 0 <= yhi)
    return False


PI_BITS = 2300  # pi to more bits than a double's exponent range, so x / pi keeps 1000 bits for any double


def _turns(v, offset, period):
    """floor((v - offset) / period), offset and period in units of pi"""
    with _gmpy2.context(precision=PI_BITS):
        pi = _gmpy2.const_pi()
        return _gmpy2.floor((_gmpy2.mpfr(v) - offset * pi) / (period * pi))


def pole_free(lo, hi):
    """no pole of tan, (k + 1/2) pi, in [lo, hi] (a double is never a pole)"""
    return _turns(lo, 0.5, 1) == _turns(hi, 0.5, 1)


def extrema(name, lo, hi):
    """sin's and cos's values at their maxima and minima inside [lo, hi], as `ORACLE` gives values"""
    peaks = {'sin': ((0.5, 1.0), (-0.5, -1.0)), 'cos': ((0, 1.0), (1, -1.0))}[name]
    return [(v, v) for offset, v in peaks if _turns(lo, offset, 2) != _turns(hi, offset, 2)]


def jump_free(name, lo, hi):
    """no point of [lo, hi] where the step function jumps (exact)"""
    lo, hi = Fraction(lo), Fraction(hi)
    if name == 'sign':
        return not lo <= 0 <= hi
    if name in ('floor', 'ceil'):
        return math.ceil(lo) > hi
    if name == 'trunc':
        return math.ceil(lo) > hi or (math.ceil(lo) == 0 and math.floor(hi) == 0)
    half = Fraction(1, 2)  # the rounds jump at the odd halves
    return math.ceil(lo - half) + half > hi


STEPS = {'sign', 'floor', 'ceil', 'trunc', 'roundTiesToEven', 'roundTiesToAway'}


def expected_decoration(name, kinds, oargs, os_):
    """1788's decoration of the result, where this tool can decide it on its own (None elsewhere): the step
    functions off their jumps, tan off its poles, atan2 on a box touching its cut from above, pow on a base
    from 0 with a positive exponent, pow on a base reaching below 0. the operands' decorations cap it"""
    ivs = [a for a, k in zip(oargs, kinds) if k == 'I']
    if not ivs or any(I.is_empty(a) for a in ivs) or ivs[0].decoration is None:
        return None
    given = min((I.decoration_part(a).value for a in ivs), key=RANK.get)
    cap = lambda d: d if RANK[d] < RANK[given] else given
    bounded = os_ is not None and math.isfinite(os_[0]) and math.isfinite(os_[1])
    box = [(I.inf(a), I.sup(a)) for a in ivs]
    if name in STEPS or name == 'tan':
        lo, hi = box[0]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        if jump_free(name, lo, hi) if name in STEPS else pole_free(lo, hi):
            return cap('com' if bounded else 'dac')
        return None
    if name == 'atan2':  # defined (no origin), continuous on the box but not at its points on the cut
        (ylo, yhi), (xlo, xhi) = box
        return cap('dac') if ylo == 0 and xhi < 0 else None
    if name == 'pow':
        (xlo, xhi), (ylo, yhi) = box
        if xlo < 0:
            return 'trv'  # the box is not inside pow's domain
        if xlo == 0 and ylo > 0:  # defined, and continuous at every point of the box
            whole = all(math.isfinite(v) for b in box for v in b)
            return cap('com' if whole and bounded else 'dac')
    return None


def _outward(q, up):
    """the double just below (or above) the exact rational q"""
    with _gmpy2.context(precision=53, emin=-1073, emax=1024, subnormalize=True,
                        round=_gmpy2.RoundUp if up else _gmpy2.RoundDown):
        return float(_gmpy2.mpfr(_gmpy2.mpq(q.numerator, q.denominator)))


def _bounds(a):
    return None if I.is_empty(a) else (I.inf(a), I.sup(a))


def cancel_exact(name, oargs):
    """1788's cancelMinus (1788.1 4.5.3), exactly in Fractions then outward: (lo, hi), None for empty"""
    x, y = map(_bounds, oargs)
    if name == 'cancelPlus' and y is not None:
        y = (-y[1], -y[0])
    finite = lambda v: v is None or all(math.isfinite(e) for e in v)
    if x is None and finite(y):
        return None
    if x is not None and y is not None and finite(x) and finite(y):
        (x1, x2), (y1, y2) = [tuple(map(Fraction, v)) for v in (x, y)]
        if y2 - y1 <= x2 - x1:
            return _outward(x1 - y1, False), _outward(x2 - y2, True)
    return -INF, INF


def mul_rev_meets(oargs):
    """whether some b in B and t in X have b * t in C (mulRev's answer is empty exactly when not): the
    products' range lies between its corner values, exact in Fractions; a corner 0 * inf is 0, which the
    zero end times any finite point of the other operand attains"""
    boxes = list(map(_bounds, oargs))
    if any(v is None for v in boxes):
        return False
    b, c = boxes[0], boxes[1]
    x = boxes[2] if len(boxes) == 3 else (-INF, INF)
    products = []
    for u in b:
        for v in x:
            products.append(Fraction(u) * Fraction(v) if math.isfinite(u) and math.isfinite(v)
                            else math.copysign(INF, u) * math.copysign(1, v) if u and v else 0)
    lo, hi = min(products), max(products)
    return max(lo, c[0]) <= min(hi, c[1])


def referee(name, kinds, oargs, os_, ts):
    """(sound, tightest). sound: ours holds the value at every real point of the box (an empty ours: there
    is none) and every limit. tightest: each of our ends is theirs or, where the op is `explainable`, the
    outward value of the extreme point; for an empty ours, no point is even a limit (None: not decided)"""
    if name in ('cancelMinus', 'cancelPlus'):
        exp = cancel_exact(name, oargs)
        if exp is None or os_ is None:
            return exp is None or os_ is not None, os_ == exp
        return os_[0] <= exp[0] and exp[1] <= os_[1], os_ == exp
    if name in ('mulRev', 'mulRevTen'):
        meets = mul_rev_meets(oargs)
        if os_ is None:
            return not meets, not meets
        return True, False if not meets else None
    if name not in ORACLE_OPS:
        return True, None
    pts = points(name, oargs, kinds)
    if pts is None:
        return True, None
    dom, lim = DOMAIN.get(name), LIMIT.get(name)
    real = [p for p in pts if all(math.isfinite(v) for v in p) and (dom is None or dom(*p))]
    def reached(p):  # a real point, an infinite end's limit, or an open edge's limit from inside the box
        if dom is None or dom(*p):
            return True
        if not lim(*p):
            return False
        for i, edges in OPEN_EDGES.get(name, {}).items():
            side = edges.get(p[i])
            if side and not (I.sup(oargs[i]) > p[i] if side > 0 else I.inf(oargs[i]) < p[i]):
                return False
        return True
    values = [v for v in (ORACLE(name, p) for p in pts if reached(p)) if not math.isnan(v[0])]
    if name in ('sin', 'cos') and all(math.isfinite(v) for v in pts[0] + pts[-1]):
        values += extrema(name, pts[0][0], pts[-1][0])
    if os_ is None:
        return not real, (not values) if name in DOMAIN else None
    sound = all(os_[0] <= dn and up <= os_[1] for dn, up in values)
    if not values or not explainable(name, oargs):
        return sound, None
    lo_ok = ts is not None and os_[0] == ts[0] or os_[0] == min(v[0] for v in values)
    hi_ok = ts is not None and os_[1] == ts[1] or os_[1] == max(v[1] for v in values)
    return sound, lo_ok and hi_ok


def classify(name, kinds, result, oargs, ours, theirs):
    """[class, ...]: the set's class, the referee's and, decorated, the decoration's"""
    if result != 'set':
        same = ours == theirs or (isinstance(ours, float) and isinstance(theirs, float)
                                  and math.isnan(ours) and math.isnan(theirs))
        return ['equal' if same else 'differ']
    (os_, od), (ts, td) = ours_set(ours), theirs_set(theirs)
    if ts == 'nai':
        return ['theirs nai']
    cls, wider = compare_sets(os_, ts)
    sound, tightest = referee(name, kinds, oargs, os_, ts if ts != 'nai' else None)
    if cls == 'ours tighter':
        cls = 'ours tighter, ' + ('<=2 ulps' if wider <= 2 else '>2 ulps')
        if tightest:
            cls += ', ours the tightest'
    out = [cls] if sound else [cls, 'UNSOUND: ours misses part of the true answer']
    if tightest is False and name in ('cancelMinus', 'cancelPlus', 'mulRev', 'mulRevTen'):
        out.append("set NOT 1788's exact answer")
    if od is not None:
        dec = 'dec equal' if od == td else 'dec ours stronger' if RANK[od] > RANK[td] else 'dec theirs stronger'
        expected = expected_decoration(name, kinds, oargs, os_)
        if expected is not None and od != expected:
            out.append(f"dec NOT 1788's: {expected} expected")
        elif expected is not None and dec != 'dec equal':
            dec += ", ours 1788's"
        out.append(dec)
    return out


def is_lead(name, cls):
    if cls in ('equal', 'dec equal') or cls.endswith(('ours the tightest', "ours 1788's")):
        return False
    if cls == 'ours tighter, <=2 ulps':
        return name in TIGHT
    return True


def call(f, args):
    try:
        return f(*args), None
    except Exception as e:  # noqa: BLE001: either side raising is a class of its own
        return None, f'{type(e).__name__}: {e}'


def run(n, seed, only, flavours):
    rng = random.Random(seed)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = ROOT / '.scratch' / 'pyintval-check'
    out_dir.mkdir(parents=True, exist_ok=True)
    leads_path = out_dir / f'leads-{stamp}.tsv'
    counts = defaultdict(Counter)
    seconds = defaultdict(lambda: [0.0, 0.0])
    shown = Counter()
    n_leads = 0
    with open(leads_path, 'w', encoding='utf-8') as leads:
        leads.write('op\tflavour\tclass\toperands\tours\ttheirs\n')
        for name, (kinds, ours_f, theirs_f, result) in OPS.items():
            if only and name not in only:
                continue
            for flavour in flavours:
                if flavour == 'decorated' and result != 'set':
                    continue  # 1788's numbers and booleans are of the interval part
                key = (name, flavour)
                for _ in range(n):
                    oargs, targs, text = gen_args(rng, kinds, flavour == 'decorated')
                    t0 = time.perf_counter()
                    o, oerr = call(ours_f, oargs)
                    t1 = time.perf_counter()
                    t, terr = call(theirs_f, targs)
                    t2 = time.perf_counter()
                    seconds[key][0] += t1 - t0
                    seconds[key][1] += t2 - t1
                    if oerr or terr:
                        classes = ['both raised'] if oerr and terr else ['ours raised' if oerr else 'theirs raised']
                    else:
                        classes = classify(name, kinds, result, oargs, o, t)
                    for cls in classes:
                        counts[key][cls] += 1
                        if not is_lead(name, cls) or cls == 'both raised':
                            continue
                        n_leads += 1
                        ours_text = oerr or (str(o) if result == 'set' else repr(o))
                        theirs_text = terr or (str(t) if result == 'set' else repr(t))
                        leads.write(f'{name}\t{flavour}\t{cls}\t{text}\t{ours_text}\t{theirs_text}\n')
                        if shown[key, cls] < LEAD_LIMIT:
                            shown[key, cls] += 1
                            print(f'  LEAD {name} [{flavour}] {cls}: {name} {text}\n'
                                  f'      ours   {ours_text}\n      theirs {theirs_text}')
                leads.flush()
                c = counts[key]
                lead_n = sum(v for cls, v in c.items() if is_lead(name, cls) and cls != 'both raised')
                summary = ', '.join(f'{cls} {v}' for cls, v in sorted(c.items()))
                o_s, t_s = seconds[key]
                print(f'{name:16} {flavour:9} leads {lead_n:5}  | {summary} | ours {o_s:.2f}s theirs {t_s:.3f}s',
                      flush=True)
    total_o = sum(s[0] for s in seconds.values())
    total_t = sum(s[1] for s in seconds.values())
    print(f'\n{n_leads} lead(s) in {leads_path.relative_to(ROOT)}; seed {seed}, {n} per op and flavour; '
          f'time ours {total_o:.1f}s, theirs {total_t:.2f}s')
    return 1 if n_leads else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--n', type=int, default=2000, help='operand tuples per op and flavour')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only', default='', help='comma-separated 1788 names (`OPS`)')
    parser.add_argument('--flavour', choices=['bare', 'decorated', 'both'], default='both')
    a = parser.parse_args(argv)
    only = {s for s in a.only.split(',') if s}
    unknown = only - OPS.keys()
    if unknown:
        parser.error(f'not in OPS: {sorted(unknown)}')
    flavours = ['bare', 'decorated'] if a.flavour == 'both' else [a.flavour]
    warnings.simplefilter('ignore')  # PossiblyUndefinedOperationWarning: 1788 signals it, pyintval does not
    return run(a.n, a.seed, only, flavours)


if __name__ == '__main__':
    sys.exit(main())
