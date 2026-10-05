"""
reference oracles for the arithmetic tests

independent of `multiinterval.applicator` and `multiinterval.ops` on purpose: everything here is derived from
the pointwise table (v2-plan.md "domain and semantics"), one pair of points at a time, so it can
check the applicator's shape-then-attainment algorithm instead of restating it.

the table, for x in A and y in B, with +-inf as ordinary points:

    add   x + y; inf + (-inf) is indeterminate, inf + y = inf otherwise
    sub   x - y = x + (-y)
    mul   x * y; 0 * +-inf is indeterminate, +-inf * nonzero is the signed infinity
    div   y != 0: finite / +-inf = 0, +-inf / +-inf indeterminate, +-inf / finite = signed infinity.
          y == 0: 0 / 0 indeterminate; x != 0 gives sign(x) * -inf if the piece of B holding 0 has
          points below 0 and sign(x) * inf if it has points above (none for a degenerate [0])
    reciprocal(A) = div([1], A);  neg, pos, abs pointwise
    pow   n >= 1: x ** n, (+-inf) ** n by parity; n == 0: [1] for a non-empty A;
          n < 0: reciprocal(pow(A, -n)), the pole direction read from the image pow(A, -n)
    mod   python's floor-mod x - y * floor(x / y), the result taking the divisor's sign. +-inf mod y
          and x mod 0 have no value; finite x mod inf is x for x >= 0 and inf for x < 0, and
          x mod -inf is x for x <= 0 and -inf for x > 0 (D8, python's own values)

an indeterminate pair contributes nothing. `attained` decides exactly whether a value is produced
by some defined pair: finite floats are read as the Fraction they denote, +-inf symbolically.
"""
import math
import operator
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from multiinterval.cuts import normalize_value
from multiinterval.kernel import Cuts
from multiinterval.kernel import contains_point
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces

INF = math.inf
OPS = ('add', 'sub', 'mul', 'div', 'reciprocal', 'neg', 'pos', 'abs', 'pow')
BINARY = ('add', 'sub', 'mul', 'div')
# mod is not in OPS: the M6 property tests iterate OPS over `multiinterval.ops`, mod has its own module
MOD_OPS = ('mod',)
UNARY = ('reciprocal', 'neg', 'pos', 'abs')
ONE: Cuts = normalize([piece(1, 1)])


# NUMBERS

def _inf(x) -> bool:
    return x == INF or x == -INF


def _finite_float(x) -> bool:
    return isinstance(x, float) and not _inf(x)


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def _exact(x):
    """a finite float becomes the Fraction it denotes; int, Fraction and +-inf pass through"""
    if _finite_float(x):
        return normalize_value(Fraction(x))
    return normalize_value(x)


def _exact_cuts(cuts: Cuts) -> Cuts:
    return normalize(piece(_exact(lo), _exact(hi), lc, hc) for lo, lc, hi, hc in pieces(cuts))


def _negated(cuts: Cuts) -> Cuts:
    return normalize(piece(-hi, -lo, hc, lc) for lo, lc, hi, hc in pieces(cuts))


def _quotient(x, y):
    """finite x / finite nonzero y: exact unless a float is involved"""
    if isinstance(x, float) or isinstance(y, float):
        return _once(operator.truediv, x, y)
    return normalize_value(Fraction(x) / Fraction(y))


def _once(fn, x, y):
    """
    fn(x, y) on finite x, y where a float is involved, rounded once as the library rounds it: two
    floats by python's own float op, and a mixed pair computed exactly, then rounded to nearest
    (python would round the exact operand first, and `Fraction(1, 3) / 2.75` lands one ulp below
    the correctly rounded quotient)

    >>> Fraction(1, 3) / 2.75, _once(operator.truediv, Fraction(1, 3), 2.75)
    (0.1212121212121212, 0.12121212121212122)
    """
    if isinstance(x, float) and isinstance(y, float):
        return fn(x, y)
    value = fn(Fraction(x), Fraction(y))
    try:
        return float(value)
    except OverflowError:
        return INF if value > 0 else -INF


def _zero_piece(b: Optional[Cuts]) -> Tuple:
    """(lo, hi) of the piece of b that holds 0, which decides the direction of a pole"""
    if b is None:
        raise ValueError('a pole needs the denominator set to know its direction')
    for lo, lc, hi, hc in pieces(b):
        if (lo < 0 < hi) or (lo == 0 and lc) or (hi == 0 and hc):
            return lo, hi
    raise ValueError(f'0 is not a point of the denominator {b!r}')


# POINTWISE

def pointwise(op: str, x, y=None, a: Optional[Cuts] = None, b: Optional[Cuts] = None) -> list:
    """
    the values one pair contributes: [] if it is indeterminate, two values for a pole in a piece
    that crosses zero. `a`, `b` are the operand sets (needed for pole directions); for 'pow', `y`
    is the int exponent and `a` the base set

    >>> pointwise('mul', 0, math.inf)
    []
    >>> pointwise('div', Fraction(1), math.inf)
    [0]
    >>> pointwise('div', 1, 0, b=normalize([piece(-1, 1)]))
    [-inf, inf]
    """
    if op == 'add':
        return _add(x, y)
    if op == 'sub':
        return _add(x, -y)
    if op == 'mul':
        return _mul(x, y)
    if op == 'div':
        return _div(x, y, b)
    if op == 'reciprocal':
        return _div(1, x, a)
    if op == 'neg':
        return [normalize_value(-x)]
    if op == 'pos':
        return [x]
    if op == 'abs':
        return [abs(x)]
    if op == 'pow':
        return _pow(x, y, a)
    if op == 'mod':
        return _mod(x, y)
    raise ValueError(f'unknown op {op!r}')


def _add(x, y) -> list:
    if _inf(x) and _inf(y) and x != y:
        return []
    if _inf(x):
        return [x]
    if _inf(y):
        return [y]
    if isinstance(x, float) or isinstance(y, float):
        return [normalize_value(_once(operator.add, x, y))]
    return [normalize_value(x + y)]


def _mul(x, y) -> list:
    if _inf(x) or _inf(y):
        if x == 0 or y == 0:
            return []
        return [_sign(x) * _sign(y) * INF]
    if isinstance(x, float) or isinstance(y, float):
        return [normalize_value(_once(operator.mul, x, y))]
    return [normalize_value(x * y)]


def _div(x, y, b) -> list:
    if y == 0:
        if x == 0:
            return []
        lo, hi = _zero_piece(b)
        out = []
        if lo < 0:
            out.append(-_sign(x) * INF)
        if hi > 0:
            out.append(_sign(x) * INF)
        return out
    if _inf(y):
        if _inf(x):
            return []
        return [0.0 if isinstance(x, float) else 0]
    if _inf(x):
        return [_sign(x) * _sign(y) * INF]
    return [_quotient(x, y)]


def _mod(x, y) -> list:
    if _inf(x) or y == 0:
        return []
    if y == INF:
        return [x if x >= 0 else INF]
    if y == -INF:
        return [x if x <= 0 else -INF]
    if isinstance(x, float) or isinstance(y, float):
        return [normalize_value(x % y)]
    x, y = Fraction(x), Fraction(y)
    return [normalize_value(x - y * math.floor(x / y))]


def _check_exponent(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f'pow takes an int exponent, got {n!r}')


def _power(x, n: int):
    """x ** n for n >= 1, +-inf by parity"""
    if _inf(x):
        return INF if n % 2 == 0 else x
    return normalize_value(x ** n)


def _pow(x, n, a) -> list:
    _check_exponent(n)
    if n >= 1:
        return [_power(x, n)]
    if n == 0:
        return [1]
    if a is None:
        raise ValueError('a negative exponent needs the base set for the pole direction')
    if _finite_float(x) and x != 0:
        # python's float value is x ** n, one rounding; `1 / x ** -n` rounds twice and can miss the nearest
        # double (0.6 ** -2), as the library did until M14-breadth (2026-10-02)
        try:
            return [normalize_value(x ** n)]
        except OverflowError:  # x ** -n underflows: the pole's side decides, below
            pass
    return _div(1, _power(x, -n), power_image(a, -n))


def power_image(a: Cuts, n: int) -> Cuts:
    """
    {x ** n : x in a} for n >= 1, exactly: x ** n is continuous on the extended reals and monotone
    on each side of 0, so each piece maps to at most two pieces with its own flags
    """
    _check_exponent(n)
    if n < 1:
        raise ValueError(n)
    out = []
    for lo, lc, hi, hc in pieces(_exact_cuts(a)):
        if n % 2 or lo >= 0:
            out.append(piece(_power(lo, n), _power(hi, n), lc, hc))
        elif hi <= 0:
            out.append(piece(_power(hi, n), _power(lo, n), hc, lc))
        else:
            out.append(piece(0, _power(hi, n), True, hc))
            out.append(piece(0, _power(lo, n), True, lc))
    return normalize(out)


# REAL INTERVALS
# a real interval is (lo, lo_closed, hi, hi_closed) with an infinite end always open: the finite
# part of a piece. the infinite points of an operand are kept separately, as flags

RealInterval = Tuple


def _real(lo, lc, hi, hc) -> RealInterval:
    return lo, lc and not _inf(lo), hi, hc and not _inf(hi)


def _nonempty(i: RealInterval) -> bool:
    lo, lc, hi, hc = i
    return lo < hi or (lo == hi and lc and hc)


def _meet(i: RealInterval, j: RealInterval) -> RealInterval:
    (a, ac, b, bc), (c, cc, d, dc) = i, j
    lo, lc = (a, ac) if a > c else (c, cc) if c > a else (a, ac and cc)
    hi, hc = (b, bc) if b < d else (d, dc) if d < b else (b, bc and dc)
    return lo, lc, hi, hc


def _point(i: RealInterval):
    """an exact point of a non-empty real interval"""
    lo, _, hi, _ = i
    if lo == hi:
        return lo
    if not _inf(lo) and not _inf(hi):
        return normalize_value((Fraction(lo) + Fraction(hi)) / 2)
    if not _inf(hi):
        return hi - 1
    if not _inf(lo):
        return lo + 1
    return 0


def _image(i: RealInterval, f, increasing: bool) -> RealInterval:
    """the image of a real interval under a continuous monotone f, given f at (the limits of) its ends"""
    lo, lc, hi, hc = i
    flo, fhi = f(lo), f(hi)
    if increasing:
        return _real(flo, lc, fhi, hc)
    return _real(fhi, hc, flo, lc)


def _shifted(v, j: RealInterval) -> RealInterval:
    """{v - y : y in j}, v finite"""
    return _image(j, lambda y: -y if _inf(y) else v - y, increasing=False)


def _scaled(v, j: RealInterval) -> RealInterval:
    """{v * y : y in j}, v finite nonzero"""
    return _image(j, lambda y: _sign(v) * y if _inf(y) else v * y, increasing=v > 0)


def _scaled_reciprocal(v, j: RealInterval, positive: bool) -> RealInterval:
    """{v / y : y in j}, v finite nonzero, j inside (0, inf) if `positive` else inside (-inf, 0)"""
    def f(y):
        if y == 0:
            return _sign(v) * (INF if positive else -INF)
        if _inf(y):
            return 0
        return _quotient(v, y)
    return _image(j, f, increasing=v < 0)


# OPERAND SETS

class _Set(NamedTuple):
    """an exact operand, split into its finite real pieces and its infinite points"""
    cuts: Cuts
    finite: Tuple[RealInterval, ...]
    neg_inf: bool
    pos_inf: bool


# the same operands are decomposed over and over; the cached _Set is shared, so it is all tuples
@lru_cache(maxsize=4096)
def _decompose(cuts: Cuts) -> _Set:
    cuts = _exact_cuts(cuts)
    finite, neg_inf, pos_inf = [], False, False
    for lo, lc, hi, hc in pieces(cuts):
        neg_inf |= lo == -INF and lc
        pos_inf |= hi == INF and hc
        real = _real(lo, lc, hi, hc)
        if _nonempty(real):
            finite.append(real)
    return _Set(cuts, tuple(finite), neg_inf, pos_inf)


# extended ranges (lo, lo_closed, hi, hi_closed) that the case analysis asks for
ANY = (-INF, True, INF, True)
FINITE = (-INF, False, INF, False)
ZERO = (0, True, 0, True)
POS = (0, False, INF, True)  # positive, inf included
NEG = (-INF, True, 0, False)
FIN_POS = (0, False, INF, False)
FIN_NEG = (-INF, False, 0, False)
PINF = (INF, True, INF, True)
NINF = (-INF, True, -INF, True)
NOT_NINF = (-INF, False, INF, True)
NOT_PINF = (-INF, True, INF, False)


def _find(s: _Set, lo, lc, hi, hc):
    """an exact point of s inside the extended range, or None"""
    if lo == -INF and lc and s.neg_inf:
        return -INF
    if hi == INF and hc and s.pos_inf:
        return INF
    real = _real(lo, lc, hi, hc)
    for i in s.finite:
        m = _meet(i, real)
        if _nonempty(m):
            return _point(m)
    return None


def _both(a: _Set, range_a, b: _Set, range_b):
    x = _find(a, *range_a)
    y = None if x is None else _find(b, *range_b)
    return None if y is None else (x, y)


def _first(*candidates):
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


# WITNESSES
# each returns an exact defined pair that produces v, or None if there is none. the finite cases
# solve for x: x must lie in A and in the preimage of v under y -> op(., y) over a piece of B

def _witness_add(v, a: _Set, b: _Set):
    if v == INF:
        return _first(_both(a, PINF, b, NOT_NINF), _both(a, NOT_NINF, b, PINF))
    if v == -INF:
        return _first(_both(a, NINF, b, NOT_PINF), _both(a, NOT_PINF, b, NINF))
    for i in a.finite:
        for j in b.finite:
            m = _meet(i, _shifted(v, j))
            if _nonempty(m):
                x = _point(m)
                return x, normalize_value(v - x)
    return None


def _witness_mul(v, a: _Set, b: _Set):
    if v == INF:
        return _first(_both(a, PINF, b, POS), _both(a, NINF, b, NEG),
                      _both(a, POS, b, PINF), _both(a, NEG, b, NINF))
    if v == -INF:
        return _first(_both(a, PINF, b, NEG), _both(a, NINF, b, POS),
                      _both(a, POS, b, NINF), _both(a, NEG, b, PINF))
    if v == 0:
        return _first(_both(a, ZERO, b, FINITE), _both(a, FINITE, b, ZERO))
    for i in a.finite:
        for j in b.finite:
            for half, positive in ((FIN_POS, True), (FIN_NEG, False)):
                jh = _meet(j, half)
                if not _nonempty(jh):
                    continue
                m = _meet(i, _scaled_reciprocal(v, jh, positive))
                if _nonempty(m):
                    x = _point(m)
                    return x, _quotient(v, x)
    return None


def _witness_div(v, a: _Set, b: _Set):
    if _inf(v):
        # +-inf / finite of the right sign, or a pole: x != 0 over a 0 in b, direction from its piece
        same, other = (FIN_POS, FIN_NEG) if v > 0 else (FIN_NEG, FIN_POS)
        found = _first(_both(a, PINF, b, same), _both(a, NINF, b, other))
        if found is not None or not contains_point(b.cuts, 0):
            return found
        lo, hi = _zero_piece(b.cuts)
        up = POS if v > 0 else NEG  # x of this sign over points above 0 gives v
        down = NEG if v > 0 else POS
        x = _first(_find(a, *up) if hi > 0 else None, _find(a, *down) if lo < 0 else None)
        return None if x is None else (x, 0)
    if v == 0:
        return _first(_both(a, ZERO, b, POS), _both(a, ZERO, b, NEG),
                      _both(a, FINITE, b, PINF), _both(a, FINITE, b, NINF))
    for i in a.finite:
        for j in b.finite:
            for half in (FIN_POS, FIN_NEG):
                jh = _meet(j, half)
                if not _nonempty(jh):
                    continue
                m = _meet(i, _scaled(v, jh))
                if _nonempty(m):
                    x = _point(m)
                    return x, _quotient(x, v)
    return None


def _witness_mod(v, a: _Set, b: _Set):
    """
    brute force, independent of multiinterval.modulo: the quotients k = floor(x / y) are enumerated, not
    solved for. x mod y == v with y > 0 iff 0 <= v < y and x = v + k * y for an integer k (mirrored for
    y < 0), so for each piece of B beyond v this looks for a k with x in A
    """
    if v == INF:
        return _both(a, FIN_NEG, b, PINF)
    if v == -INF:
        return _both(a, FIN_POS, b, NINF)
    if v >= 0 and b.pos_inf and contains_point(a.cuts, v):
        return v, INF
    if v <= 0 and b.neg_inf and contains_point(a.cuts, v):
        return v, -INF
    for j in b.finite:
        for half, beyond in ((FIN_POS, (v, False, INF, False)), (FIN_NEG, (-INF, False, v, False))):
            if (v < 0 and half is FIN_POS) or (v > 0 and half is FIN_NEG):
                continue
            jv = _meet(_meet(j, half), beyond)
            if not _nonempty(jv):
                continue
            for i in a.finite:
                found = _witness_mod_k(v, i, jv)
                if found is not None:
                    return found
    return None


def _witness_mod_k(v, i: RealInterval, jv: RealInterval):
    """x in i and y in jv (one sign, every |y| > |v|) with x = v + k * y for an integer k"""
    lo, lc, hi, hc = i
    if _nonempty(_meet(i, (v, True, v, True))):
        return v, _point(jv)
    s = 1 if jv[2] > 0 else -1
    near = jv[0] if s > 0 else -jv[2]  # inf |y|
    far = jv[2] if s > 0 else -jv[0]  # sup |y|
    if near == 0:
        # jv touches 0, so v == 0 and any nonzero x of i is a multiple of some small enough y
        for half in (FIN_POS, FIN_NEG):
            m = _meet(i, half)
            if _nonempty(m):
                x = _point(m)
                n = 1 if _inf(far) else math.floor(abs(x) / far) + 1
                return x, normalize_value(Fraction(x) / (_sign(x) * s * n))
        return None
    if _inf(lo) or _inf(hi):
        # an unbounded piece holds v + k * y for every |k| large enough, of the right sign
        y = _point(jv)
        end, d = (lo, 1) if _inf(hi) else (hi, -1)
        start = 0 if _inf(end) else abs(Fraction(end) - v) / abs(y)
        k = d * s * (math.ceil(start) + 1)
        return normalize_value(v + k * y), y
    bound = math.ceil(max(abs(Fraction(lo) - v), abs(Fraction(hi) - v)) / near) + 1
    for k in range(-bound, bound + 1):
        if k == 0:
            continue
        ends = [((Fraction(lo) - v) / k, lc), ((Fraction(hi) - v) / k, hc)]
        (p, pc), (q, qc) = sorted(ends, key=lambda e: e[0])
        m = _meet((p, pc, q, qc), jv)
        if _nonempty(m):
            y = _point(m)
            return normalize_value(v + k * y), y
    return None


def witness(op: str, v, a: Cuts, b: Optional[Cuts] = None) -> Optional[tuple]:
    """
    an exact pair `(x, y)` (or `(x,)` for a unary op) of operand points that produces v, or None.
    not available for 'pow', whose roots may be irrational; `attained` decides pow on its image
    """
    v = _exact(v)
    sa = _decompose(a)
    if op == 'mod':
        found = _witness_mod(v, sa, _decompose(b))
        if found is not None:
            # self-check: the witness is a defined pair of the operands that produces v
            x, y = found
            assert contains_point(a, x) or contains_point(sa.cuts, x), (v, found)
            assert contains_point(b, y) or contains_point(_exact_cuts(b), y), (v, found)
            assert _mod(x, y) == [v], (v, found, _mod(x, y))
        return found
    if op in BINARY:
        sb = _decompose(b)
        if op == 'add':
            return _witness_add(v, sa, sb)
        if op == 'sub':
            found = _witness_add(v, sa, _decompose(_negated(sb.cuts)))
            return None if found is None else (found[0], normalize_value(-found[1]))
        if op == 'mul':
            return _witness_mul(v, sa, sb)
        return _witness_div(v, sa, sb)
    if op == 'reciprocal':
        found = _witness_div(v, _decompose(ONE), sa)
        return None if found is None else (found[1],)
    if op in ('neg', 'pos', 'abs'):
        candidates = {'neg': [-v], 'pos': [v], 'abs': [v, -v] if v >= 0 else []}[op]
        for x in candidates:
            if contains_point(sa.cuts, x):
                return (x,)
        return None
    raise ValueError(f'no witness for op {op!r}')


def attained(op: str, v, a: Cuts, b=None) -> bool:
    """
    exact: is v produced by some defined pair of `a` (and `b`)? for 'pow', `b` is the int exponent

    >>> attained('reciprocal', -math.inf, normalize([piece(-1, 0)]))
    True
    >>> attained('mul', 0, normalize([piece(math.inf, math.inf)]), normalize([piece(0, 1, False)]))
    False
    """
    if op not in OPS + MOD_OPS:
        raise ValueError(f'unknown op {op!r}')
    if not a or (op in BINARY + MOD_OPS and not b):
        return False
    if op == 'pow':
        _check_exponent(b)
        v = _exact(v)
        if b >= 1:
            return contains_point(power_image(a, b), v)
        if b == 0:
            return v == 1
        return witness('div', v, ONE, power_image(a, -b)) is not None
    return witness(op, v, a, b) is not None


# SAMPLING

def sample(cuts: Cuts, n: int, rng) -> list:
    """
    n points of the set: closed endpoints (+-inf included) often, points just inside an end often,
    the rest spread over the piece. a piece with no finite float endpoint gives exact points
    """
    ps = list(pieces(cuts))
    if not ps:
        return []
    return [_sample_piece(rng.choice(ps), rng) for _ in range(n)]


def _in_piece(p, lo, lc, hi, hc) -> bool:
    return (lo < p < hi) or (p == lo and lc) or (p == hi and hc)


def _sample_piece(pc, rng):
    lo, lc, hi, hc = pc
    if lo == hi:
        return lo
    kind = rng.choice(('lo', 'hi', 'near_lo', 'near_hi', 'inside', 'inside'))
    if kind == 'lo' and lc:
        return lo
    if kind == 'hi' and hc:
        return hi
    flo = None if _inf(lo) else Fraction(lo)
    fhi = None if _inf(hi) else Fraction(hi)
    if kind in ('lo', 'near_lo'):
        p = _near(flo, fhi, 1, rng)
    elif kind in ('hi', 'near_hi'):
        p = _near(fhi, flo, -1, rng)
    else:
        u = Fraction(rng.randrange(1, 1000), 1000)
        if flo is not None and fhi is not None:
            p = flo + (fhi - flo) * u
        elif flo is not None:
            p = flo + Fraction(rng.randrange(1, 10 ** 6), 1000)
        elif fhi is not None:
            p = fhi - Fraction(rng.randrange(1, 10 ** 6), 1000)
        else:
            p = Fraction(rng.randrange(-10 ** 6, 10 ** 6), 1000)
    p = normalize_value(p)
    if _finite_float(lo) or _finite_float(hi):
        try:
            as_float = float(p)
        except OverflowError:  # a point past the doubles stays exact (m14b-open, 2026-10-05)
            return p
        if _in_piece(as_float, *pc):
            return normalize_value(as_float)
    return p


def _near(end, other, direction: int, rng):
    """an exact point just inside `end` (None: the infinite end) heading towards `other`"""
    k = rng.randrange(0, 40)
    if end is None:
        base = other if other is not None else 0
        return base - direction * Fraction(2) ** k
    if other is None:
        return end + direction * Fraction(1, 2 ** k)
    return end + (other - end) / 2 ** (k + 1)


__all__ = ['OPS', 'BINARY', 'UNARY', 'MOD_OPS', 'ONE', 'pointwise', 'attained', 'witness', 'power_image', 'sample']
