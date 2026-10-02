"""
step functions over cut tuples: floor, ceil, trunc, round (ties to even or away from zero) and sign

each maps a set to the values it attains, which are points of a grid (the integers, or multiples of
`10 ** -ndigits` for `round(A, ndigits)`, or -1, 0, 1 for sign). a function's preimages are intervals
in the same order as their values, so a piece attains every grid value from the one just inside its
start to the one just inside its end:

    | f                  | just inside the start `[v` / `(v` | just inside the end `v]` / `v)` |
    |--------------------|-----------------------------------|---------------------------------|
    | floor              | floor(v) / floor(v)               | floor(v) / ceil(v) - 1          |
    | ceil               | ceil(v) / floor(v) + 1            | ceil(v) / ceil(v)               |
    | round, v a tie     | round(v) / floor(v) + 1           | round(v) / floor(v)             |
    | sign               | sign(v) / 1 if v >= 0 else -1     | sign(v) / -1 if v <= 0 else 1   |

and trunc is floor above 0 and ceil below it. up to `ENUMERATION_CAP` values are listed; past that,
or for a piece reaching ±inf, their hull is returned with a `HullWarning`. `f(±inf)` = ±inf (sign:
±1). a float end keeps its type, as python's float `floor` does not (`floor([2.5])` = `[2.0]`).

>>> from intervals.fmt import format_cuts, parse
>>> format_cuts(ceil(parse('(1, 3]')))
'{ [2] , [3] }'
>>> format_cuts(round_(parse('[1/2, 5/2]')))  # ties to even: 1/2 -> 0, 5/2 -> 2
'{ [0] , [1] , [2] }'
>>> format_cuts(round_(parse('(1/2, 5/2)')))
'{ [1] , [2] }'
>>> format_cuts(sign(parse('[-inf, 0]')))
'{ [-1] , [0] }'
"""
import math
from fractions import Fraction
from numbers import Integral
from typing import Callable
from typing import Optional
from typing import Tuple

from intervals import kernel
from intervals.applicator import split_pieces
from intervals.applicator import warn
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.kernel import Cuts
from intervals.rounding import is_float
from intervals.rounding import is_infinite
from intervals.rounding import round_piece

INF = math.inf

ENUMERATION_CAP = 1000

# (value just inside a start cut, value just inside an end cut), each from (v, closed), v finite
Rule = Tuple[Callable[[Fraction, bool], int], Callable[[Fraction, bool], int]]


def _is_tie(v: Fraction) -> bool:
    return v - math.floor(v) == Fraction(1, 2)


def _half_even(v: Fraction) -> int:
    return round(v)  # Fraction rounds ties to even


def _half_away(v: Fraction) -> int:
    n = math.floor(abs(v) + Fraction(1, 2))
    return n if v >= 0 else -n


def _sign(v) -> int:
    return (v > 0) - (v < 0)


def _round_rule(tie: Callable[[Fraction], int]) -> Rule:
    return (lambda v, closed: tie(v) if closed or not _is_tie(v) else math.floor(v) + 1,
            lambda v, closed: tie(v) if closed or not _is_tie(v) else math.floor(v))


RULES = {
    'floor': (lambda v, closed: math.floor(v),
              lambda v, closed: math.floor(v) if closed else math.ceil(v) - 1),
    'ceil': (lambda v, closed: math.ceil(v) if closed else math.floor(v) + 1,
             lambda v, closed: math.ceil(v)),
    'round': _round_rule(_half_even),
    'round_ties_away': _round_rule(_half_away),
    'sign': (lambda v, closed: _sign(v) if closed else 1 if v >= 0 else -1,
             lambda v, closed: _sign(v) if closed else -1 if v <= 0 else 1),
}


def step(name: str, a: Cuts, ndigits: Optional[int] = None, outward: bool = False) -> Cuts:
    """
    the values `f(x)` for x in a, f one of floor, ceil, trunc, round, round_ties_away, sign;
    `ndigits` (round and round_ties_away only) rounds to multiples of `10 ** -ndigits`

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(step('round', parse('[0.125, 0.135]'), ndigits=2))
    '{ [0.12] , [0.13] , [0.14] }'
    """
    if name not in RULES and name != 'trunc':
        raise ValueError(f'unknown step function {name!r}')
    if ndigits is not None:
        if name not in ('round', 'round_ties_away'):
            raise TypeError(f'{name}() takes no ndigits')
        if isinstance(ndigits, bool) or not isinstance(ndigits, Integral):
            raise TypeError(f'ndigits must be an int, got {type(ndigits).__name__}')
        ndigits = int(ndigits)  # numpy's ints too
    if not a:
        warn(EmptySetPropagationWarning, f'{name}: the operand is empty, so the result is empty')
        return kernel.EMPTY
    unit = Fraction(1) if ndigits is None else Fraction(10) ** -ndigits
    if name == 'trunc':  # floor above 0, ceil below
        tagged = [('floor' if q[0] >= 0 else 'ceil', q) for q in split_pieces(kernel.pieces(a), (0,))]
    else:
        tagged = [(name, p) for p in kernel.pieces(a)]
    # count counts distinct grid values: every f here is non-decreasing and the pieces are in order, so
    # a piece's values start at or after the last one listed, and a value two pieces share (`ceil` of
    # `[0, 1/2]` and `[7/10, 2]`, or trunc's 0 on each side of its split) is counted once (M14-breadth)
    out, count, hulled, listed = [], 0, False, None
    for rule, (lo, lo_closed, hi, hi_closed) in tagged:
        as_float = is_float(lo) or is_float(hi)
        if name != 'sign':  # f(±inf) = ±inf, and a piece reaching ±inf holds unboundedly many values
            for end, closed in ((lo, lo_closed), (hi, hi_closed)):
                if is_infinite(end) and closed:
                    out.append((end, True, end, True))
            if lo == hi and is_infinite(lo):
                continue
        first = -INF if lo == -INF and name != 'sign' else RULES[rule][0](_units(lo, unit), lo_closed)
        last = INF if hi == INF and name != 'sign' else RULES[rule][1](_units(hi, unit), hi_closed)
        start = first if listed is None else max(first, listed + 1)
        new = 0 if is_infinite(first) or is_infinite(last) else last - start + 1
        if is_infinite(first) or is_infinite(last) or count + new > ENUMERATION_CAP:
            hulled = True
            hull = (_value(first, unit), not is_infinite(first), _value(last, unit), not is_infinite(last))
            out.append(round_piece(hull, outward) if as_float else hull)
            continue
        count += max(new, 0)
        listed = last
        for n in range(first, last + 1):
            v = _value(n, unit)
            out.append(round_piece((v, True, v, True), outward) if as_float else (v, True, v, True))
    if hulled:
        warn(HullWarning, f'{name}: more than {ENUMERATION_CAP} values, or infinitely many, so their hull '
                          f'was returned')
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)


def _units(v, unit: Fraction):
    """v in grid units, exactly; -inf and inf for sign, whose rules only need the sign"""
    if is_infinite(v):
        return v
    return Fraction(v) / unit


def _value(n, unit: Fraction):
    """grid value n (an int, or ±inf) as a number; an integral Fraction becomes int in Cut"""
    if is_infinite(n):
        return n
    return n * unit


def floor(a: Cuts, outward: bool = False) -> Cuts:
    """
    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(floor(parse('{ (-1, 1/2] , (2, 3) }')))
    '{ [-1] , [0] , [2] }'
    """
    return step('floor', a, outward=outward)


def ceil(a: Cuts, outward: bool = False) -> Cuts:
    return step('ceil', a, outward=outward)


def trunc(a: Cuts, outward: bool = False) -> Cuts:
    """
    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(trunc(parse('(-2, 2)')))
    '{ [-1] , [0] , [1] }'
    """
    return step('trunc', a, outward=outward)


def round_(a: Cuts, ndigits: Optional[int] = None, outward: bool = False) -> Cuts:
    return step('round', a, ndigits, outward)


def round_ties_away(a: Cuts, ndigits: Optional[int] = None, outward: bool = False) -> Cuts:
    """
    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(round_ties_away(parse('[1/2, 5/2]')))
    '{ [1] , [2] , [3] }'
    """
    return step('round_ties_away', a, ndigits, outward)


def sign(a: Cuts) -> Cuts:
    return step('sign', a)
