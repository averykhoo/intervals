"""
interval newton over multi-intervals (H3, the solver stack's first part): `newton(f, x)` encloses
every zero of `f` in `x`, and proves which enclosures hold exactly one

a branch and prune over the pieces of `x`, each a connected set, kept on a stack:

* **prune by range**: `f` is evaluated on the piece as a set; if 0 is not in it, the piece has no zero
* **newton step**, where `f` is C¹ on the piece (below) and the piece is bounded: for a zero `z` and a
  point `m` of the piece, the mean value theorem gives `f(m) + f'(ξ) (z - m) = 0` for a `ξ` between
  them, so in the piece. so `z - m` is in `mul_rev(F', -f(m))`, the `t` with `t * y = -f(m)` for a
  `y` in `F'`, the derivative's set over the piece (`intervals.autodiff`). the piece becomes
  `piece ∩ (m + mul_rev(F', -f(m)))`. where `0 ∈ F'` that set has two pieces and so has the result:
  the step splits the piece at the gap in one go, which is what a multi-interval is for (1788's
  `mulRevToPair` gives the same two intervals as a pair; a connected interval type needs both, or
  their hull, the entire line). and `0 / 0` is not empty here, as the division's `[0] / [0]` is
  (D7): `mul_rev([0], [0])` is every `t`, since `t * 0 = 0`, which the theorem needs
* **uniqueness**: if `0 ∉ F'` and the newton set `m + mul_rev(F', -f(m))` is non-empty and inside the
  piece's interior, the piece holds exactly one zero (the interval newton theorem; with `0 ∉ F'` no
  two zeros fit, by rolle). it is kept by every later step, since a step keeps every zero
* **bisection** at the midpoint, where the step removed less than half the piece or could not run;
  a piece spanning more than a factor of 16 in magnitude is split by its exponents first
  (`_magnitude_split`), with no newton step

a piece is output as a `Root` once its width is at most `tol` (newton may go on past it for a few
steps while it halves the piece, to prove a zero unique), or its zero is proved unique and a step no
longer narrows it, or it cannot be split (no float strictly inside), or after `max_steps` pieces
(then every piece left is output as it is). before a piece is output unproved, a closed end where
`f` is exactly 0 is output alone, as a unique zero: a zero on a split point is a closed end of its
piece, and no newton set fits inside the interior there.

**C¹, proved by decorations.** the mean value theorem needs `f` differentiable, which an enclosure of
`f'` does not say (`abs` at 0 has one). so `f` is evaluated on a `Dual` of two `DecoratedInterval`s,
and the step runs only where both the value and the derivative are decorated dac or com: every op,
and every op of the chain rule, defined and continuous on the piece (see `intervals.autodiff`). where
`f` is not C¹, as `sqrt` at 0, or has a pole or a jump, the pieces are only pruned and bisected

**enclosures.** the pieces are `OutwardMultiInterval`s, so every value encloses; int and Fraction
stay exact. `f` takes one argument and uses the library's ops on it (`+ - * / **`, `abs`, the
elementary functions as methods), with numbers as its constants: it is called with a decorated
`Dual`, and with a single point as an `OutwardMultiInterval`. the library's warnings inside `f` are
not the caller's (a piece the solver made up can be an indeterminate point), so they are silenced

>>> from intervals import MultiInterval as M
>>> for root in newton(lambda x: x ** 2 - 2, M(-10, 10)):
...     print(root.unique, root.interval)
True (-1.4142135623730951, -1.414213562373095)
True (1.414213562373095, 1.4142135623730951)
>>> for root in newton(lambda x: x ** 2 - 2, M(-10, 10), max_steps=1):   # the first step alone
...     print(root.unique, root.interval)
False [-10, -0.09999999999999999)
False (0.09999999999999999, 10]
"""
import math
import warnings
from numbers import Real
from typing import NamedTuple
from typing import Tuple

from intervals.autodiff import Dual
from intervals.decorated import DecoratedInterval
from intervals.decorated import Decoration
from intervals.errors import IntervalWarning
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import OutwardMultiInterval
from intervals.reverse import mul_rev


_MAX = 1.7976931348623157e308  # the largest float
_SPAN = 16  # a piece wider than this factor in magnitude is split in the middle of its exponents
_PAST_TOL = 8  # newton steps on a piece already narrower than tol, to prove its zero unique


class Root(NamedTuple):
    """a connected piece of `x` holding every zero of `f` in it that `newton` did not put in another
    piece; `unique`: it holds exactly one zero, proved (False says nothing: none, one or several)"""
    interval: MultiInterval
    unique: bool


def newton(f, x, *, tol=1e-10, max_steps=10_000) -> Tuple[Root, ...]:
    """
    the zeros of `f` in `x` (a set or a number), as disjoint `Root`s in increasing order: every zero
    of `f` in `x` is in one of them (see the module docstring for the method and what `f` may use)

    >>> from intervals import MultiInterval as M
    >>> [str(r.interval) for r in newton(lambda x: x.sin(), M(-4, 4))]  # -pi, 0, pi
    ['(-3.1415926535897936, -3.141592653589793)', '[0.0]', '(3.141592653589793, 3.1415926535897936)']
    >>> newton(lambda x: x ** 2 + 1, M(-10, 10))   # no zero: pruned by range and by newton
    ()
    >>> newton(lambda x: abs(x), M(-1, 1))   # not C¹ at 0: bisected, onto 0, where f is exactly 0
    (Root(interval=OutwardMultiInterval.parse('[0.0]'), unique=True),)
    """
    if isinstance(x, Real) and not isinstance(x, bool):
        x = MultiInterval(x)
    if not isinstance(x, MultiInterval):
        raise TypeError(f'expected a MultiInterval or a number, got {type(x).__name__}')
    x = OutwardMultiInterval.from_cuts(x.cuts)
    # the stack: (piece, unique, newton steps taken on it past tol)
    work = [(piece, False, 0) for piece in reversed(x.pieces)]
    roots = []
    steps = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        while work:
            piece, unique, past = work.pop()
            steps += 1
            if steps > max_steps:
                roots.append(Root(piece, unique))
                roots.extend(Root(p, u) for p, u, _ in work)
                break
            value, slope, smooth = _evaluate(f, piece)
            if 0 not in value:
                continue
            if piece.is_degenerate:
                # a point: its zero is certain when the value is exactly [0]
                roots.append(Root(piece, value == type(value)(0)))
                continue
            if smooth and piece.is_finite and _magnitude_split(piece) is None:
                point = _point_in(piece)
                if point is not None:
                    narrowed, proved = _newton_step(piece, slope, point, _value_at(f, point))
                    unique = unique or proved
                    if not narrowed:
                        continue
                    if len(narrowed) > 1:
                        work.extend((p, False, past) for p in reversed(narrowed.pieces))
                        continue
                    if unique:
                        # a proved zero is narrowed while newton narrows it (quadratically), past tol
                        if narrowed != piece:
                            work.append((narrowed, True, past))
                        else:
                            roots.append(Root(narrowed, True))
                        continue
                    # newton goes on while it halves the piece, and past tol for _PAST_TOL steps: at
                    # a simple zero it converges quadratically, and a step or two more proves the
                    # zero unique; at a multiple zero only linearly (x ** 2 at 0 by 3/8 a step)
                    width = narrowed.wid()
                    if width <= piece.wid() / 2 and (width > tol or past < _PAST_TOL):
                        work.append((narrowed, False, past + (width <= tol)))
                        continue
                    piece = narrowed
            if piece.wid() <= tol or (halves := _bisect(piece)) is None:
                _finish(f, piece, unique, roots, work)
                continue
            work.extend((half, False, 0) for half in reversed(halves))
    return tuple(sorted(roots, key=lambda root: root.interval.sort_key))


def _newton_step(piece: MultiInterval, slope: MultiInterval, point, value: MultiInterval):
    """
    the piece narrowed by one newton step from `point` (f's `value` there, `slope` f''s set over the
    piece), and whether the step proves the piece holds exactly one zero: the newton set is non-empty
    and inside the interior, and `0 ∉ slope`. the last is not implied by the others: a `slope` with
    0 as an isolated point (`{0} ∪ [1, 2]`) gives a bounded newton set, yet rolle's argument needs no
    0 at all
    """
    image = point + mul_rev(slope, -value)
    return piece & image, bool(image) and 0 not in slope and image.issubset(piece.interior)


def _finish(f, piece: MultiInterval, unique: bool, roots: list, work: list):
    """output the piece, but a closed end where f is exactly 0 first, as a unique zero of its own: a
    zero on a split point is a closed end of its piece, where no newton set fits inside the interior.
    the rest of the piece goes back on the stack, to be pruned or output"""
    if not unique:
        cls = type(piece)
        ends = {end for end, closed in ((piece.inf, piece.inf_closed), (piece.sup, piece.sup_closed))
                if closed and _value_at(f, end) == cls(0)}
        if ends:
            roots.extend(Root(cls(end), True) for end in ends)
            work.extend((p, False, _PAST_TOL) for p in piece.difference(*map(cls, ends)).pieces)
            return
    roots.append(Root(piece, unique))


def _evaluate(f, piece: MultiInterval):
    """f's set and f''s set over the piece, and whether f is C¹ there (both decorated dac or com)"""
    y = f(Dual(DecoratedInterval(piece), DecoratedInterval(type(piece)(1))))
    if isinstance(y, Real) and not isinstance(y, bool):
        return type(piece)(y), type(piece)(0), True
    if not isinstance(y, Dual):
        raise TypeError(f'f returned {type(y).__name__} for a Dual, not a Dual or a number')
    value, slope = y.value, y.derivative
    smooth = value.decoration >= Decoration.DAC and slope.decoration >= Decoration.DAC
    return value.interval, slope.interval, smooth


def _value_at(f, point) -> MultiInterval:
    """f's set at a single point, in the outward class"""
    y = f(OutwardMultiInterval(point))
    if isinstance(y, Real) and not isinstance(y, bool):
        return OutwardMultiInterval(y)
    if not isinstance(y, MultiInterval):
        raise TypeError(f'f returned {type(y).__name__} for a MultiInterval, not a MultiInterval or a number')
    return y


def _point_in(piece: MultiInterval):
    """the piece's midpoint (of its hull, the piece being connected), as a float if one is in the
    piece, since exact midpoints make the fractions grow at every step; None if it is not in the
    piece (a half-bounded piece's midpoint is ±max float, or a piece with no float inside)"""
    mid = piece.mid()
    if not isinstance(mid, float):
        rounded = float(mid)
        if math.isfinite(rounded) and rounded in piece:
            return rounded
    return mid if mid in piece else None


def _magnitude_split(piece: MultiInterval):
    """
    on a piece spanning more than a factor of 16 in magnitude, a point in the middle of its
    exponents: 0 if the piece holds it, else ±1 if an end is 0, else ±2 ** the mean binary exponent.
    None on a narrower piece. newton from far off a zero gains about a constant factor a step, so
    such a piece is bisected here first: `[0, 1e300]` or `[-inf, inf]` reaches the scale of its zeros
    in about a dozen splits, not hundreds of steps
    """
    lo, hi = max(piece.inf, -_MAX), min(piece.sup, _MAX)
    if lo < 0 < hi:
        return 0 if max(-lo, hi) > _SPAN * min(-lo, hi) else None
    sign, lo, hi = (1, lo, hi) if lo >= 0 else (-1, -hi, -lo)
    if hi <= 1 or (lo > 0 and hi <= _SPAN * lo):
        return None
    if lo == 0:
        return sign
    return sign * 2.0 ** ((math.frexp(lo)[1] + math.frexp(hi)[1]) // 2)


def _bisect(piece: MultiInterval):
    """the piece split at `_magnitude_split`, else at `_point_in`, the point going left; None if one
    side would be empty"""
    point = _magnitude_split(piece)
    if point is None or point not in piece:
        point = _point_in(piece)
    if point is None:
        return None
    cls = type(piece)
    left = piece & cls(-math.inf, point)
    right = piece & cls(point, math.inf, start_closed=False)
    if not left or not right:
        return None
    return left, right
