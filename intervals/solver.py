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
import itertools
import math
import warnings
from fractions import Fraction
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
    piece (a half-bounded piece's midpoint is ±max float, or a piece with no float inside). an
    exact midpoint beyond the doubles (`[10 ** 400, 10 ** 401]`) is returned exact"""
    mid = piece.mid()
    if not isinstance(mid, float):
        try:
            rounded = float(mid)
        except OverflowError:
            rounded = math.inf
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


# SEVERAL VARIABLES (M16, H3's second part): a square system F(x) = 0 over a box

class RootBox(NamedTuple):
    """a box, one connected set per coordinate, holding every zero of `F` in it that `solve` did not
    put in another box; `unique`: it holds exactly one zero, proved (False says nothing)"""
    box: Tuple[MultiInterval, ...]
    unique: bool


def solve(F, xs, *, tol=1e-10, max_steps=10_000) -> Tuple[RootBox, ...]:
    """
    the zeros of `F` in the box `xs` (a list or a tuple of n sets or numbers, not decorated), as
    `RootBox`es, pairwise disjoint (in some coordinate), each inside `xs`, sorted by their
    components' `sort_key`s: every zero of `F` in `xs` is in one of them. `F` takes n positional
    arguments and returns a list or a tuple of n `Dual`s, sets or numbers, using the library's ops
    as `newton`'s `f` does; it is called with decorated `Dual`s, with boxes of
    `OutwardMultiInterval`s and with points. n == 1 is `newton`, exactly

    a branch and prune like `newton`'s, per box: **prune by range** (`F` on the box; a component of
    the value without 0 drops it); **a point** is unique iff `F` is exactly `[0]` there; **the step**,
    on a box with no wide component (unbounded, or spanning more than a factor of 16 in magnitude)
    where `F` is proved C¹ on the closed hull `H` by decorations (every value and partial of the n
    decorated passes dac or better): the jacobian `J` over `H`, a point `m` of the box, `Y` the
    float inverse of `mid J` (the identity where it has none), `Mx = Y J`, `b = Y F(m)`, then

    * **krawczyk's test**: `K = m - b + (I - Mx)(H - m)`; every `K[i]` non-empty and inside
      `H[i]`'s interior proves exactly one zero in `H`, hence in the box (krawczyk 1969; rump,
      acta numerica 2010). `g(x) = x - Y F(x)` maps `H` into `K` by the mean value theorem row by
      row, so into `H`: brouwer gives a fixed point, a zero once `Y` is regular. uniqueness: take
      `K*`, the same form with the closed range of the true partials over the compact `H` in place
      of `J`; `K* ⊆ K ⊆ int H` and `K*` compact make `K*` strictly narrower than `H` in every
      coordinate, so `ρ(|I - Y A|) < 1` for every real `A` of the mean value theorem: `Y` is
      regular and `g` contracts
    * **the narrowing**, preconditioned interval gauss-seidel with `mul_rev` (hansen and sengupta):
      row by row `Z[i] = Z[i] ∩ (m[i] + mul_rev(Mx[i][i], -b[i] - Σ_{j≠i} Mx[i][j] (Z[j] - m[j])))`,
      each row using the rows before it. valid since a zero `z` gives a real `A` with rows in `J(H)`
      and `F(m) + A (z - m) = 0`. where `0 ∈ Mx[i][i]` the row is two pieces: the box splits there

    then `newton`'s rules: a unique box is narrowed while the step narrows it; an unproved one goes
    on while the step halves it (and past `tol` for a few steps); else it is **bisected**, a wide
    component first (round robin), else the widest. before an unproved box is output, **its
    simplest point** (the simplest rational of each closed component, when inside it) is output
    alone as a unique zero if `F` is exactly `[0]` there, the rest of the box going back as up to
    2n boxes; else krawczyk's test runs once more on the box **inflated and clipped to its region**
    (the part of the input the box stands for, cut only where a box is split, on each side of the
    cut: every zero of the region is in the box): a zero proved in the inflated box is in the
    region, so in the box. one row that pins a coordinate (`y - 1/4`) makes
    the box degenerate there, where no `K` fits inside an interior; the inflation gives it one

    known limits: a zero on a split face that is not a simple rational in every coordinate, or a
    singular zero, is enclosed by unproved boxes of width `tol`, often with unproved slivers beside
    it (sound, not proved). a continuum of zeros is bisected to `tol` everywhere, each box of it
    giving its simplest point and up to 2n rest boxes. an exact end beyond the doubles makes the
    float preconditioner overflow `b`, so such a box is bisected, not stepped

    >>> from intervals import MultiInterval as M
    >>> for root in solve(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [M(-10, 10), M(-10, 10)]):
    ...     print(root.unique, *root.box)
    True (-0.7071067811865476, -0.7071067811865475) (-0.7071067811865476, -0.7071067811865475)
    True (0.7071067811865475, 0.7071067811865476) (0.7071067811865475, 0.7071067811865476)
    >>> from fractions import Fraction
    >>> for root in solve(lambda x, y: (x ** 2 - 2, y - Fraction(1, 4)), [M(-10, 10), M(-10, 10)]):
    ...     print(root.unique, *root.box)          # y pinned at once: proved on the inflated box
    True (-1.4142135623730951, -1.414213562373095) [0.25]
    True (1.414213562373095, 1.4142135623730951) [0.25]
    >>> for root in solve(lambda x, y: (abs(x) + x / 2 - y, y - Fraction(1, 4)), [M(-1, 3), M(-1, 1)]):
    ...     print(root.unique, *root.box)          # not C¹ at x = 0; exact points
    True [-0.5] [0.25]
    True [1/6] [1/4]
    """
    box = _input_box(xs)
    n = len(box)
    if n == 1:
        roots = newton(lambda t: _outputs(F(t), 1)[0], box[0], tol=tol, max_steps=max_steps)
        return tuple(RootBox((r.interval,), r.unique) for r in roots)
    # the stack: (box, unique, steps taken past tol, the coordinate to split next, the box's region)
    work = [(b, False, 0, 0, b) for b in reversed(list(itertools.product(*(c.pieces for c in box))))]
    roots = []
    steps = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        while work:
            box, unique, past, turn, region = work.pop()
            steps += 1
            if steps > max_steps:
                roots.append(RootBox(box, unique))
                roots.extend(RootBox(b, u) for b, u, *_ in work)
                break
            values = _values(F, box)
            if any(0 not in v for v in values):
                continue
            if all(c.is_degenerate for c in box):
                # a point: its zero is certain when every value is exactly [0]
                roots.append(RootBox(box, all(v == type(v)(0) for v in values)))
                continue
            if not any(_wide(c) for c in box):
                J, smooth = _jacobian(F, box)
                m = _points(box)
                if smooth and m is not None:
                    Mx, b = _precondition(J, _values(F, tuple(map(OutwardMultiInterval, m))))
                    unique = unique or _krawczyk(box, Mx, b, m)
                    narrowed = _gauss_seidel(box, Mx, b, m)
                    if not narrowed:
                        continue
                    if len(narrowed) > 1:
                        pieces = list(zip(narrowed, _regions(region, narrowed)))
                        work.extend((p, False, past, turn, r) for p, r in reversed(pieces))
                        continue
                    narrowed = narrowed[0]
                    if unique:
                        if narrowed != box:
                            work.append((narrowed, True, past, turn, region))
                        else:
                            roots.append(RootBox(narrowed, True))
                        continue
                    width = _width(narrowed)
                    if 2 * width <= _width(box) and (width > tol or past < _PAST_TOL):
                        work.append((narrowed, False, past + (width <= tol), turn, region))
                        continue
                    box = narrowed
            if _width(box) <= tol or (choice := _choose(box, turn)) is None:
                _finish_box(F, box, unique, region, roots, work, turn)
                continue
            k, halves = choice
            halves = [box[:k] + (half,) + box[k + 1:] for half in halves]
            work.extend((h, False, 0, k + 1, r) for h, r in reversed(list(zip(halves, _regions(region, halves)))))
    return tuple(sorted(roots, key=lambda root: tuple(c.sort_key for c in root.box)))


def _input_box(xs) -> tuple:
    """xs as a tuple of `OutwardMultiInterval`s, a number as its point; a decorated set is refused,
    as `newton` refuses it"""
    if not isinstance(xs, (list, tuple)):
        raise TypeError(f'xs is a list or a tuple of sets or numbers, got {type(xs).__name__}')
    if not xs:
        raise ValueError('solve needs at least one variable')
    box = []
    for x in xs:
        if isinstance(x, Real) and not isinstance(x, bool):
            x = MultiInterval(x)
        if not isinstance(x, MultiInterval):
            raise TypeError(f'expected a MultiInterval or a number, got {type(x).__name__}')
        box.append(OutwardMultiInterval.from_cuts(x.cuts))
    return tuple(box)


def _outputs(ys, n: int):
    """F's outputs, a list or a tuple of n"""
    if not isinstance(ys, (list, tuple)) or len(ys) != n:
        what = f'{len(ys)} values' if isinstance(ys, (list, tuple)) else type(ys).__name__
        raise TypeError(f'F returned {what}, not a list or a tuple of {n}')
    return ys


def _values(F, box) -> tuple:
    """F's sets over the box (or at a point), a number as its point in the outward class"""
    values = []
    for y in _outputs(F(*box), len(box)):
        if isinstance(y, Real) and not isinstance(y, bool):
            y = OutwardMultiInterval(y)
        if not isinstance(y, MultiInterval):
            raise TypeError(f'F returned {type(y).__name__} for a MultiInterval, not a MultiInterval or a number')
        values.append(y)
    return tuple(values)


def _jacobian(F, box):
    """the jacobian over the closed hull of the box, by n decorated passes, and whether F is C¹ there:
    every value and every partial dac or better. the closed hull, since krawczyk's theorem is for a
    compact box (a box open at a pole, `(0, 1]` for `1 / x`, is not stepped)"""
    hull = [DecoratedInterval(c.closed_hull) for c in box]
    n = len(box)
    columns = []
    smooth = True
    for j in range(n):
        seeds = [DecoratedInterval(type(c.interval)(int(k == j))) for k, c in enumerate(hull)]
        column = []
        for y in _outputs(F(*(Dual(c, s) for c, s in zip(hull, seeds))), n):
            if isinstance(y, Real) and not isinstance(y, bool):
                column.append(OutwardMultiInterval(0))
                continue
            if not isinstance(y, Dual):
                raise TypeError(f'F returned {type(y).__name__} for a Dual, not a Dual or a number')
            smooth = smooth and y.value.decoration >= Decoration.DAC and y.derivative.decoration >= Decoration.DAC
            column.append(y.derivative.interval)
        columns.append(column)
    return tuple(tuple(columns[j][i] for j in range(n)) for i in range(n)), smooth


def _points(box):
    """a point of each component: `_point_in`'s float midpoint, else, on a bounded component with no
    float inside, the exact midpoint (so a component narrowed to two adjacent doubles can still be
    stepped); None if a component has neither"""
    m = []
    for c in box:
        p = _point_in(c)
        if p is None and c.is_finite:
            p = (Fraction(c.inf) + Fraction(c.sup)) / 2
        if p is None or p not in c:
            return None
        m.append(p)
    return tuple(m)


def _mid(s) -> float:
    """s's midpoint as a float, nan where it has none (empty) or it is beyond the doubles"""
    try:
        return float(s.mid())
    except (OverflowError, ValueError):
        return math.nan


def _inverse(A):
    """the inverse of a float matrix, by gauss-jordan with partial pivoting; None if an entry or the
    result is not finite or a pivot is 0"""
    n = len(A)
    if not all(math.isfinite(a) for row in A for a in row):
        return None
    rows = [list(row) + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(rows[r][c]))
        if rows[p][c] == 0:
            return None
        rows[c], rows[p] = rows[p], rows[c]
        rows[c] = [v / rows[c][c] for v in rows[c]]
        for r in range(n):
            if r != c and rows[r][c] != 0:
                f = rows[r][c]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[c])]
    Y = [row[n:] for row in rows]
    return Y if all(math.isfinite(a) for row in Y for a in row) else None


def _combine(row, sets):
    """Σ_k row[k] * sets[k], a term with an exact 0 coefficient skipped: 0 times an entry `[±inf]`
    is empty (D2's corner), where the real product is 0 (`0 * (a, inf)` is already `[0]`)"""
    total = OutwardMultiInterval(0)
    for y, s in zip(row, sets):
        if y != 0:
            total = total + s * y
    return total


def _precondition(J, fm):
    """`Mx = Y J` and `b = Y F(m)`, `Y` the float inverse of `mid J`, or the identity where it has
    none (a midpoint not finite, beyond the doubles, or a singular matrix): any real `Y` keeps the
    step valid, so the fallback is only weaker"""
    n = len(J)
    Y = _inverse([[_mid(J[i][j]) for j in range(n)] for i in range(n)])
    if Y is None:
        Y = [[int(i == j) for j in range(n)] for i in range(n)]
    Mx = tuple(tuple(_combine(Y[i], [J[k][j] for k in range(n)]) for j in range(n)) for i in range(n))
    return Mx, tuple(_combine(Y[i], fm) for i in range(n))


def _krawczyk(box, Mx, b, m) -> bool:
    """krawczyk's test on the closed hull `H` of the box: every `K[i] = m[i] - b[i] + Σ_j (δ_ij -
    Mx[i][j]) (H[j] - m[j])` non-empty and inside `H[i]`'s interior (see `solve`). a degenerate
    component has an empty interior, so a box with one is never proved this way"""
    hull = [c.closed_hull for c in box]
    for i in range(len(box)):
        k = m[i] - b[i]
        for j, h in enumerate(hull):
            k = k + (int(i == j) - Mx[i][j]) * (h - m[j])
        if not k or not k.issubset(hull[i].interior):
            return False
    return True


def _gauss_seidel(box, Mx, b, m) -> tuple:
    """the preconditioned gauss-seidel step with `mul_rev` (see `solve`): the boxes it leaves, none
    if a row is empty, two if a row is two pieces (the sweep stops there, the other components as
    narrowed so far), else the narrowed box"""
    Z = list(box)
    for i in range(len(box)):
        c = -b[i]
        for j in range(len(box)):
            if j != i:
                c = c - Mx[i][j] * (Z[j] - m[j])
        z = Z[i] & (m[i] + mul_rev(Mx[i][i], c))
        if not z:
            return ()
        if len(z.pieces) > 1:
            return tuple(tuple(Z[:i]) + (p,) + tuple(Z[i + 1:]) for p in z.pieces)
        Z[i] = z
    return (tuple(Z),)


def _width(box):
    return max(c.wid() for c in box)


def _wide(c) -> bool:
    """a component the step does not run on: unbounded (`[-inf, inf]` included), or spanning more
    than a factor of 16 in magnitude"""
    return not c.is_finite or _magnitude_split(c) is not None


def _choose(box, turn):
    """the coordinate to bisect and its halves: a wide component first, round robin from `turn` (x
    on `[-inf, inf]²` split to its scale while y stays the whole line prunes nothing), then the
    other components, widest first (ties to the lower index); None if no component can be split"""
    n = len(box)
    wide = [k % n for k in range(turn, turn + n) if _wide(box[k % n])]
    rest = sorted((k for k in range(n) if not _wide(box[k])), key=lambda k: -box[k].wid())
    for k in wide + rest:
        halves = _bisect(box[k])
        if halves is not None:
            return k, halves
    return None


def _simplest_between(lo: Fraction, hi: Fraction) -> Fraction:
    """the rational with the smallest denominator in `[lo, hi]` (`lo <= hi`), by continued fractions"""
    if lo <= 0 <= hi:
        return Fraction(0)
    if hi < 0:
        return -_simplest_between(-hi, -lo)
    whole = math.floor(lo)
    if whole == lo:
        return Fraction(whole)
    if whole + 1 <= hi:
        return Fraction(whole + 1)
    # lo and hi share their integer part: the simplest of the reciprocals of the fractional parts
    return whole + 1 / _simplest_between(1 / (hi - whole), 1 / (lo - whole))


def _simplest_point(box):
    """the simplest rational of each component's closed hull, if every one is inside its component;
    None if one is not, or a component is unbounded"""
    point = []
    for c in box:
        if not c.is_finite:
            return None
        v = _simplest_between(Fraction(c.inf), Fraction(c.sup))
        v = int(v) if v.denominator == 1 else v
        if v not in c:
            return None
        point.append(OutwardMultiInterval(v))
    return tuple(point)


def _finish_box(F, box, unique, region, roots, work, turn):
    """output the box; but, if it is unproved, first its simplest point, alone, as a unique zero if F
    is exactly [0] there (a zero at a simple rational sits on a split face or a degenerate component,
    where no krawczyk set fits inside the interior), the rest of the box back on the stack as up to
    2n boxes; else krawczyk's test on the box inflated within its region"""
    if not unique:
        point = _simplest_point(box)
        if point is not None and all(v == type(v)(0) for v in _values(F, point)):
            roots.append(RootBox(point, True))
            for k in range(len(box)):
                p = point[k].inf
                for piece in box[k].difference(point[k]).pieces:
                    side = (OutwardMultiInterval(-math.inf, p, end_closed=False) if piece.sup <= p
                            else OutwardMultiInterval(p, math.inf, start_closed=False))
                    rest = point[:k] + (piece,) + box[k + 1:]
                    work.append((rest, False, _PAST_TOL, turn, point[:k] + (region[k] & side,) + region[k + 1:]))
            return
        unique = _inflated_unique(F, box, region)
    roots.append(RootBox(box, unique))


def _regions(region, boxes) -> list:
    """the regions of boxes split from one box in one coordinate i, in increasing order there: the
    old region cut between the neighbouring boxes in coordinate i. a zero of a new region is a zero
    of the old one, so in one of the boxes, and on its own box's side of the cuts: in its own box.
    the other coordinates keep the old region's extent, which the inflation needs (a coordinate
    already pinned by a row stays wide in the region)"""
    if len(boxes) < 2:
        return [region] * len(boxes)
    i = next(k for k in range(len(region)) if boxes[0][k] != boxes[1][k])
    regions = []
    for n, box in enumerate(boxes):
        r = region[i]
        if n > 0:
            below = boxes[n - 1][i]
            r = r & OutwardMultiInterval(below.sup, math.inf, start_closed=not below.sup_closed)
        if n + 1 < len(boxes):
            above = boxes[n + 1][i]
            r = r & OutwardMultiInterval(-math.inf, above.inf, end_closed=not above.inf_closed)
        regions.append(region[:i] + (r,) + region[i + 1:])
    return regions


def _inflate(box, region):
    """the closed hull of (the box inflated) ∩ the region: each component widened on both sides by
    twice its width (at least 1e-12 relative, 1e-30 absolute), each end then clipped to the region's
    closed hull, an inflated end rounded toward the box. None if the box is unbounded or an end
    overflows"""
    hull = []
    for c, r in zip(box, region):
        if not c.is_finite:
            return None
        lo, hi = Fraction(c.inf), Fraction(c.sup)
        d = 2 * max(hi - lo, max(abs(lo), abs(hi)) / 10 ** 12, Fraction(1, 10 ** 30))
        try:
            lo, hi = _rounded(lo - d, up=True), _rounded(hi + d, up=False)
        except OverflowError:
            return None
        hull.append(OutwardMultiInterval(max(lo, r.inf), min(hi, r.sup)))
    return tuple(hull)


def _rounded(q: Fraction, up: bool) -> float:
    """q as a float, rounded up or down"""
    v = float(q)
    if up and v < q:
        return math.nextafter(v, math.inf)
    if not up and v > q:
        return math.nextafter(v, -math.inf)
    return v


def _inflated_unique(F, box, region) -> bool:
    """krawczyk's test on `H = _inflate(box, region)`, with its own jacobian and C¹ gate over `H`.
    every zero of the region is in the box, and the box is inside `H`: exactly one zero in `H`, in
    `K ⊆ int H ⊆ region`, is in the box, and is the box's only one (critique B1)"""
    H = _inflate(box, region)
    if H is None:
        return False
    J, smooth = _jacobian(F, H)
    m = _points(H)
    if not smooth or m is None:
        return False
    Mx, b = _precondition(J, _values(F, tuple(map(OutwardMultiInterval, m))))
    return _krawczyk(H, Mx, b, m)
