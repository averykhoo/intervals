"""
the generic arithmetic applicator: shape first, then attainment

an op is an `OpDescriptor`; `apply_unary` and `apply_binary` take and return cut tuples:

1. split every operand piece at the descriptor's split points (zero for mul, div, reciprocal, abs).
   a piece with 0 strictly inside becomes `[lo, 0]` and `[0, hi]`, both holding the 0; a piece
   that only touches 0, and a degenerate `[0]`, stay whole
2. for each box (one piece per operand), locate the result with every endpoint treated closed: the
   min and max of the op over the box's corners. a corner where the op has no value (`0 * inf`,
   `inf - inf`, `inf / inf`, `0 / 0`) contributes the limit along each non-degenerate edge leaving
   it, and nothing if the box *is* that point (then the call warns once)
3. close each result endpoint iff the box attains it: at a corner inside the box, or along a flat
   edge (`inf + y`, `0 * y`, `x / inf`, `x / 0`) whose fixed coordinate is inside the box. for a
   finite endpoint of an injective op only the corner can do it, which is the corner-flag rule;
   infinite endpoints and flat spots need the edges
4. union the boxes and normalize

attainment is decided per box, exactly, which gives the same union as deciding it against the full
operands. the face rule assumes what holds for `+ - * /` once split at zero: along an edge the op is
either constant or injective, every point without a value is a corner of the box, and an edge
leaving such a corner is constant. an op for which that fails supplies its own `attained`.

infinities are exact points: descriptors evaluate them symbolically, they never go through the
rounding hook, and a pole's sign comes from the side of zero its piece lies on (`dirs` below),
never from a sign bit.
"""
import math
import os
import sys
import warnings
from itertools import product
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from multiinterval import fmt
from multiinterval import kernel
from multiinterval.cuts import Value
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import IndeterminateResultWarning
from multiinterval.kernel import Cuts

Piece = Tuple[Value, bool, Value, bool]
Box = Tuple[Piece, ...]


class OpDescriptor(NamedTuple):
    """
    how the applicator evaluates one op

    * `fn(*args)`: the op at one point of the extended reals; `None` where it has no value
    * `monotone`: a direction per argument (+1 / -1) if the op is monotone on every split box; the
      result is then located from two corners instead of all of them
    * `split_points`: values every operand piece is split at
    * `attained(v, box) -> bool`: replaces the face rule (see the module docstring) if given
    * `rounded`: `(fn_down, fn_up)`, used for a corner whose operands are all finite and at least one
      is a float, and for a corner whose `fn` value is an `Unbuilt` (pown's power of exact operands
      too long to build); `None` is the identity. other exact corners and infinite operands never go
      through it
    * `pole(args, dirs)`: the value where `fn` has none but a limit from the piece's side does
      (`x / 0`); `dirs[i]` is +1 / -1 if piece i extends above / below `args[i]`, 0 if degenerate
    """
    name: str
    fn: Callable[..., Optional[Value]]
    monotone: Optional[Tuple[int, ...]] = None
    split_points: Tuple[Value, ...] = ()
    attained: Optional[Callable[[Value, Box], bool]] = None
    rounded: Optional[Tuple[Callable[..., Value], Callable[..., Value]]] = None
    pole: Optional[Callable[[tuple, tuple], Optional[Value]]] = None


class Unbuilt:
    """
    a corner value too long to build exactly (`ops._NotADouble` is the one kind): it stands for a
    finite value that equals no double and no other corner, so its corner always goes through the
    descriptor's rounding hooks, exact operands included, and attains nothing
    """
    __slots__ = ()


# SCALAR HELPERS (shared with ops.py)

def is_infinite(x) -> bool:
    return x == math.inf or x == -math.inf


def sign(x) -> int:
    return (x > 0) - (x < 0)


def signed_inf(s: int) -> float:
    return math.inf if s > 0 else -math.inf


# ENTRY POINTS

def apply_unary(desc: OpDescriptor, a: Cuts) -> Cuts:
    """
    >>> from multiinterval.ops import RECIPROCAL
    >>> fmt.format_cuts(apply_unary(RECIPROCAL, fmt.parse('[-1, 1]')))
    '{ [-inf, -1] , [1, inf] }'
    """
    return _apply(desc, (a,))


def apply_binary(desc: OpDescriptor, a: Cuts, b: Cuts) -> Cuts:
    """
    >>> from multiinterval.ops import MUL
    >>> fmt.format_cuts(apply_binary(MUL, fmt.parse('[0, 1]'), fmt.parse('(2, 3)')))
    '[0, 3)'
    """
    return _apply(desc, (a, b))


def _apply(desc: OpDescriptor, operands: Tuple[Cuts, ...]) -> Cuts:
    if not all(operands):
        warn(EmptySetPropagationWarning, f'{desc.name}: an operand is empty, so the result is empty')
        return kernel.EMPTY
    split = [split_pieces(kernel.pieces(cuts), desc.split_points) for cuts in operands]
    out = []
    empty_boxes = []
    for box in product(*split):
        result = evaluate_box(desc, box)
        if result is None:
            empty_boxes.append(box)
        else:
            lo, lo_closed, hi, hi_closed = result
            out.append(kernel.piece(lo, hi, lo_closed, hi_closed))
    if empty_boxes:
        shown = ', '.join(fmt.format_piece(*kernel.piece(lo, hi, lo_c, hi_c)) for lo, lo_c, hi, hi_c in empty_boxes[0])
        warn(IndeterminateResultWarning,
             f'{desc.name}({shown}) has no value at any point (an indeterminate form, or a pole with '
             f'no side), so that part contributes nothing')
    return kernel.normalize(out)


# SPLITTING

def split_pieces(pieces, points) -> List[Piece]:
    """
    split each piece at every point strictly inside it; the point goes into both halves

    >>> split_pieces([(-1, False, 2, True), (0, True, 0, True)], (0,))
    [(-1, False, 0, True), (0, True, 2, True), (0, True, 0, True)]
    """
    out = []
    for p in pieces:
        parts = [p]
        for s in points:
            parts = [half for lo, lo_c, hi, hi_c in parts
                     for half in (((lo, lo_c, s, True), (s, True, hi, hi_c)) if lo < s < hi
                                  else ((lo, lo_c, hi, hi_c),))]
        out.extend(parts)
    return out


# ONE BOX

def evaluate_box(desc: OpDescriptor, box: Box) -> Optional[Piece]:
    """
    the image of one box as `(lo, lo_closed, hi, hi_closed)`, or None if no point of it has a value

    >>> from multiinterval.ops import ADD
    >>> evaluate_box(ADD, ((math.inf, True, math.inf, True), (1, False, 2, False)))
    (inf, True, inf, True)
    """
    ends = [_ends(p) for p in box]
    if desc.monotone is None:
        corners = product(*ends)
    else:
        corners = (tuple(e[0] if d > 0 else e[-1] for e, d in zip(ends, desc.monotone)),
                   tuple(e[-1] if d > 0 else e[0] for e, d in zip(ends, desc.monotone)))
    lows, highs = [], []
    for corner in corners:
        args = tuple(value for value, _, _ in corner)
        dirs = tuple(d for _, _, d in corner)
        value = desc.fn(*args)
        if value is not None:
            if desc.rounded is not None and (_rounds(args) or isinstance(value, Unbuilt)):
                lows.append(desc.rounded[0](*args))
                highs.append(desc.rounded[1](*args))
            else:
                lows.append(value)
                highs.append(value)
            continue
        # a pole is an exact infinity: it never goes through the rounding hook
        value = _value(desc, args, dirs)
        if value is not None:
            lows.append(value)
            highs.append(value)
            continue
        # no value here: the limit along each non-degenerate edge leaving the corner
        for i, d in enumerate(dirs):
            if d:
                limit = _value(desc, _put(args, i, _stand_ins(box[i])[0]), _put(dirs, i, 0))
                if limit is not None:
                    lows.append(limit)
                    highs.append(limit)
    if not lows:
        return None
    lo = min(lows, key=lambda v: (v, _is_float(v)))
    hi = max(highs, key=lambda v: (v, not _is_float(v)))
    lo_closed, hi_closed = _attained(desc, box, ends, lo), _attained(desc, box, ends, hi)
    if lo == hi and not (lo_closed and hi_closed):
        # a box with a value is never empty. exactly, a single-valued box attains that value; this
        # is a float piece that rounding (underflow, overflow, absorption) squeezed to one point,
        # and a rounded flag is conservative, so the piece keeps that point
        lo_closed = hi_closed = True
    return lo, lo_closed, hi, hi_closed


def _attained(desc: OpDescriptor, box: Box, ends, v) -> bool:
    if desc.attained is not None:
        return desc.attained(v, box)
    for corner in product(*ends):
        if all(closed for _, closed, _ in corner):
            if _value(desc, tuple(c[0] for c in corner), tuple(c[2] for c in corner)) == v:
                return True
    # an edge: one coordinate free over its open piece, the others fixed at a closed end. the op is
    # constant along it or injective, so equal values at two stand-ins mean it is flat
    for i, free in enumerate(box):
        if free[0] == free[2]:
            continue
        for fixed in product(*(e for j, e in enumerate(ends) if j != i)):
            if not fixed or not all(closed for _, closed, _ in fixed):
                continue
            args = tuple(c[0] for c in fixed)
            dirs = tuple(c[2] for c in fixed)
            if all(_value(desc, args[:i] + (t,) + args[i:], dirs[:i] + (0,) + dirs[i:]) == v
                   for t in _stand_ins(free)):
                return True
    return False


def _ends(p: Piece):
    """the distinct ends of a piece as `(value, closed, direction into the piece)`"""
    lo, lo_closed, hi, hi_closed = p
    if lo == hi:
        return (lo, True, 0),
    return (lo, lo_closed, 1), (hi, hi_closed, -1)


def _stand_ins(p: Piece):
    """
    two finite points with the sign of the open piece (it lies on one side of zero once split,
    or crosses it for add/sub). along an edge from a special value only that sign matters
    """
    lo, _, hi, _ = p
    return (1, 2) if lo >= 0 else (-1, -2) if hi <= 0 else (-1, 1)


def _value(desc: OpDescriptor, args, dirs):
    value = desc.fn(*args)
    if value is None and desc.pole is not None:
        value = desc.pole(args, dirs)
    return value


def _put(t: tuple, i: int, x) -> tuple:
    return t[:i] + (x,) + t[i + 1:]


def _is_float(v) -> bool:
    return isinstance(v, float) and not is_infinite(v)


def _rounds(args) -> bool:
    return all(not is_infinite(x) for x in args) and any(isinstance(x, float) for x in args)


# WARNINGS

_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__))) + os.sep


def warn(category, message: str) -> None:
    """warnings.warn, attributed to the first caller outside this package"""
    frame, level = sys._getframe(0), 1
    while frame is not None and os.path.normcase(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
