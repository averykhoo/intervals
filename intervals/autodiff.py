"""
forward-mode automatic differentiation over sets (H3, the solver stack's first part): `Dual`, a value
and its derivative, each a set

`Dual.variable(x)` seeds the derivative with `[1]`; then every op on a `Dual` computes the op's set on
the value and the chain rule's set on the derivative, with the library's own arithmetic. so for `f`
written with `+ - * / **`, `abs` and the elementary functions, `f(Dual.variable(X)).derivative`
encloses `{f'(x) : x ∈ X, f differentiable at x}`: every piece of every sub-expression is evaluated
as a set, and the sets enclose, exactly for int and Fraction and outward in an `OutwardMultiInterval`
(to nearest in a `MultiInterval` with float ends, which is not an enclosure). it is the derivative's
*values*, not a proof that f is differentiable: `abs` at 0 gives `sign(0) = 0`, a value that is not a
derivative. the parts may be `DecoratedInterval`s, and then the two decorations are that proof: a
value and a derivative both decorated dac or better say that every op, and every op of the chain
rule's formula, was defined and continuous on the input, so `f` is C¹ there (`intervals.solver`
relies on it). a derivative formula is undefined exactly where its op is not differentiable (`sqrt`
at 0 divides by 0; `abs` at 0 gives `sign`, not continuous there), so there it is trv or def

a number or a bare interval in an op is a constant, with derivative `[0]`; a decorated `Dual` takes
numbers as constants (a `DecoratedInterval` refuses a bare `MultiInterval`, and so does this)

>>> from intervals import MultiInterval as M
>>> x = Dual.variable(M(1, 2))
>>> y = x ** 2 - 3 * x                 # each set evaluated once: the value is not tight
>>> print(y.value, y.derivative)
[-5, 1] [-1, 1]
>>> print(derivative(lambda t: 1 / t, M(-1, 1)))   # -1/t**2 over both sides of the pole
[-inf, -1]
"""
from numbers import Real

from intervals import numpy_compat
from intervals.decorated import DecoratedInterval
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import _is_integral
from intervals.multi_interval import _refuse_shift_count

_PART = (MultiInterval, DecoratedInterval)


def _constant(like, value):
    """`value` as a point of the same kind as `like`: its class, and decorated if it is"""
    if isinstance(like, DecoratedInterval):
        return DecoratedInterval(type(like.interval)(value))
    return type(like)(value)


class Dual:
    """
    a value and its derivative, each a `MultiInterval` (either class) or a `DecoratedInterval`, the
    two of one kind. immutable; the ops are the arithmetic dunders, `abs`, `**` and the elementary
    functions of `MultiInterval`, each with its chain rule (see the module docstring)
    """
    __slots__ = ('_value', '_derivative')
    __array_ufunc__ = numpy_compat.array_ufunc  # numpy interop, as for MultiInterval (M16d)

    def __init__(self, value, derivative):
        if not isinstance(value, _PART) or not isinstance(derivative, _PART):
            raise TypeError('a Dual is two MultiIntervals or two DecoratedIntervals')
        if isinstance(value, DecoratedInterval) != isinstance(derivative, DecoratedInterval):
            raise TypeError('a Dual is two MultiIntervals or two DecoratedIntervals, not one of each')
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, '_derivative', derivative)

    @classmethod
    def variable(cls, x) -> 'Dual':
        """the independent variable at `x` (a set, or a number as its point): derivative `[1]`"""
        if isinstance(x, Real) and not isinstance(x, bool):
            x = MultiInterval(x)
        if not isinstance(x, _PART):
            raise TypeError(f'expected a MultiInterval, a DecoratedInterval or a number, got {type(x).__name__}')
        return cls(x, _constant(x, 1))

    @classmethod
    def constant(cls, x) -> 'Dual':
        """a constant at `x` (a set): derivative `[0]`"""
        if not isinstance(x, _PART):
            raise TypeError(f'expected a MultiInterval or a DecoratedInterval, got {type(x).__name__}')
        return cls(x, _constant(x, 0))

    @property
    def value(self):
        return self._value

    @property
    def derivative(self):
        return self._derivative

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._value!r}, {self._derivative!r})'

    def __str__(self) -> str:
        return f'{self._value} d {self._derivative}'

    def _coerce(self, other):
        """a Dual as it is; a number or a bare set as a constant; else NotImplemented"""
        if isinstance(other, Dual):
            return other
        if isinstance(other, Real) and not isinstance(other, bool):
            return Dual(_constant(self._value, other), _constant(self._value, 0))
        if isinstance(other, _PART):
            return Dual.constant(other)
        return NotImplemented

    def _chain(self, value, factor) -> 'Dual':
        """a function of self: `value`, and the derivative `factor * self'`"""
        return Dual(value, factor * self._derivative)

    # ARITHMETIC

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Dual(self._value + other._value, self._derivative + other._derivative)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Dual(self._value - other._value, self._derivative - other._derivative)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other - self

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Dual(self._value * other._value,
                    self._derivative * other._value + self._value * other._derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """`(u / v)' = (u' v - u v') / v ** 2`, the square a pown, so `1 / x` gives `-1 / x ** 2`,
        never positive (the form `(u' - (u / v) v') / v` loses the sign across a pole)"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        u, v = self._value, other._value
        return Dual(u / v, (self._derivative * v - u * other._derivative) / v ** 2)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self

    def __lshift__(self, n):
        """`(u << n)' = u' << n`: `u * 2 ** n` for an int n of either sign (`MultiInterval.__lshift__`)"""
        value = self._value.__lshift__(n)
        return NotImplemented if value is NotImplemented else Dual(value, self._derivative << n)

    def __rshift__(self, n):
        """`(u >> n)' = u' >> n`: `u * 2 ** -n`"""
        value = self._value.__rshift__(n)
        return NotImplemented if value is NotImplemented else Dual(value, self._derivative >> n)

    def __rlshift__(self, other):
        return _refuse_shift_count(self, other, '<<')

    def __rrshift__(self, other):
        return _refuse_shift_count(self, other, '>>')

    def reciprocal(self) -> 'Dual':
        """`(1 / u)' = -u' / u ** 2`"""
        return self._chain(self._value.reciprocal(), -1 / self._value ** 2)

    def __neg__(self) -> 'Dual':
        return Dual(-self._value, -self._derivative)

    def __pos__(self) -> 'Dual':
        return self

    def __abs__(self) -> 'Dual':
        """`abs(u)' = sign(u) u'`: at `u = 0` the value 0, which is no derivative (see the module)"""
        return self._chain(abs(self._value), self._value.sign())

    def __pow__(self, exponent, modulo=None):
        """
        a number exponent `r`: `r u ** (r - 1) u'`, the power as `MultiInterval.__pow__` reads it
        (pown for an integral `r`, else pow, whose domain drops `u < 0`); `u ** 0` is the constant 1.
        `r - 1` is never computed in r's own arithmetic, which rounds to nearest (`2.0 ** 60 - 1` is
        `2.0 ** 60`, and a rounded `0.1 - 1` let an outward derivative miss its true value, M15): an
        integral `r` is the int `n`, so `n - 1` is exact, and any other `r` is a point set `e` of
        u's kind, so `e - 1` is the library's own subtraction, exact or outward (M16d, 2026-09-28).
        a `Dual` or set exponent `v`: `u ** v (v' log u + v u' / u)`, pow over `u > 0`
        """
        if modulo is not None or isinstance(exponent, bool):
            return NotImplemented
        if isinstance(exponent, Real):
            value = self._value ** exponent
            if _is_integral(exponent):
                n = int(exponent)
                if n == 0:
                    return Dual(value, _constant(self._value, 0))
                return self._chain(value, n * self._value ** (n - 1))
            e = _constant(self._value, exponent)
            return self._chain(value, e * self._value ** (e - 1))
        exponent = self._coerce(exponent)
        if exponent is NotImplemented:
            return NotImplemented
        return _pow(self, exponent)

    def __rpow__(self, base):
        base = self._coerce(base)
        if base is NotImplemented:
            return NotImplemented
        return _pow(base, self)

    # ELEMENTARY FUNCTIONS: each is `_chain(f(u), f'(u))`, f' written with f(u) where that is shorter

    def sqrt(self) -> 'Dual':
        r = self._value.sqrt()
        return self._chain(r, 1 / (2 * r))

    def cbrt(self) -> 'Dual':
        r = self._value.cbrt()
        return self._chain(r, 1 / (3 * r ** 2))

    def rootn(self, n: int) -> 'Dual':
        r = self._value.rootn(n)  # the core checks n: any Integral but bool
        n = int(n)  # so `n - 1` is exact, never an int64 that wraps (M16d)
        return self._chain(r, 1 / (n * r ** (n - 1)))

    def exp(self) -> 'Dual':
        r = self._value.exp()
        return self._chain(r, r)

    def expm1(self) -> 'Dual':
        return self._chain(self._value.expm1(), self._value.exp())

    def exp2(self) -> 'Dual':
        r = self._value.exp2()
        return self._chain(r, r * _constant(self._value, 2).log())

    def exp10(self) -> 'Dual':
        r = self._value.exp10()
        return self._chain(r, r * _constant(self._value, 10).log())

    def log(self, base=None) -> 'Dual':
        """the natural log, or to a number `base`: `u' / (u log base)`"""
        u = self._value
        if base is None:
            return self._chain(u.log(), 1 / u)
        return self._chain(u.log(base), 1 / (u * _constant(u, base).log()))

    def log2(self) -> 'Dual':
        return self._chain(self._value.log2(), 1 / (self._value * _constant(self._value, 2).log()))

    def log10(self) -> 'Dual':
        return self._chain(self._value.log10(), 1 / (self._value * _constant(self._value, 10).log()))

    def log1p(self) -> 'Dual':
        return self._chain(self._value.log1p(), 1 / (1 + self._value))

    def sin(self) -> 'Dual':
        return self._chain(self._value.sin(), self._value.cos())

    def cos(self) -> 'Dual':
        return self._chain(self._value.cos(), -self._value.sin())

    def tan(self) -> 'Dual':
        r = self._value.tan()
        return self._chain(r, 1 + r ** 2)

    def cot(self) -> 'Dual':
        r = self._value.cot()
        return self._chain(r, -(1 + r ** 2))

    def sec(self) -> 'Dual':
        r = self._value.sec()
        return self._chain(r, r * self._value.tan())

    def csc(self) -> 'Dual':
        r = self._value.csc()
        return self._chain(r, -(r * self._value.cot()))

    def asin(self) -> 'Dual':
        return self._chain(self._value.asin(), 1 / (1 - self._value ** 2).sqrt())

    def acos(self) -> 'Dual':
        return self._chain(self._value.acos(), -1 / (1 - self._value ** 2).sqrt())

    def atan(self) -> 'Dual':
        return self._chain(self._value.atan(), 1 / (1 + self._value ** 2))

    def acot(self) -> 'Dual':
        return self._chain(self._value.acot(), -1 / (1 + self._value ** 2))

    def sinh(self) -> 'Dual':
        return self._chain(self._value.sinh(), self._value.cosh())

    def cosh(self) -> 'Dual':
        return self._chain(self._value.cosh(), self._value.sinh())

    def tanh(self) -> 'Dual':
        r = self._value.tanh()
        return self._chain(r, 1 - r ** 2)

    def coth(self) -> 'Dual':
        r = self._value.coth()
        return self._chain(r, 1 - r ** 2)

    def sech(self) -> 'Dual':
        r = self._value.sech()
        return self._chain(r, -(r * self._value.tanh()))

    def csch(self) -> 'Dual':
        r = self._value.csch()
        return self._chain(r, -(r * self._value.coth()))

    def asinh(self) -> 'Dual':
        return self._chain(self._value.asinh(), 1 / (self._value ** 2 + 1).sqrt())

    def acosh(self) -> 'Dual':
        return self._chain(self._value.acosh(), 1 / (self._value ** 2 - 1).sqrt())

    def atanh(self) -> 'Dual':
        return self._chain(self._value.atanh(), 1 / (1 - self._value ** 2))

    def acoth(self) -> 'Dual':
        return self._chain(self._value.acoth(), 1 / (1 - self._value ** 2))


def _pow(base: Dual, exponent: Dual) -> Dual:
    """`u ** v`, pow: `(u ** v)' = u ** v (v' log u + v u' / u)`"""
    u, v = base.value, exponent.value
    value = u ** v
    return Dual(value, value * (exponent.derivative * u.log() + v * base.derivative / u))


def derivative(f, x):
    """
    `f'` over `x` (a set or a number): `f(Dual.variable(x)).derivative`, the set that encloses the
    derivative's values (see the module docstring); `[0]` if `f` returns a number
    """
    variable = Dual.variable(x)
    y = f(variable)
    if isinstance(y, Real) and not isinstance(y, bool):
        return _constant(variable.value, 0)
    if not isinstance(y, Dual):
        raise TypeError(f'f returned {type(y).__name__}, not a Dual or a number')
    return y.derivative


# SEVERAL VARIABLES (M16): a gradient or a jacobian is n passes, pass j with x_j the variable

def _box(xs) -> tuple:
    """xs as n sets of one kind: a number is a point of the first set's kind (a `MultiInterval` if
    every entry is a number); a bare set beside a decorated one is refused here, up front, since `f`
    need not combine the two (a `lambda x, y: x` would never meet the bare `y`)"""
    if not isinstance(xs, (list, tuple)):
        raise TypeError(f'xs is a list or a tuple of sets or numbers, got {type(xs).__name__}')
    if not xs:
        raise ValueError('xs needs at least one variable')
    for x in xs:
        if not isinstance(x, _PART) and not (isinstance(x, Real) and not isinstance(x, bool)):
            raise TypeError(f'expected a MultiInterval, a DecoratedInterval or a number, got {type(x).__name__}')
    sets = [x for x in xs if isinstance(x, _PART)]
    if len({isinstance(x, DecoratedInterval) for x in sets}) > 1:
        raise TypeError('xs is MultiIntervals or DecoratedIntervals, not one of each')
    like = sets[0] if sets else MultiInterval(0)
    return tuple(x if isinstance(x, _PART) else _constant(like, x) for x in xs)


def _entry(y, zero):
    """an output's derivative: a Dual's, or `zero` for a number (a constant)"""
    if isinstance(y, Real) and not isinstance(y, bool):
        return zero
    if not isinstance(y, Dual):
        raise TypeError(f'f returned {type(y).__name__}, not a Dual or a number')
    return y.derivative


def _passes(F, xs, row):
    """the n passes of F over the box xs, each an output's derivatives, as columns; `row(ys)` reads
    F's outputs as a sequence"""
    box = _box(xs)
    columns = []
    for j, x in enumerate(box):
        ys = row(F(*(Dual.variable(c) if k == j else Dual.constant(c) for k, c in enumerate(box))))
        columns.append(tuple(_entry(y, _constant(x, 0)) for y in ys))
    if len({len(column) for column in columns}) > 1:
        raise ValueError('F returned sequences of different lengths')
    return columns


def _sequence(ys):
    if not isinstance(ys, (list, tuple)):
        raise TypeError(f'F returned {type(ys).__name__}, not a list or a tuple of Duals or numbers')
    return ys


def gradient(f, xs) -> tuple:
    """
    `(∂f/∂x_1, ..., ∂f/∂x_n)` over the box `xs` (a list or a tuple of n sets or numbers, every set
    bare or every set decorated): n passes of `f`, pass j calling `f(c_1, ..., Dual.variable(x_j),
    ..., c_n)` with `c_k = Dual.constant(x_k)` and reading the derivative. `f` takes n positional
    arguments and returns a `Dual` or a number (a constant, whose partials are `[0]`). at n == 1 it
    is `(derivative(f, x),)`, the same pass

    decorated, the entries carry the C¹ proof of `derivative`, now in n variables and relative to
    the box: a value and every partial dac or better say every op, and every op of its chain rule's
    formula, was defined and continuous on the box, so `f` is C¹ on the box relative to the box (the
    multivariate chain rule is the same products). that is what the mean value theorem on the box
    needs, one-sided at its faces; it says nothing outside the box (`x ** 1.5` over `[0, 1]` is com,
    with no point below 0 in its domain, and `abs` over the point `[0]` has a dac derivative)

    >>> from intervals import MultiInterval as M
    >>> [str(d) for d in gradient(lambda x, y: x * y.sin(), [M(1, 2), 0])]
    ['[0]', '[1, 2]']
    """
    return tuple(column[0] for column in _passes(f, xs, lambda y: (y,)))


def jacobian(F, xs) -> tuple:
    """
    the jacobian of `F` over the box `xs`, as rows: `jacobian(F, xs)[i][j]` is `∂F_i/∂x_j` (see
    `gradient`; the same n passes, each giving one column of every row). `F` takes n positional
    arguments and returns a list or a tuple of m `Dual`s or numbers; the jacobian is m x n

    >>> from intervals import MultiInterval as M
    >>> for row in jacobian(lambda x, y: (x * y, x - y ** 2), [2, 3]):
    ...     print(*row)
    [3] [2]
    [1] [-6]
    """
    columns = _passes(F, xs, _sequence)
    return tuple(tuple(column[i] for column in columns) for i in range(len(columns[0])))
