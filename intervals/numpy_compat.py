"""
numpy interop (H3's second part, M16d): `array_ufunc`, the `__array_ufunc__` of `MultiInterval`,
`DecoratedInterval` and `Dual`, and `array`, the `__array__` of `MultiInterval`

numpy stays optional: this module imports nothing from numpy or from the package, and numpy is
imported inside the two hooks, which only numpy calls, so it is loaded already. "ours" is a type
whose `__array_ufunc__` is `array_ufunc`, so this module sits below the three classes.

**why the hook reproduces python's operators.** once a class has an `__array_ufunc__` that is not
None, a numpy scalar on the left of an operator (`np.float64(2) + A`) no longer returns
NotImplemented to python: numpy calls the ufunc (`np.add`), so this hook, and a hook that declined
would make it a TypeError where python ran `A.__radd__` before. so every operator ufunc runs
python's protocol **on our dunders only** (a numpy operand never re-enters numpy): `x.__op__(y)` if
x is ours, then `y.__rop__(x)` if y is ours, both ours python's own operator (subclass first, so
`np.add(M, O)` is `M + O`, an `OutwardMultiInterval`); `==` and `!=` fall back to identity, as
python's do, so `np.float64(2) == M(2)` stays False. numpy hands a scalar on the left of a
comparison over as a 0-d array, which is unwrapped first.

**the other ufuncs** are the method that computes the set image of the ufunc's pointwise function
(`np.sin(A)` is `A.sin()`, `np.arcsin(A)` is `A.asin()`, `np.rint(A)` is `A.round()`, ties to even
as numpy's rint, `np.square(A)` is `A ** 2`, `np.minimum(1, A)` is `A.minimum(1)`,
`np.arctan2(y, A)` is `y.atan2(A)`), so with both operands ours the methods' own rule, the
operators' (`multi_interval.py::_subclass_decides`): `np.hypot(M, O)` is `M.hypot(O)`, an
`OutwardMultiInterval` in either order, as `M + O` is; a class without the method (`Dual` has
no `floor`) is a TypeError. `fmin`/`fmax` are `minimum`/`maximum` (owner, 2026-10-03, Q15(f)): they
agree wherever there is no nan, and a nan operand is the library's ValueError, as for every op,
not numpy's skipping of it. not mapped, so a TypeError: `fmod` (C's truncated remainder, not `%`,
`fmod(-7, 2)` is -1), `float_power`, `isnan` and the other predicates of a point, `matmul`, every other ufunc, every method but `__call__`
(`reduce`, `outer`, ...) and every keyword (`out=`, `where=`, `dtype=`, ...). the table is keyed by
the ufunc *objects*, so a foreign ufunc sharing a name (scipy's) is not numpy's.

**an ndarray meeting one of ours** (`np.linspace(0, 1, 3) + A`) is elementwise into an object
array: each element (a python number, as `tolist()` gives it) meets the object as the scalar path
does, and an element with no answer raises. `==` and `!=` too, into a bool array, as numpy compares
an array with any element type (owner, 2026-10-03, Q15(c)): `np.array([A, B]) == A` is
`[True, False]`, so `A in arr` is True where arr holds A, and `f == A` for a float array is all
False; each element is still compared structurally. a list is not an array here
(`np.add([1, 2], A)` is a TypeError).

**`array`** makes `np.array(A)` a 0-d object array holding A, so numpy takes a `MultiInterval` for
one element and not for the sequence of its pieces (`len`, `iter`, `A[a:b]`): `np.array([A, B])`
has shape `(2,)`. a float dtype is numpy's cast, `float()` of each element, **to nearest in both
classes**: `np.asarray(O(Fraction(1, 3)), dtype=float)` is the double nearest 1/3, as
`float(O(Fraction(1, 3)))` is.
"""
import operator

# the binary operator ufuncs: numpy's name -> (dunder, reflected dunder, python's operator)
_OPERATORS = {
    'add': ('__add__', '__radd__', operator.add),
    'subtract': ('__sub__', '__rsub__', operator.sub),
    'multiply': ('__mul__', '__rmul__', operator.mul),
    'divide': ('__truediv__', '__rtruediv__', operator.truediv),
    'floor_divide': ('__floordiv__', '__rfloordiv__', operator.floordiv),
    'remainder': ('__mod__', '__rmod__', operator.mod),
    'divmod': ('__divmod__', '__rdivmod__', divmod),
    'power': ('__pow__', '__rpow__', operator.pow),
    'left_shift': ('__lshift__', '__rlshift__', operator.lshift),
    'right_shift': ('__rshift__', '__rrshift__', operator.rshift),
    'bitwise_and': ('__and__', '__rand__', operator.and_),
    'bitwise_or': ('__or__', '__ror__', operator.or_),
    'bitwise_xor': ('__xor__', '__rxor__', operator.xor),
    'less': ('__lt__', '__gt__', operator.lt),
    'less_equal': ('__le__', '__ge__', operator.le),
    'greater': ('__gt__', '__lt__', operator.gt),
    'greater_equal': ('__ge__', '__le__', operator.ge),
    'equal': ('__eq__', '__eq__', operator.eq),
    'not_equal': ('__ne__', '__ne__', operator.ne),
}
# the unary operator ufuncs: numpy's name -> dunder (`invert` is `~`, the complement)
_UNARY = {'negative': '__neg__', 'positive': '__pos__', 'absolute': '__abs__', 'fabs': '__abs__',
          'invert': '__invert__'}
# the ufuncs whose set image a method computes: numpy's name -> the method
_METHODS = {
    'sqrt': 'sqrt', 'cbrt': 'cbrt', 'exp': 'exp', 'exp2': 'exp2', 'expm1': 'expm1', 'log': 'log',
    'log2': 'log2', 'log10': 'log10', 'log1p': 'log1p', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan', 'sinh': 'sinh', 'cosh': 'cosh',
    'tanh': 'tanh', 'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh', 'floor': 'floor',
    'ceil': 'ceil', 'trunc': 'trunc', 'rint': 'round', 'sign': 'sign', 'reciprocal': 'reciprocal',
}
# binary and symmetric: the method of whichever operand is ours, the other its argument
_SYMMETRIC = {'minimum': 'minimum', 'maximum': 'maximum', 'hypot': 'hypot', 'fmin': 'minimum', 'fmax': 'maximum'}
_OTHERS = ('square', 'arctan2')

_table = None  # {ufunc object: numpy's name}, built on the first call (numpy is loaded by then)


def _ufunc_names() -> dict:
    global _table
    if _table is None:
        import numpy as np
        names = (*_OPERATORS, *_UNARY, *_METHODS, *_SYMMETRIC, *_OTHERS)
        _table = {getattr(np, name): name for name in names if hasattr(np, name)}
    return _table


def _special(cls, name: str):
    """a special method as python looks it up: in the class's MRO, never on its metaclass
    (`hasattr(Dual, '__or__')` is True through `type.__or__`); None where there is none"""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return None


def _ours(x) -> bool:
    return getattr(type(x), '__array_ufunc__', None) is array_ufunc


def _operator(name: str, x, y):
    """python's protocol for `x op y`, run on our dunders only"""
    forward, reflected, python = _OPERATORS[name]
    if _ours(x) and _ours(y):
        return python(x, y)
    result = NotImplemented
    if _ours(x) and _special(type(x), forward) is not None:
        result = _special(type(x), forward)(x, y)
    if result is NotImplemented and _ours(y) and _special(type(y), reflected) is not None:
        result = _special(type(y), reflected)(y, x)
    if result is NotImplemented and name == 'equal':
        return x is y  # python's fallback
    if result is NotImplemented and name == 'not_equal':
        return x is not y
    return result


def _method(x, name: str):
    return getattr(x, name, None) if _ours(x) else None


def _apply(name: str, *args):
    """one tuple of scalars: the answer, or NotImplemented where ours has none"""
    if name in _OPERATORS:
        return _operator(name, *args)
    if name == 'square':
        return _operator('power', args[0], 2)
    if name in _UNARY:
        (x,) = args
        dunder = _special(type(x), _UNARY[name]) if _ours(x) else None
        return NotImplemented if dunder is None else dunder(x)
    if name in _METHODS:
        method = _method(args[0], _METHODS[name])
        return NotImplemented if method is None else method()
    if name in _SYMMETRIC:
        x, y = args if _ours(args[0]) else args[::-1]
        method = _method(x, _SYMMETRIC[name])
        return NotImplemented if method is None else method(y)
    # arctan2(y, x): y.atan2(x), a number y a point of x's kind first
    y, x = args
    if not _ours(y):
        if _method(x, 'atan2') is None:
            return NotImplemented
        y = x._coerce(y)
        if y is NotImplemented:
            return NotImplemented
    method = _method(y, 'atan2')
    return NotImplemented if method is None else method(x)


def array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """`__array_ufunc__` of the three classes (see the module docstring)"""
    if method != '__call__' or kwargs:
        return NotImplemented
    name = _ufunc_names().get(ufunc)
    if name is None:
        return NotImplemented
    import numpy as np
    if not any(isinstance(x, np.ndarray) and x.ndim > 0 for x in inputs):
        # the scalar path: the dunder or the method python reaches, on the scalars numpy hands over
        return _apply(name, *(x[()] if isinstance(x, np.ndarray) else x for x in inputs))

    def element(*args):
        result = _apply(name, *args)
        if result is NotImplemented:
            raise TypeError(f'{name} is not defined for {", ".join(type(a).__name__ for a in args)}')
        return result
    # elementwise: ours as 0-d object arrays, so numpy does not call this hook again. numpy reads the
    # floating-point status flags after an object loop, and python's own float arithmetic inside
    # the library leaves them set (`1 / 2.2e-309` overflows, and the library knows it): errstate
    # keeps that from reaching the caller as a numpy RuntimeWarning the scalar path never gives
    with np.errstate(all='ignore'):
        out = np.frompyfunc(element, len(inputs), ufunc.nout)(*(_zero_d(x) if _ours(x) else x for x in inputs))
    # `==` and `!=` give numpy's bool array, as its object loop does for any other element type
    return out.astype(bool) if name in ('equal', 'not_equal') else out


def _zero_d(x):
    import numpy as np
    out = np.empty((), dtype=object)
    out[()] = x
    return out


def array(self, dtype=None, copy=None):
    """`MultiInterval.__array__`: a 0-d object array holding self (numpy casts it where a dtype is
    asked); `copy=False` is a ValueError, as numpy 2 asks, since there is no array to share"""
    if copy is False:
        raise ValueError(f'a {type(self).__name__} is not an array: a copy cannot be avoided')
    return _zero_d(self)
