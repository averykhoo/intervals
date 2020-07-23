import math
from numbers import Real
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class MultiInterval:
    endpoints: List[Tuple[Real, int]]

    def __init__(self):
        self.endpoints = []

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return len(self.endpoints) == 0

    @property
    def is_degenerate(self) -> bool:
        return len(self.endpoints) == 2 and self.inf == self.sup

    @property
    def inf(self) -> Optional[Real]:
        if len(self.endpoints) > 0:
            return self.endpoints[0][0]

    @property
    def sup(self) -> Optional[Real]:
        if len(self.endpoints) > 0:
            return self.endpoints[-1][0]

    @property
    def cardinality(self) -> Tuple[int, float, int]:
        """
        intuitively this is proportional to the overall length,
        but because of infinities and degeneracy it's a 3-tuple of:
            (1) number of open half-rays from 0 (infinite length, uncountable points,   0 <= n <= 2)
            (2) remaining length of open sets   (finite length,   uncountable points, -inf < n < inf)
            (3) remaining endpoints             (zero length,     countable points,   -inf < n < inf)

        e.g.: (1, inf)
            = (0, inf) - (0, 1]
            = (0, inf) - (0, 1) - [1]
            cardinality = (1, -1, -1)
        """
        # todo: count stuff
        half_rays = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        length = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        n_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)
        return half_rays, length, n_endpoints

    # UTILITY

    def __contains__(self, other: Union[Real, 'MultiInterval']) -> bool:
        raise NotImplementedError

    def expand(self, distance: Real) -> 'MultiInterval':
        raise NotImplementedError

    def closed_hull(self):
        raise NotImplementedError

    def overlaps(self, other: Union[Real, 'MultiInterval'], or_adjacent=False) -> bool:
        raise NotImplementedError

    # MISC

    def _consistency_check(self):
        # length must be even
        assert len(self.endpoints) % 2 == 0, len(self.endpoints)

        def _check_endpoint_tuple(endpoint):
            assert isinstance(endpoint, tuple), endpoint
            assert len(endpoint) == 2, endpoint
            assert isinstance(endpoint[0], Real), endpoint
            assert isinstance(endpoint[1], int), endpoint
            assert endpoint[1] in {-1, 0, 1}, endpoint

        # must be sorted
        if len(self.endpoints) > 0:
            prev = self.endpoints[0]
            _check_endpoint_tuple(prev)

            for elem in self.endpoints[1:]:
                _check_endpoint_tuple(elem)
                assert prev <= elem, (prev, elem)
                prev = elem

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: Union[Real, 'MultiInterval']) -> bool:
        return not self.overlaps(other)

    def issubset(self, other: Union[Real, 'MultiInterval']) -> bool:
        if self.is_empty:
            return True
        elif isinstance(other, MultiInterval):
            return self in other
        elif isinstance(other, Real):
            return self.is_degenerate and float(self) == other
        else:
            raise TypeError(other)

    def issuperset(self, other: Union[Real, 'MultiInterval']) -> bool:
        return other in self

    # SET: BOOLEAN ALGEBRA

    def update(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def intersection_update(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def difference_update(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def symmetric_difference_update(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def union(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def intersection(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def difference(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def symmetric_difference(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    # SET: ITEMS

    def add(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def clear(self) -> Optional['MultiInterval']:
        raise NotImplementedError

    def copy(self) -> Optional['MultiInterval']:
        raise NotImplementedError

    def discard(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    def pop(self) -> Optional['MultiInterval']:
        raise NotImplementedError

    def remove(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        raise NotImplementedError

    # INTERVAL ARITHMETIC (GENERIC)

    def _apply_monotonic_unary_function(self, func: Callable) -> 'MultiInterval':
        raise NotImplementedError

    def _apply_monotonic_binary_function(self,
                                         func: Callable,
                                         other: Union[Real, 'MultiInterval'],
                                         right_hand_side: bool = False
                                         ) -> 'MultiInterval':
        raise NotImplementedError

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __radd__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __sub__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __rsub__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __mul__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __rmul__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __rfloordiv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __rmod__(self, other):
        raise NotImplementedError

    def __divmod__(self, other):
        raise NotImplementedError

    def __rdivmod__(self, other):
        raise NotImplementedError

    def __pow__(self, power: Union[Real, 'MultiInterval'], modulo: Optional[Real] = None) -> 'MultiInterval':
        raise NotImplementedError

    def __rpow__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    # INTERVAL ARITHMETIC: INTEGERS ONLY

    def __lshift__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __rlshift__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __rshift__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    def __rrshift__(self, other: Union[Real, 'MultiInterval']) -> 'MultiInterval':
        raise NotImplementedError

    # INTERVAL ARITHMETIC: UNARY

    def reciprocal(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError

    def __invert__(self):
        raise NotImplementedError

    def __trunc__(self):  # towards 0.0
        raise NotImplementedError

    def __round__(self, n=None):  # towards nearest
        raise NotImplementedError

    def __floor__(self):  # towards -inf
        raise NotImplementedError

    def __ceil__(self):  # towards +inf
        raise NotImplementedError

    # TYPE CASTING

    def __bool__(self) -> bool:
        return not self.is_empty

    def __complex__(self) -> complex:
        if self.is_degenerate:
            return complex(float(self))
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to int')

    def __float__(self) -> float:
        if self.is_degenerate:
            return float(self.inf)
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to float')

    def __int__(self) -> int:
        if self.is_degenerate:
            return int(float(self))
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to complex')

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        def _interval_to_str(start_tuple, end_tuple):
            assert start_tuple <= end_tuple, (start_tuple, end_tuple)

            # unpack
            _start, _start_epsilon = start_tuple
            _end, _end_epsilon = end_tuple
            assert _start_epsilon in {-1, 0}, (start_tuple, end_tuple)
            assert _end_epsilon in {0, 1}, (start_tuple, end_tuple)

            # degenerate interval
            if _start == _end:
                return f'[{_start}]'

            # left side
            if _start == -math.inf:
                _start = '-∞'
                _left_bracket = '('
            else:
                _start = _start
                _left_bracket = '(' if _start_epsilon else '['

            # right side
            if _end == math.inf:
                _end = '∞'
                _right_bracket = ')'
            else:
                _end = _end
                _right_bracket = ')' if _end_epsilon else ']'

            return f'{_left_bracket}{_start}, {_end}{_right_bracket}'

        # null set: {}
        if self.is_empty:
            return '{}'

        # single contiguous interval: [x, y)
        elif len(self.endpoints) == 2:
            return _interval_to_str(*self.endpoints)  # handles degenerate intervals too

        # multiple intervals: { [x, y) | [z] | (a, b) }
        else:
            str_intervals = []
            for idx in range(0, len(self.endpoints), 2):
                str_intervals.append(_interval_to_str(self.endpoints[idx], self.endpoints[idx + 1]))

            if len(str_intervals) == 1:
                return str_intervals[0]

            return f'{{ {" | ".join(str_intervals)} }}'
