import math
import operator
import re
from numbers import Real
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

INFINITY_IS_NOT_FINITE = True


class MultiInterval:
    endpoints: List[Tuple[Real, int]]  # todo: explain epsilon

    # CONSTRUCTORS

    def __init__(self,
                 start: Optional[Real] = None,
                 end: Optional[Real] = None,
                 *,
                 start_closed: Optional[bool] = True,
                 end_closed: Optional[bool] = True
                 ):
        # todo: typecheck and sanity check and raise appropriate errors

        # no interval, create the null set
        if start is None:
            if end is not None:
                raise ValueError
            if not start_closed or not end_closed:
                raise ValueError
            self.endpoints = []

        # degenerate interval
        elif end is None and start_closed and end_closed:
            if math.isinf(start) and INFINITY_IS_NOT_FINITE:
                raise ValueError('the degenerate interval at infinity cannot exist')
            if start_closed and end_closed:
                assert not math.isinf(start)
                self.endpoints = [(start, 0), (start, 0)]
            elif not start_closed and not end_closed:
                self.endpoints = []
            else:
                raise ValueError((start, start_closed, end_closed))  # degenerate interval cannot be half open

        # null set
        elif end is None and not start_closed and not end_closed:
            self.endpoints = []

        # half-open degenerate interval makes no sense
        elif end is None:
            raise ValueError((start, start_closed, end_closed))

        # contiguous interval
        else:
            _start = (start, 0 if start_closed else 1)
            _end = (end, 0 if end_closed else -1)
            if _start > _end:
                raise ValueError((_start, _end))
            self.endpoints = [_start, _end]

        self._consistency_check()

    @classmethod
    def from_str(cls, text):
        """
        e.g. [1, 2] or [1,2]
        e.g. [0] or {0}
        e.g. {} or [] or ()
        e.g. { [1, 2) | [3, 4) } or {[1,2),[3,4)} or even [1,2)[3,4)
        """
        re_interval = re.compile(r'[\[(]\s*(?:(?:-\s*)?\d+(?:\.\d+)?\s*(?:[,|;]\s*(?:-\s*)?\d+(?:\.\d+)?\s*)?)?[)\]]',
                                 flags=re.U)
        re_set = re.compile(r'{\s*(?:(?:-\s*)?\d+(?:\.\d+)?\s*(?:[,;]\s*(?:-\s*)?\d+(?:\.\d+)?\s*)*)?}',
                            flags=re.U)
        re_num = re.compile(r'(?:-\s*)?\d+(?:\.\d+)?',
                            flags=re.U)

        out = MultiInterval()
        for interval_str in re_interval.findall(text):
            nums = re_num.findall(interval_str)
            if len(nums) == 2:
                out.update(MultiInterval(start=float(nums[0]) if '.' in nums[0] else int(nums[0]),
                                         end=float(nums[1]) if '.' in nums[1] else int(nums[1]),
                                         start_closed=interval_str[0] == '[',
                                         end_closed=interval_str[-1] == ']'))
            elif len(nums) == 1:
                if interval_str[0] == '[':
                    assert interval_str[-1] == ']'
                    out.update(MultiInterval(start=float(nums[0]) if '.' in nums[0] else int(nums[0])))
                else:
                    assert interval_str[-1] == ')'
            else:
                assert len(nums) == 0

        for set_str in re_set.findall(text):
            for num in re_num.findall(set_str):
                out.update(MultiInterval(start=float(num) if '.' in num else int(num)))

        return out

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return len(self.endpoints) == 0

    @property
    def is_degenerate(self) -> bool:
        return len(self.endpoints) == 2 and self.infimum == self.supremum

    @property
    def is_contiguous(self) -> bool:
        return len(self.endpoints) == 2

    @property
    def infimum(self) -> Optional[Real]:
        if len(self.endpoints) > 0:
            return self.endpoints[0][0]

    @property
    def supremum(self) -> Optional[Real]:
        if len(self.endpoints) > 0:
            return self.endpoints[-1][0]

    @property
    def cardinality(self) -> Tuple[int, float, int]:
        """
        intuitively this is proportional to the overall length,
        but because of infinities and degeneracy it's a 3-tuple of:
            (1) number of open half-rays from 0 (infinite length, uncountable points,   0 <= n <= 2)
            (2) remaining length of open sets   (finite length,   uncountable points, -inf < n < inf)
            (3) avg closed endpoints            (zero length,     countable points,   -inf < n < inf)

        e.g.: (1, inf]
            = (0, inf] - (0, 1]
            = (0, inf] - (0, 1) - [1]
            cardinality = (1, -1, 0)
        """
        # todo: count stuff
        half_rays = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        length = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        n_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)
        return half_rays, length, n_endpoints

    @property
    def contiguous_intervals(self) -> List['MultiInterval']:
        out = []
        for idx in range(0, len(self.endpoints), 2):
            out.append(MultiInterval(start=self.endpoints[idx][0],
                                     end=self.endpoints[idx + 1][0],
                                     start_closed=self.endpoints[idx][1] == 0,
                                     end_closed=self.endpoints[idx + 1][1] == 0))
        return out

    # COMPARISONS

    def __compare(self, func: Callable, other: Union[Real, 'MultiInterval']) -> bool:
        if isinstance(other, MultiInterval):
            return func(self.endpoints, other.endpoints)
        elif isinstance(other, Real):
            return func(self.endpoints, [(other, 0), (other, 0)])
        else:
            raise TypeError(other)

    def __lt__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.lt, other)

    def __le__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.le, other)

    def __eq__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.eq, other)

    def __ne__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.ne, other)

    def __ge__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.ge, other)

    def __gt__(self, other: Union[Real, 'MultiInterval']):
        return self.__compare(operator.gt, other)

    # UTILITY

    def __contains__(self, other: Union[Real, 'MultiInterval']) -> bool:
        raise NotImplementedError

    def expand(self, distance: Real) -> 'MultiInterval':
        raise NotImplementedError

    def closed_hull(self):
        raise NotImplementedError

    def overlaps(self, other: Union[Real, 'MultiInterval'], or_adjacent=False) -> bool:
        raise NotImplementedError

    # FILTER

    def filter(self,
               start: Real = -math.inf,
               end: Real = math.inf,
               start_closed: bool = True,
               end_closed: bool = True
               ) -> 'MultiInterval':
        raise NotImplementedError

    def __getitem__(self, item: Union[slice, 'MultiInterval']) -> 'MultiInterval':
        if isinstance(item, MultiInterval):
            return self.intersection(item)

        elif isinstance(item, slice):
            if item.step is not None:
                raise ValueError(item)

            _start = item.start or -math.inf
            if not isinstance(_start, Real):
                raise TypeError(_start)

            _end = item.stop or math.inf
            if not isinstance(_end, Real):
                raise TypeError(_end)

            return self.filter(start=_start, end=_end)

        else:
            raise TypeError

    def positive(self) -> 'MultiInterval':
        return self.filter(start=0, start_closed=False)

    def negative(self) -> 'MultiInterval':
        return self.filter(end=0, end_closed=False)

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

        # no degenerate interval at infinity
        if self.is_degenerate:
            assert not math.isinf(self.infimum)

        # infinity cannot be contained within an interval
        assert (-math.inf, 0) > self.endpoints[0]
        assert self.endpoints[-1] < (math.inf, 0)
        assert math.inf not in self
        assert -math.inf not in self

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
        self.update(other)
        return self

    def clear(self) -> Optional['MultiInterval']:
        self.endpoints.clear()
        return self

    def copy(self) -> Optional['MultiInterval']:
        out = MultiInterval()
        out.endpoints = self.endpoints.copy()
        return out

    def discard(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        if other in self:
            self.difference_update(other)
        return self

    def pop(self) -> Optional['MultiInterval']:
        if self.is_empty:
            raise KeyError('pop from empty MultiInterval')

        out = MultiInterval()
        self.endpoints, out.endpoints = self.endpoints[:-2], self.endpoints[-2:]
        return out

    def remove(self, other: Union[Real, 'MultiInterval']) -> Optional['MultiInterval']:
        if other not in self:
            raise KeyError(other)
        self.difference_update(other)
        return self

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
            return float(self.infimum)
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
        def _interval_to_str(start_tuple, end_tuple, fancy_inf=False):
            assert start_tuple <= end_tuple, (start_tuple, end_tuple)

            # unpack
            _start, _start_epsilon = start_tuple
            _end, _end_epsilon = end_tuple
            assert _start_epsilon in {-1, 0}, (start_tuple, end_tuple)
            assert _end_epsilon in {0, 1}, (start_tuple, end_tuple)

            # degenerate interval
            if _start == _end:
                assert _start_epsilon == 0
                assert _end_epsilon == 0
                return f'[{_start}]'

            # print the mathematically standard but logically inconsistent way
            if fancy_inf:
                # left side
                if _start == -math.inf:
                    assert _start_epsilon == 0
                    _start = '-∞'
                    _left_bracket = '('
                else:
                    _left_bracket = '(' if _start_epsilon else '['

                # right side
                if _end == math.inf:
                    assert _end_epsilon == 0
                    _end = '∞'
                    _right_bracket = ')'
                else:
                    _right_bracket = ')' if _end_epsilon else ']'

                return f'{_left_bracket}{_start}, {_end}{_right_bracket}'

            # for the sake of consistency, a closed endpoint at ±inf will be square instead of round
            else:
                return f'{"(" if _start_epsilon else "["}{_start}, {_end}{")" if _end_epsilon else "]"}'

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
