import math
import operator
import re
from numbers import Real
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

INFINITY_IS_NOT_FINITE = True  # don't allow ±inf to be contained inside intervals


def _str_to_num(text: str):
    text = text.strip()
    if text.isdigit():
        return int(text)
    else:
        return float(text)


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

        # no interval, create the null set
        if start is None:
            if end is not None:
                raise ValueError
            if start_closed != end_closed:
                raise ValueError
            self.endpoints = []

        # null set
        elif end is None and not start_closed and not end_closed:
            self.endpoints = []

        # degenerate interval
        elif end is None and start_closed and end_closed:
            if math.isinf(start) and INFINITY_IS_NOT_FINITE:
                raise ValueError('the degenerate interval at infinity cannot exist')
            else:
                self.endpoints = [(start, 0), (start, 0)]

        # half-open degenerate interval makes no sense
        elif end is None:
            raise ValueError((start, start_closed, end_closed))

        # infinity
        elif math.isinf(start) and start_closed and INFINITY_IS_NOT_FINITE:
            raise ValueError(f'{start} cannot be contained in Interval')
        elif math.isinf(end) and end_closed and INFINITY_IS_NOT_FINITE:
            raise ValueError(f'{start} cannot be contained in Interval')
        elif start == math.inf and not start_closed:
            raise ValueError('cannot start an Interval after inf')
        elif end == -math.inf and not end_closed:
            raise ValueError('cannot end an Interval before -inf')

        # contiguous interval (possibly degenerate)
        else:
            _start = (start, 0 if start_closed else 1)
            _end = (end, 0 if end_closed else -1)
            if _start > _end:
                raise ValueError(f'Interval start {_start} cannot be before end {_end}')
            self.endpoints = [_start, _end]

        self._consistency_check()

    @classmethod
    def from_str(cls, text) -> 'MultiInterval':
        """
        e.g. [1, 2] or [1,2]
        e.g. [0] or {0}
        e.g. {} or [] or () or (123)
        e.g. { [1, 2) | [3, 4) } or {[1,2),[3,4)} or even [1,2)[3,4)
        """
        re_num = re.compile(r'(?:-\s*)?(?:inf|\d+(?:\.\d+)?(?:e-?\d+)?)\s*', flags=re.U)
        re_interval = re.compile(fr'[\[(]\s*(?:{re_num.pattern}(?:[,;]\s*{re_num.pattern})?)?[)\]]', flags=re.U)
        re_set = re.compile(fr'{{\s*(?:{re_num.pattern}(?:[,;]\s*{re_num.pattern})*)?}}', flags=re.U)

        out = MultiInterval()
        for interval_str in re_interval.findall(text):
            _start_closed = interval_str[0] == '['
            _end_closed = interval_str[-1] == ']'
            _nums = re_num.findall(interval_str)

            if len(_nums) == 2:
                out.update(MultiInterval(start=_str_to_num(_nums[0]),
                                         end=_str_to_num(_nums[1]),
                                         start_closed=_start_closed,
                                         end_closed=_end_closed))
            elif len(_nums) == 1:
                out.update(MultiInterval(start=_str_to_num(_nums[0]),
                                         start_closed=_start_closed,
                                         end_closed=_end_closed))
            elif len(_nums) > 0:
                raise ValueError(f'Interval can only have 2 endpoints: {interval_str}')

        for set_str in re_set.findall(text):
            for num in re_num.findall(set_str):
                out.update(MultiInterval(start=_str_to_num(num)))

        out._consistency_check()
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
            (1) number of open half-rays from 0 (infinite length, uncountable points, int,     0 <= n <= 2)
            (2) remaining length of open sets   (finite length,   uncountable points, float, -inf < n < inf)
            (3) avg closed endpoints            (zero length,     countable points,   int,   -inf < n < inf)

        e.g.: (1, inf]
            = (0, inf] - (0, 1]
            = (0, inf] - (0, 1) - [1]
            cardinality = (1, -1, 0)

        invariants that don't change overall cardinality:
            *   moving an interval:               [1, 2) -> [2, 3)
            *   extracting a degenerate interval: [1, 2) -> [1] + (1, 2)
            *   splitting an interval:            [1, 3) -> [1, 2) + [2, 3)
            *   joining intervals: [1, 2) + [2] + (2, 3) -> [1, 3)
            *   mirroring interval:               [1, 2) -> (-2, -1]
        """
        negative_half_ray = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        positive_half_ray = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        length_before_zero = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        length_after_zero = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        closed_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)
        open_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)

        # iterate through to count all the things
        for idx in range(0, len(self.endpoints), 2):
            _start, _start_epsilon = self.endpoints[idx]
            _end, _end_epsilon = self.endpoints[idx + 1]
            assert _start <= _end

            # check if half-rays exist
            if _start == -math.inf < float(_end):
                negative_half_ray += 1
            if _start < math.inf == _end:
                positive_half_ray += 1

            # count length
            if _start < _end <= 0:
                length_before_zero += _end - _start
            elif _start <= 0 <= float(_end):
                length_before_zero -= _start
                length_after_zero += _end
            elif 0 <= float(_start) < float(_end):
                length_after_zero += _end - _start

            # count endpoints
            if _end_epsilon == 0:
                closed_endpoints += 1
            else:
                open_endpoints += 1
            if _start_epsilon == 0:
                closed_endpoints += 1
            else:
                open_endpoints += 1

        # subtract half-rays
        if negative_half_ray:
            assert negative_half_ray == 1  # sanity check
            length_before_zero = -length_before_zero
        if positive_half_ray:
            assert positive_half_ray == 1  # sanity check
            length_after_zero = -length_after_zero

        # return 3-tuple
        return (negative_half_ray + positive_half_ray,
                length_before_zero + length_after_zero,
                closed_endpoints - open_endpoints)

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

    def __compare(self, func: Callable, other: Union['MultiInterval', Real]) -> bool:
        if isinstance(other, MultiInterval):
            return func(self.endpoints, other.endpoints)
        elif isinstance(other, Real):
            return func(self.endpoints, [(other, 0), (other, 0)])
        else:
            raise TypeError(other)

    def __lt__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.lt, other)

    def __le__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.le, other)

    def __eq__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.eq, other)

    def __ne__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.ne, other)

    def __ge__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.ge, other)

    def __gt__(self, other: Union['MultiInterval', Real]) -> bool:
        return self.__compare(operator.gt, other)

    # UTILITY

    def copy(self) -> 'MultiInterval':
        out = MultiInterval()
        out.endpoints = self.endpoints.copy()
        return out

    def __sizeof__(self) -> int:
        return self.endpoints.__sizeof__()  # probably correct?

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

    # FILTERING

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

            return self.intersection(MultiInterval(start=_start,
                                                   end=_end,
                                                   start_closed=not math.isinf(_start) or not INFINITY_IS_NOT_FINITE,
                                                   end_closed=not math.isinf(_end) or not INFINITY_IS_NOT_FINITE))

        else:
            raise TypeError

    def positive(self) -> 'MultiInterval':
        return self.intersection(MultiInterval(start=0,
                                               end=math.inf,
                                               start_closed=False,
                                               end_closed=not INFINITY_IS_NOT_FINITE))

    def negative(self) -> 'MultiInterval':
        return self.intersection(MultiInterval(start=-math.inf,
                                               end=0,
                                               start_closed=not INFINITY_IS_NOT_FINITE,
                                               end_closed=False))

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: Union['MultiInterval', Real]) -> bool:
        return not self.overlaps(other)

    def issubset(self, other: Union['MultiInterval', Real]) -> bool:
        if self.is_empty:
            return True
        elif isinstance(other, MultiInterval):
            return self in other
        elif isinstance(other, Real):
            return self.is_degenerate and float(self) == other
        else:
            raise TypeError(other)

    def issuperset(self, other: Union['MultiInterval', Real]) -> bool:
        return other in self

    # SET: ITEMS (INPLACE)

    def add(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        # todo: use O(log(n)) algo instead
        self.update(other)
        return self

    def clear(self) -> 'MultiInterval':
        self._consistency_check()
        self.endpoints.clear()
        return self

    def discard(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        if other in self:
            self.difference_update(other)
        return self

    def pop(self) -> 'MultiInterval':
        if self.is_empty:
            raise KeyError('pop from empty MultiInterval')

        out = MultiInterval()
        self.endpoints, out.endpoints = self.endpoints[:-2], self.endpoints[-2:]
        self._consistency_check()
        return out

    def remove(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        if other not in self:
            raise KeyError(other)
        self.difference_update(other)
        return self

    # SET: BOOLEAN ALGEBRA (INPLACE)

    def update(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        self._consistency_check()
        _endpoints = self.endpoints.copy()

        for _other in other:
            if isinstance(_other, MultiInterval):
                _other._consistency_check()
                _endpoints.extend(_other.endpoints)
            elif isinstance(_other, Real):
                _endpoints.append((_other, 0))
                _endpoints.append((_other, 0))
            else:
                raise TypeError(_other)

        self.endpoints = _endpoints
        self.merge_adjacent(sort=True)
        return self

    def intersection_update(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def difference_update(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def symmetric_difference_update(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        tmp = other.difference(self)
        self.difference_update(other)
        self.update(tmp)
        self._consistency_check()
        return self

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().update(*other)

    def intersection(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().intersection_update(other)

    def difference(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().difference_update(other)

    def symmetric_difference(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().symmetric_difference_update(other)

    # INTERVAL OPERATIONS (INPLACE)

    def merge_adjacent(self, distance: Real = 0, sort: bool = False) -> 'MultiInterval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        if self.is_empty or self.is_contiguous:
            return self

        if sort:
            _endpoints, self.endpoints = self.endpoints, []
            for _start, _end in sorted((_endpoints[idx], _endpoints[idx + 1]) for idx in range(0, len(_endpoints), 2)):
                self.endpoints.append(_start)
                self.endpoints.append(_endpoints)

        _endpoints, self.endpoints = self.endpoints, []
        idx = 0
        _end = _endpoints[0]
        while idx < len(_endpoints):
            self.endpoints.append(_endpoints[idx])
            _end = max(_end, _endpoints[idx + 1])
            idx += 2

            while idx < len(_endpoints) and _endpoints[idx] <= (_end[0] + distance, _end[1] + 1):
                _end = max(_end, _endpoints[idx + 1])
                idx += 2

            self.endpoints.append(_end)

        # allow operator chaining
        self._consistency_check()
        return self

    def abs(self) -> 'MultiInterval':
        # todo: fix this freakishly inefficient implementation
        self.update(self.negative().mirror())
        self.endpoints = self[0:].endpoints
        return self

    def invert(self) -> 'MultiInterval':
        self.symmetric_difference_update(MultiInterval(start=-math.inf,
                                                       end=math.inf,
                                                       start_closed=not INFINITY_IS_NOT_FINITE,
                                                       end_closed=not INFINITY_IS_NOT_FINITE))
        return self

    def mirror(self) -> 'MultiInterval':
        _endpoints, self.endpoints = self.endpoints[::-1], []
        for point, epsilon in _endpoints:
            self.endpoints.append((-point, -epsilon))
        return self

    def expand(self, distance: Real) -> 'MultiInterval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        for idx in range(0, len(self.endpoints), 2):
            self.endpoints[idx] = (self.endpoints[idx][0] - distance, self.endpoints[idx][1])
            self.endpoints[idx + 1] = (self.endpoints[idx + 1][0] + distance, self.endpoints[idx + 1][1])

        self.merge_adjacent()
        return self

    # INTERVAL OPERATIONS

    def __contains__(self, other: Union['MultiInterval', Real]) -> bool:
        raise NotImplementedError

    def closed_hull(self):
        return MultiInterval(start=self.infimum,
                             end=self.supremum,
                             start_closed=not math.isinf(self.infimum) or not INFINITY_IS_NOT_FINITE,
                             end_closed=not math.isinf(self.supremum) or not INFINITY_IS_NOT_FINITE)

    def overlaps(self, other: Union['MultiInterval', Real], or_adjacent=False) -> bool:
        raise NotImplementedError

    # INTERVAL ARITHMETIC: GENERIC

    def _apply_monotonic_unary_function(self, func: Callable, inplace: bool = False) -> 'MultiInterval':
        # by default, do this to a copy
        if not inplace:
            return self.copy()._apply_monotonic_unary_function(func, inplace=True)

        self._consistency_check()
        _endpoints, self.endpoints = self.endpoints, []

        for idx in range(0, len(_endpoints), 2):
            _start = func(_endpoints[idx][0])
            _end = func(_endpoints[idx + 1][0])
            _start_epsilon = _endpoints[idx][1]
            _end_epsilon = _endpoints[idx + 1][1]
            self.endpoints.append(min((_start, _start_epsilon), (_end, -_end_epsilon)))
            self.endpoints.append(max((_start, -_start_epsilon), (_end, _end_epsilon)))

        self.merge_adjacent(sort=True)
        return self

    def _apply_monotonic_binary_function(self,
                                         func: Callable,
                                         other: Union['MultiInterval', Real],
                                         right_hand_side: bool = False,
                                         inplace: bool = False
                                         ) -> 'MultiInterval':
        # by default, do this to a copy
        if not inplace:
            return self.copy()._apply_monotonic_binary_function(func,
                                                                other,
                                                                right_hand_side=right_hand_side,
                                                                inplace=True)

        # get other's interval list as [((start, eps), (end, eps)), ...]
        if isinstance(other, MultiInterval):
            _second = [(other.endpoints[idx], other.endpoints[idx + 1]) for idx in range(0, len(other.endpoints), 2)]
        elif isinstance(other, Real):
            _second = [((other, 0), (other, 0))]
        else:
            raise TypeError(other)

        # clear own endpoints while reading, since we'll replace everything
        _endpoints, self.endpoints = self.endpoints, []
        _first = [(_endpoints[idx], _endpoints[idx + 1]) for idx in range(0, len(_endpoints), 2)]

        # swap for RHS
        if right_hand_side:
            _first, _second = _second, _first

        # union of: func(x, y) for x in first for y in second
        for (_first_start, _first_start_epsilon), (_first_end, _first_end_epsilon) in _first:
            for (_second_start, _second_start_epsilon), (_second_end, _second_end_epsilon) in _second:
                _start_start = func(_first_start, _second_start)
                _start_end = func(_first_start, _second_end)
                _end_start = func(_first_end, _second_start)
                _end_end = func(_first_end, _second_end)

                self.endpoints.append(min((_start_start, _first_start_epsilon or _second_start_epsilon),
                                          (_start_end, _first_start_epsilon or -_second_end_epsilon),
                                          (_end_start, -_first_end_epsilon or _second_start_epsilon),
                                          (_end_end, -_first_end_epsilon or -_second_end_epsilon)))

                self.endpoints.append(max((_start_start, -_first_start_epsilon or -_second_start_epsilon),
                                          (_start_end, -_first_start_epsilon or _second_end_epsilon),
                                          (_end_start, _first_end_epsilon or -_second_start_epsilon),
                                          (_end_end, _first_end_epsilon or _second_end_epsilon)))

        # we may be out of order or have overlapping intervals, so merge
        self.merge_adjacent(sort=True)
        return self

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.add, other)

    def __radd__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.add, other, right_hand_side=True)

    def __sub__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.sub, other)

    def __rsub__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.sub, other, right_hand_side=True)

    def __mul__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.mul, other)

    def __rmul__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.mul, other, right_hand_side=True)

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

    # def __pow__(self, power: Union['MultiInterval', Real], modulo: Optional[Real] = None) -> 'MultiInterval':
    def __pow__(self, power: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def __rpow__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    # INTERVAL ARITHMETIC: INTEGERS ONLY

    def __lshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.lshift, other)

    def __rlshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.lshift, other, right_hand_side=True)

    def __rshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.rshift, other)

    def __rrshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self._apply_monotonic_binary_function(operator.rshift, other, right_hand_side=True)

    # INTERVAL ARITHMETIC: UNARY

    def reciprocal(self):
        raise NotImplementedError

    def __neg__(self):
        return self.copy().mirror()

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        return self.copy().abs()

    def __invert__(self):
        return self.copy().invert()

    def exp(self):
        return self._apply_monotonic_unary_function(math.exp)

    def log(self, base: float = math.e):
        return self._apply_monotonic_unary_function(lambda num: math.log(num, base=base))

    def __round__(self, n_digits: int = 0):  # towards nearest integer
        return self._apply_monotonic_unary_function(lambda num: round(num, ndigits=n_digits))

    def __trunc__(self):  # towards 0.0
        return self._apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.trunc(num))

    def __floor__(self):  # towards -inf
        return self._apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.floor(num))

    def __ceil__(self):  # towards +inf
        return self._apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.ceil(num))

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
