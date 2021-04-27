import bisect
import math
import operator
import random
import re
import time
import warnings
from numbers import Real
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

# this affects whether it's possible to create an interval closed at infinity
# mathematically it doesn't make sense, but it's possible to represent and work with
# turning this flag off should never be needed in practice, and hence isn't recommended
INFINITY_IS_NOT_FINITE = True  # don't allow ±inf to be contained inside intervals


class MultiInterval:
    """
    represents zero or more non-overlapping intervals
    each interval can be degenerate or infinite
    adjacent intervals are merged automatically if you use the provided functions

    each interval is represented as a pair of endpoints
    each endpoint is represented as a tuple of the point's location and epsilon
    where epsilon is either -1, 0, or 1

    epsilon helps to represent whether the point is open or closed in a way that simplifies the algorithms
        *   epsilon=0 represents a closed start or end point
        *   epsilon=1 represents an open start point
        *   epsilon=-1 represents an open end point
    this representation makes comparing two points trivial, regardless of whether they are start or end points
    and this is important because it allows interval union to run in linear time (for 2 intervals)
    """
    endpoints: List[Tuple[Real, int]]  # (location, epsilon)

    # CONSTRUCTORS

    def __init__(self,
                 start: Optional[Real] = None,
                 end: Optional[Real] = None,
                 *,
                 start_closed: Optional[bool] = True,
                 end_closed: Optional[bool] = True
                 ):

        # handle nan
        if end is not None and math.isnan(end):
            end = None
        if start is not None and math.isnan(start):
            start, end = end, None

        # no interval, create the null set
        if start is None:
            if end is not None:
                raise ValueError
            if start_closed != end_closed:
                raise ValueError
            self.endpoints = []

        # null set
        elif end is None and not start_closed and not end_closed:
            warnings.warn(f'creating a closed interval: ({start}) == []')
            self.endpoints = []

        # degenerate interval
        elif end is None and start_closed and end_closed:
            assert start is not None  # convince type checker
            if math.isinf(start) and INFINITY_IS_NOT_FINITE:
                raise ValueError('the degenerate interval at infinity cannot exist')
            else:
                if start == 0 and math.copysign(1.0, start) == -1.0:
                    warnings.warn('negative zero will be converted to zero')
                    start = 0.0
                self.endpoints = [(start, 0), (start, 0)]

        # half-open degenerate interval makes no sense
        elif end is None:
            raise ValueError((start, start_closed, end_closed))

        # infinity
        # nullity checks are unnecessary, but help to convince the type checker
        elif start is not None and math.isinf(start) and start_closed and INFINITY_IS_NOT_FINITE:
            raise ValueError(f'{start} cannot be contained in Interval')
        elif end is not None and math.isinf(end) and end_closed and INFINITY_IS_NOT_FINITE:
            raise ValueError(f'{start} cannot be contained in Interval')
        elif start == math.inf and not start_closed:
            raise ValueError('cannot start an Interval after inf')
        elif end == -math.inf and not end_closed:
            raise ValueError('cannot end an Interval before -inf')

        # contiguous interval (possibly degenerate)
        else:
            assert end is not None  # convince type checker
            if start == 0 and math.copysign(1.0, start) == -1.0:
                warnings.warn('negative zero will be converted to zero')
                start = 0.0
                if end == 0 and math.copysign(1.0, end) == -1.0:
                    end = 0.0
            elif end == 0 and math.copysign(1.0, end) == -1.0:
                warnings.warn('negative zero will be converted to zero')
                end = 0.0

            _start = (start, 0 if start_closed else 1)
            _end = (end, 0 if end_closed else -1)
            if _start > _end:
                raise ValueError(f'Interval start {_start} is after end {_end}')
            self.endpoints = [_start, _end]

        self._consistency_check()

    @classmethod
    def merge(cls,
              *interval: Union['MultiInterval', Real, Set[int], List[int], Tuple[int, int], str],
              n_overlaps: Optional[Union[int, Iterable[int]]] = None
              ) -> 'MultiInterval':

        # check counts
        _overlaps = None
        if n_overlaps is not None:
            if isinstance(n_overlaps, int):
                _overlaps = {n_overlaps}
            elif isinstance(n_overlaps, Iterable):
                _overlaps = set(n_overlaps)
                if not all(isinstance(elem, int) for elem in _overlaps):
                    raise TypeError(n_overlaps)
                if any(elem <= 0 for elem in _overlaps):
                    raise ValueError(n_overlaps)
            else:
                raise TypeError(n_overlaps)

        # check intervals
        _points = []
        n_intervals = 0
        for _interval in interval:
            n_intervals += 1

            # is an interval
            if isinstance(_interval, MultiInterval):
                for idx in range(0, len(_interval.endpoints), 2):
                    _points.append((_interval.endpoints[idx], False))
                    _points.append((_interval.endpoints[idx + 1], True))

            # is a real number (degenerate interval)
            elif isinstance(_interval, Real):
                _points.append(((_interval, 0), False))
                _points.append(((_interval, 0), True))

            # is a set (of real numbers)
            elif isinstance(_interval, Set):
                for elem in _interval:
                    if not isinstance(elem, Real):
                        raise TypeError(elem)
                    _points.append(((elem, 0), False))
                    _points.append(((elem, 0), True))

            # is a list of 2 reals (shorthand for closed interval)
            elif isinstance(_interval, List):
                if len(_interval) == 1:  # degenerate case
                    if not isinstance(_interval[0], Real):
                        raise TypeError(_interval[0])
                    _points.append(((_interval[0], 0), False))
                    _points.append(((_interval[0], 0), True))
                elif len(_interval) == 2:  # non-degenerate case
                    if not isinstance(_interval[0], Real):
                        raise TypeError(_interval[0])
                    if not isinstance(_interval[1], Real):
                        raise TypeError(_interval[1])
                    _points.append(((_interval[0], 0), False))
                    _points.append(((_interval[1], 0), True))
                elif len(_interval) > 2:
                    raise TypeError(_interval)

            # is a tuple (shorthand for open interval)
            elif isinstance(_interval, Tuple):
                if len(_interval) == 2:
                    if not isinstance(_interval[0], Real):
                        raise TypeError(_interval[0])
                    if not isinstance(_interval[1], Real):
                        raise TypeError(_interval[1])
                    _points.append(((_interval[0], 1), False))
                    _points.append(((_interval[1], -1), True))
                elif len(_interval) > 2:
                    raise TypeError(_interval)

            # is a string
            # maybe refactor this out into a from_string function?
            elif isinstance(_interval, str):
                # e.g. [1, 2] or [1,2]
                # e.g. [0] or {0}
                # e.g. {} or [] or () or (123)
                # e.g. { [1, 2) | [3, 4) } or {[1,2),[3,4)} or even [1,2)[3,4)

                re_num = re.compile(r'(?:-\s*)?(?:inf|\d+(?:\.\d+)?(?:e-?\d+)?)\s*', flags=re.U)
                re_interval = re.compile(fr'[\[(]\s*(?:{re_num.pattern}(?:[,;]\s*{re_num.pattern})?)?[)\]]', flags=re.U)
                re_set = re.compile(fr'{{\s*(?:{re_num.pattern}(?:[,;]\s*{re_num.pattern})*)?}}', flags=re.U)

                def _str_to_num(_num: str) -> Union[int, float]:
                    _num = _num.strip()
                    if ''.join(_num.split()).lstrip('-').isdigit():
                        return int(_num)
                    else:
                        return float(_num)

                out = MultiInterval()
                for interval_str in re_interval.findall(_interval):
                    _start_closed = interval_str[0] == '['
                    _end_closed = interval_str[-1] == ']'
                    _nums = re_num.findall(interval_str)

                    if len(_nums) == 1:
                        if _start_closed != _end_closed:
                            raise ValueError(interval_str)
                        elif _start_closed:
                            out.update(MultiInterval(start=_str_to_num(_nums[0])))

                    elif len(_nums) == 2:
                        out.update(MultiInterval(start=_str_to_num(_nums[0]),
                                                 end=_str_to_num(_nums[1]),
                                                 start_closed=_start_closed,
                                                 end_closed=_end_closed))
                    elif len(_nums) > 2:
                        raise ValueError(f'Interval can only have 2 endpoints: {interval_str}')

                for set_str in re_set.findall(_interval):
                    for num in re_num.findall(set_str):
                        out.update(MultiInterval(start=_str_to_num(num)))

                out._consistency_check()
                for idx in range(0, len(out.endpoints), 2):
                    _points.append((out.endpoints[idx], False))
                    _points.append((out.endpoints[idx + 1], True))

            # is something else, invalid
            else:
                raise TypeError(_interval)

        # set counts if not yet set
        if _overlaps is None:
            _overlaps = set(range(1, n_intervals + 1))

        # sort points
        _points = sorted(_points)
        if len(_points) == 0:
            return MultiInterval()
        out = MultiInterval()

        # go through all the endpoints and count how many are open for each interval
        start, is_end = _points[0]
        assert is_end is False
        count_overlaps = 1
        for point, is_end in _points[1:]:
            # found an endpoint, only break off interval if we didn't just break it, then decrement count
            if is_end:
                assert start <= (point[0], point[1] + 1), (start, point, _points)
                if count_overlaps in _overlaps and start <= point:
                    out.endpoints.append(start)
                    out.endpoints.append(point)
                start = (point[0], point[1] + 1)
                count_overlaps -= 1

            # found the same start point, just increment count
            elif point == start:
                assert point >= start, (start, point, _points)
                count_overlaps += 1
                continue

            # found a new start point, break off current interval and increment count
            else:
                assert point >= start, (start, point, _points)
                if count_overlaps in _overlaps:
                    out.endpoints.append(start)
                    out.endpoints.append((point[0], point[1] - 1))
                start = point
                count_overlaps += 1

        # merge all the intervals we kept
        assert count_overlaps == 0
        out.merge_adjacent()
        return out

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        self._consistency_check()
        return len(self.endpoints) == 0

    @property
    def is_contiguous(self) -> bool:
        """
        check if this is a non-empty contiguous interval
        a singleton degenerate interval is considered contiguous

        note that { [1] , (1, 2] } is contiguous
        while it's not possible to create such an interval using the provided methods, you can manually make one
        hence we need to merge adjacent intervals before we can do this check
        """
        # note that { [1] , (1, 2] } is contiguous
        return len(self.copy().merge_adjacent().endpoints) == 2

    @property
    def is_degenerate(self) -> bool:
        """
        multi-interval is degenerate if it is not empty and all contained intervals are degenerate
        """
        if self.is_empty:  # this runs _consistency_check
            return False
        else:
            return all(self.endpoints[idx] == self.endpoints[idx + 1] for idx in range(0, len(self.endpoints), 2))

    @property
    def is_finite(self) -> bool:
        """
        checks if the start or end point is at infinity
        ignores INFINITY_IS_NOT_FINITE because it would be silly to always return False in that scenario
        """
        return self.is_empty or not (math.isinf(self.infimum) or math.isinf(self.supremum))

    @property
    def is_integral(self) -> bool:
        """
        checks whether this is a non-empty set of integers (ie. degenerate points at integers)
        """
        if self.is_empty or not self.is_finite:
            return False
        for idx in range(0, len(self.endpoints), 2):
            if self.endpoints[idx] != self.endpoints[idx + 1]:  # check degeneracy
                return False
            if self.endpoints[idx][0] % 1 != 0:  # check that it has no fractional part
                return False
        return True

    @property
    def is_positive(self) -> bool:
        return not self.is_empty and self.endpoints[0] > (0, 0)

    @property
    def is_negative(self) -> bool:
        return not self.is_empty and self.endpoints[-1] < (0, 0)

    @property
    def is_non_negative(self) -> bool:
        return self.is_empty or self.endpoints[0] >= (0, 0)

    @property
    def is_non_positive(self) -> bool:
        return self.is_empty or self.endpoints[-1] <= (0, 0)

    @property
    def finite(self) -> 'MultiInterval':
        """
        returns a new MultiInterval made from all finite contained intervals
        ignores INFINITY_IS_NOT_FINITE because it would be silly
        """
        self._consistency_check()
        out = MultiInterval()
        for idx in range(0, len(self.endpoints), 2):
            if not (math.isinf(self.endpoints[idx][0]) or math.isinf(self.endpoints[idx + 1][0])):
                out.endpoints.append(self.endpoints[idx])
                out.endpoints.append(self.endpoints[idx + 1])
        out._consistency_check()
        out.merge_adjacent()
        return out

    @property
    def positive(self) -> 'MultiInterval':
        return self.intersection(MultiInterval(start=0,
                                               end=math.inf,
                                               start_closed=False,  # zero is not positive
                                               end_closed=not INFINITY_IS_NOT_FINITE))

    @property
    def negative(self) -> 'MultiInterval':
        return self.intersection(MultiInterval(start=-math.inf,
                                               end=0,
                                               start_closed=not INFINITY_IS_NOT_FINITE,
                                               end_closed=False))  # zero is not positive

    @property
    def infimum(self) -> Real:
        if self.is_empty:
            raise KeyError('no infimum in empty interval')  # basically min of empty list
        return self.endpoints[0][0]

    @property
    def infimum_is_closed(self) -> bool:
        if self.is_empty:
            raise KeyError('no infimum in empty interval')  # basically min of empty list
        return self.endpoints[0][1] == 0

    @property
    def supremum(self) -> Real:
        if self.is_empty:
            raise KeyError('no supremum in empty interval')  # basically max of empty list
        return self.endpoints[-1][0]

    @property
    def supremum_is_closed(self) -> bool:
        if self.is_empty:
            raise KeyError('no supremum in empty interval')  # basically max of empty list
        return self.endpoints[-1][1] == 0

    @property
    def degenerate_points(self) -> Set[Real]:
        if self.is_empty:
            return set()
        else:
            return set([self.endpoints[idx][0]
                        for idx in range(0, len(self.endpoints), 2)
                        if self.endpoints[idx] == self.endpoints[idx + 1]])

    @property
    def closed_hull(self) -> Optional['MultiInterval']:
        """
        smallest contiguous cover
        closed at endpoints unless the point is at infinity
        """
        if not self.is_empty:
            return MultiInterval(start=self.infimum,
                                 end=self.supremum,
                                 start_closed=not (math.isinf(self.infimum) and INFINITY_IS_NOT_FINITE),
                                 end_closed=not (math.isinf(self.supremum) and INFINITY_IS_NOT_FINITE))

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
        self._consistency_check()

        negative_half_ray = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        positive_half_ray = 0  # only (-inf, 0) and/or (0, inf), if -inf or inf are included
        length_before_zero = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        length_after_zero = 0  # remaining length of open intervals, after removing half-rays (can be negative)
        closed_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)
        open_endpoints = 0  # number of endpoints, after removing open intervals (can be negative)

        # iterate through to count all the things
        for idx in range(0, len(self.endpoints), 2):
            start, start_epsilon = self.endpoints[idx]
            end, end_epsilon = self.endpoints[idx + 1]
            assert start <= end

            # check if half-rays exist
            if start == -math.inf < float(end):
                negative_half_ray += 1
            if start < math.inf == end:
                positive_half_ray += 1

            # count length
            if start < end <= 0:
                length_before_zero += end - start
            elif start <= 0 <= float(end):
                length_before_zero -= start
                length_after_zero += end
            elif 0 <= float(start) < float(end):
                length_after_zero += end - start

            # count endpoints
            if end_epsilon == 0:
                closed_endpoints += 1
            else:
                open_endpoints += 1
            if start_epsilon == 0:
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
        """
        compares two intervals as if they were lists
        not especially meaningful but might be helpful for very similar intervals
        does not call merge_adjacent before comparison
        """
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
        self._consistency_check()
        out = MultiInterval()
        out.endpoints = self.endpoints.copy()
        return out

    def __sizeof__(self) -> int:
        return self.endpoints.__sizeof__()  # probably correct? not sure if it counts the size of this obj though

    def _consistency_check(self, skip=False):
        # switch this to default to True for actual use
        if skip:
            return

        # length must be even
        assert len(self.endpoints) % 2 == 0, len(self.endpoints)

        # must be sorted
        if len(self.endpoints) > 0:

            def _check_endpoint_tuple(endpoint):
                assert isinstance(endpoint, tuple), endpoint
                assert len(endpoint) == 2, endpoint
                assert isinstance(endpoint[0], Real), endpoint
                assert isinstance(endpoint[1], int), endpoint
                assert endpoint[1] in {-1, 0, 1}, endpoint
                if endpoint[0] == 0 and math.copysign(1.0, endpoint[0]) == -1.0:
                    warnings.warn('negative zero exists in this interval')
                assert not (math.isinf(endpoint[0]) and endpoint[1] == 0 and INFINITY_IS_NOT_FINITE)

            prev = self.endpoints[0]
            _check_endpoint_tuple(prev)

            for elem in self.endpoints[1:]:
                _check_endpoint_tuple(elem)
                assert prev <= elem, (prev, elem)
                prev = elem

            # no degenerate interval at infinity
            if self.endpoints[0] == self.endpoints[-1]:
                assert not (math.isinf(self.endpoints[0][0]) and INFINITY_IS_NOT_FINITE)

            # infinity cannot be contained within an interval
            if INFINITY_IS_NOT_FINITE:
                assert (-math.inf, 0) < self.endpoints[0]
                assert self.endpoints[-1] < (math.inf, 0)
            else:
                assert (-math.inf, 0) <= self.endpoints[0]
                assert self.endpoints[-1] <= (math.inf, 0)

    def __getitem__(self, item: Union[slice, 'MultiInterval', Real]) -> 'MultiInterval':
        if isinstance(item, MultiInterval):
            return self.intersection(item)

        elif isinstance(item, slice):
            if item.step is not None:
                raise TypeError(item)
            if not (item.start is None or isinstance(item.start, Real)):
                raise TypeError(item)
            if not (item.stop is None or isinstance(item.stop, Real)):
                raise TypeError(item)

            start = item.start or -math.inf
            start_epsilon = 1 if (math.isinf(start) and INFINITY_IS_NOT_FINITE) else 0
            end = item.stop or math.inf
            end_epsilon = -1 if (math.isinf(end) and INFINITY_IS_NOT_FINITE) else 0

            if (start, start_epsilon) > (end, end_epsilon):
                raise ValueError('slice cannot go backwards')

            out = MultiInterval()
            left_idx = bisect.bisect_left(self.endpoints, (start, start_epsilon))
            if left_idx % 2 == 1:
                out.endpoints.append((start, start_epsilon))
            right_idx = bisect.bisect_right(self.endpoints, (end, end_epsilon), lo=left_idx)
            out.endpoints.extend(self.endpoints[left_idx:right_idx])
            if right_idx % 2 == 1:
                out.endpoints.append((end, end_epsilon))

            out._consistency_check()
            return out

        elif isinstance(item, Real):
            if item in self:
                return MultiInterval(item)
            else:
                return MultiInterval()

        else:
            raise TypeError

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: Union['MultiInterval', Real]) -> bool:
        return not self.overlaps(other)

    def issubset(self, other: Union['MultiInterval', Real]) -> bool:
        # i.e. other contains all elements of self
        if self.is_empty:
            return True
        elif isinstance(other, MultiInterval):
            return self in other
        elif isinstance(other, Real):
            return self.is_degenerate and self.is_contiguous and float(self) == other
        else:
            raise TypeError(other)

    def issuperset(self, other: Union['MultiInterval', Real]) -> bool:
        # i.e. self contains all elements of other
        return other in self

    # SET: ITEMS (INPLACE)

    def add(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
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
        self._consistency_check()

        out = MultiInterval()
        self.endpoints, out.endpoints = self.endpoints[:-2], self.endpoints[-2:]

        out._consistency_check()
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

        if self.is_empty and len(other) == 1:
            self.endpoints = other[0].endpoints.copy()
            self._consistency_check()
            return self

        for _other in other:
            if isinstance(_other, MultiInterval):
                _other._consistency_check()
                self.endpoints.extend(_other.endpoints)
            elif isinstance(_other, Real):
                self.endpoints.append((_other, 0))
                self.endpoints.append((_other, 0))
            else:
                raise TypeError(_other)

        self.merge_adjacent(sort=True)
        self._consistency_check()
        return self

    def intersection_update(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        self._consistency_check()
        self.endpoints = MultiInterval.merge(self, *other, n_overlaps=len(other) + 1).endpoints
        self._consistency_check()
        return self

    def difference_update(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        """
        O(m + n) implementation, where m = len(other) and n = len(self)
        """
        self._consistency_check()

        if len(other) == 0:
            raise ValueError

        # get all the other endpoints merged together
        _other_endpoints = MultiInterval().update(*other).endpoints

        # nothing to remove
        if self.is_empty or len(_other_endpoints) == 0:
            return self

        # clear own endpoints while reading, since we'll replace everything
        _endpoints, self.endpoints = self.endpoints, []

        # iterate through other but not necessarily until the end
        other_idx = 0
        other_start, other_end = _other_endpoints[other_idx], _other_endpoints[other_idx + 1]

        # iterate through self exactly once
        for self_idx in range(0, len(_endpoints), 2):
            self_start, self_end = _endpoints[self_idx], _endpoints[self_idx + 1]

            while True:
                if self_start < other_start:
                    self.endpoints.append(self_start)
                    self.endpoints.append(min(self_end, (other_start[0], other_start[1] - 1)))

                if other_end < self_end:
                    self_start = max(self_start, (other_end[0], other_end[1] + 1))

                    if other_idx + 2 < len(_other_endpoints):
                        other_idx += 2
                        other_start, other_end = _other_endpoints[other_idx], _other_endpoints[other_idx + 1]
                        continue

                break

            # end of other, quick exit
            if other_end < self_start:
                assert self_start <= self_end
                assert other_idx + 2 >= len(_other_endpoints)
                self.endpoints.append(self_start)
                self.endpoints.extend(_endpoints[self_idx + 1:])
                break

        # allow operator chaining
        self._consistency_check()
        return self

    def symmetric_difference_update(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        self._consistency_check()
        self.endpoints = MultiInterval.merge(self, *other, n_overlaps=range(1, len(other) + 1, 2)).endpoints
        self._consistency_check()
        return self

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().update(*other)

    def intersection(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().intersection_update(*other)

    def difference(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().difference_update(*other)

    def symmetric_difference(self, *other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.copy().symmetric_difference_update(*other)

    # INTERVAL OPERATIONS (INPLACE)

    def merge_adjacent(self, distance: Real = 0, sort: bool = False) -> 'MultiInterval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        if len(self.endpoints) == 0:
            return self
        elif len(self.endpoints) == 2:
            assert self.endpoints[0] <= self.endpoints[1]  # consistency check
            return self

        # sort contiguous intervals by start, end
        if sort:
            _endpoints, self.endpoints = self.endpoints, []
            for _start, _end in sorted((_endpoints[idx], _endpoints[idx + 1]) for idx in range(0, len(_endpoints), 2)):
                self.endpoints.append(_start)
                self.endpoints.append(_end)

        # merge adjacent intervals
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
        # self._consistency_check()
        return self

    def abs(self) -> 'MultiInterval':
        self.endpoints = MultiInterval.merge(self[0], self.positive, self.negative.mirror()).endpoints
        return self

    def invert(self) -> 'MultiInterval':
        other = MultiInterval(start=-math.inf,
                              end=math.inf,
                              start_closed=not INFINITY_IS_NOT_FINITE,
                              end_closed=not INFINITY_IS_NOT_FINITE)
        self.endpoints, other.endpoints = other.endpoints, self.endpoints
        self.difference_update(other)
        return self

    def mirror(self) -> 'MultiInterval':
        _endpoints, self.endpoints = self.endpoints[::-1], []
        for point, epsilon in _endpoints:
            self.endpoints.append((-point, -epsilon))
        return self

    def expand(self, distance: Real, inplace=False) -> 'MultiInterval':
        if not inplace:
            return self.copy().expand(distance, inplace=True)

        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)  # todo: allow negative numbers

        for idx in range(0, len(self.endpoints), 2):
            self.endpoints[idx] = (self.endpoints[idx][0] - distance, self.endpoints[idx][1])
            self.endpoints[idx + 1] = (self.endpoints[idx + 1][0] + distance, self.endpoints[idx + 1][1])

        self.merge_adjacent()
        return self

    # INTERVAL OPERATIONS

    def __contains__(self, other: Union['MultiInterval', Real]) -> bool:
        self._consistency_check()

        if isinstance(other, MultiInterval):
            if self.is_empty:
                return False
            self_idx = 0
            for other_idx in range(0, len(other.endpoints), 2):
                while self_idx + 2 < len(self.endpoints) and self.endpoints[self_idx + 1] < other.endpoints[other_idx]:
                    self_idx += 2
                if self.endpoints[self_idx + 1] < other.endpoints[other_idx]:
                    assert self_idx + 2 >= len(self.endpoints)
                    return False
                if self.endpoints[self_idx] > other.endpoints[other_idx]:
                    return False
                if self.endpoints[self_idx + 1] < other.endpoints[other_idx + 1]:
                    return False

            return True

        elif isinstance(other, Real):
            idx = bisect.bisect_left(self.endpoints, (other, 0))
            if idx % 2 == 1:
                return True  # between existing endpoints
            elif idx < len(self.endpoints) and self.endpoints[idx] == (other, 0):
                return True  # touching start point
            return False

        else:
            raise TypeError(other)

    def overlapping(self, other: Union['MultiInterval', Real], or_adjacent: bool = False) -> 'MultiInterval':
        self._consistency_check()

        if isinstance(other, Real):
            if self.is_empty:
                return MultiInterval()

            # deal with infinity if it's not allowed in a closed interval
            elif math.isinf(other) and INFINITY_IS_NOT_FINITE:
                if or_adjacent and self.infimum == other:
                    assert other == -math.inf
                    return MultiInterval(start=-math.inf,
                                         end=self.endpoints[1][0],
                                         start_closed=False,
                                         end_closed=self.endpoints[1][1] == 0)
                if or_adjacent and self.supremum == other:
                    assert other == math.inf
                    return MultiInterval(start=self.endpoints[-2][0],
                                         end=math.inf,
                                         start_closed=self.endpoints[-2][1] == 0,
                                         end_closed=False)
                return MultiInterval()

            # convert to interval
            else:
                other = MultiInterval(other)

        if not isinstance(other, MultiInterval):
            raise TypeError(other)

        if self.is_empty or other.is_empty:
            return MultiInterval()

        # whether to consider adjacent points (one epsilon away)
        if or_adjacent:
            adj_epsilon = 1
        else:
            adj_epsilon = 0

        # iterate exactly once through other
        out = MultiInterval()
        other_idx = 0
        other._consistency_check()
        other_start = other.endpoints[other_idx]
        other_end = other.endpoints[other_idx + 1]

        # iterate exactly once through self
        for idx in range(0, len(self.endpoints), 2):
            start, start_epsilon = self.endpoints[idx]
            end, end_epsilon = self.endpoints[idx + 1]

            # look for the next possibly overlapping segment in other
            while other_idx + 2 < len(other.endpoints) and other_end < (start, start_epsilon - adj_epsilon):
                other_idx += 2
                other_start = other.endpoints[other_idx]
                other_end = other.endpoints[other_idx + 1]

            # no more overlapping segments, exit
            if (start, start_epsilon - adj_epsilon) > other_end:
                assert other_idx + 2 >= len(other.endpoints)
                break

            # check if other segment overlaps current segment
            elif other_start <= (end, end_epsilon + adj_epsilon):
                out.endpoints.append((start, start_epsilon))
                out.endpoints.append((end, end_epsilon))

        out._consistency_check()
        return out

    def overlaps(self, other: Union['MultiInterval', Real], or_adjacent: bool = False) -> bool:
        # too complex for it to be worth making this more efficient
        return not self.overlapping(other, or_adjacent=or_adjacent).is_empty

    # INTERVAL ARITHMETIC: GENERIC

    def apply_monotonic_unary_function(self, func: Callable, inplace: bool = False) -> 'MultiInterval':
        # by default, do this to a copy
        if not inplace:
            return self.copy().apply_monotonic_unary_function(func, inplace=True)

        self._consistency_check()
        _endpoints, self.endpoints = self.endpoints, []

        for idx in range(0, len(_endpoints), 2):
            start = func(_endpoints[idx][0])
            end = func(_endpoints[idx + 1][0])
            start_epsilon = _endpoints[idx][1]
            end_epsilon = _endpoints[idx + 1][1]
            self.endpoints.append(min((start, start_epsilon), (end, -end_epsilon)))
            self.endpoints.append(max((start, -start_epsilon), (end, end_epsilon)))

        self.merge_adjacent(sort=True)
        return self

    def apply_monotonic_binary_function(self,
                                        func: Callable,
                                        other: Union['MultiInterval', Real],
                                        right_hand_side: bool = False,
                                        inplace: bool = False
                                        ) -> 'MultiInterval':
        # by default, do this to a copy
        if not inplace:
            return self.copy().apply_monotonic_binary_function(func,
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

        # only allow empty first arg, not second
        if len(_second) == 0:
            raise ValueError(f'cannot run {func} with null set')
        elif len(_first) == 0:
            return MultiInterval()

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
        return self.apply_monotonic_binary_function(operator.add, other)

    def __radd__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.add, other, right_hand_side=True)

    def __sub__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.sub, other)

    def __rsub__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.sub, other, right_hand_side=True)

    def __mul__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.mul, other)

    def __rmul__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.mul, other, right_hand_side=True)

    def __truediv__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        # todo: deal with inf and zero
        if isinstance(other, MultiInterval):
            if other.is_empty:
                raise ValueError('cannot divide by nothing')
            return self * other.reciprocal()

        elif isinstance(other, Real):
            if float(other) == 0:
                return MultiInterval(start=-math.inf,
                                     end=math.inf,
                                     start_closed=not INFINITY_IS_NOT_FINITE,
                                     end_closed=not INFINITY_IS_NOT_FINITE)
            else:
                return self.apply_monotonic_binary_function(operator.truediv, other)

        else:
            raise TypeError(other)

    def __rtruediv__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        # todo: deal with inf and zero
        if self.is_empty:
            raise ValueError('cannot divide by nothing')
        return other * self.reciprocal()

    # noinspection DuplicatedCode
    def __floordiv__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        out = self.__truediv__(other)
        if not out.is_finite:
            raise OverflowError('cannot convert float infinity to integer')

        _endpoints, out.endpoints = out.endpoints, []
        for point, _ in _endpoints:
            out.endpoints.append((math.floor(point), 0))
        out.merge_adjacent()
        return out

    # noinspection DuplicatedCode
    def __rfloordiv__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        out = self.__rtruediv__(other)
        if not out.is_finite:
            raise OverflowError('cannot convert float infinity to integer')

        _endpoints, out.endpoints = out.endpoints, []
        for point, _ in _endpoints:
            out.endpoints.append((math.floor(point), 0))
        out.merge_adjacent()
        return out

    def __mod__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        # other MUST be finite and cannot include zero
        # process the positive and negative parts separately?

        if isinstance(other, MultiInterval):
            if 0 in other:
                raise ZeroDivisionError(other)  # todo: decide if this is an error or not (also see below)
            elif not self.is_finite:
                raise ValueError('cannot mod infinity')
            elif self.is_empty or other.is_empty:
                return MultiInterval()
            elif not other.is_non_negative:
                return self.__mod__(other.negative.mirror()).mirror().union(self.__mod__(other.positive))

            # current state: other.is_positive & self.is_finite & not(self.is_empty) & not(other.is_empty)
            raise NotImplementedError  # todo

        elif isinstance(other, Real):
            if float(other) == 0:
                raise ZeroDivisionError(other)  # todo: decide if this is an error or not (see above)
            elif not self.is_finite:
                raise ValueError('cannot mod infinity')
            elif self.is_empty:
                return MultiInterval()
            elif other < 0:
                return self.__mod__(-other).mirror()

            # current state: (other > 0) & self.is_finite & not(self.is_empty)
            out = MultiInterval()
            for idx in range(0, len(self.endpoints), 2):
                start = self.endpoints[idx][0]
                end = self.endpoints[idx + 1][0]
                start_epsilon = self.endpoints[idx][1]
                end_epsilon = self.endpoints[idx + 1][1]
                delta_floor = (end // other) - (start // other)
                assert delta_floor >= 0
                if delta_floor == 0:
                    out.endpoints.append((start % other, start_epsilon))
                    out.endpoints.append((end % other, end_epsilon))
                elif delta_floor == 1:
                    out.endpoints.append((0, 0))
                    out.endpoints.append((end % other, end_epsilon))
                    out.endpoints.append((start % other, start_epsilon))
                    out.endpoints.append((other, -1))
                else:
                    return MultiInterval(start=0,
                                         end=other,
                                         end_closed=False)

            out.merge_adjacent(sort=True)
            return out

        else:
            raise TypeError(other)

    def __rmod__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def __divmod__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def __rdivmod__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    def __pow__(self,
                power: Union['MultiInterval', Real],
                modulo: Optional[Union['MultiInterval', Real]] = None
                ) -> 'MultiInterval':
        """
        edge cases all the way down

        | POW >= 0   | 0           | 0 to 1         | 1              | 1 to inf        | inf  |
        |------------|-------------|----------------|----------------|-----------------|------|
        | inf        | monotonic 1 | monotonic inf  | monotonic == i | monotonic inf   | inf  |
        | 1 to inf   | monotonic 1 | monotonic -> 0 | monotonic ==   | monotonic -> i  | inf  |
        | 1          | monotonic 1 | monotonic 1    | monotonic == 1 | monotonic 1     | 1    |
        | 0 to 1     | monotonic 1 | monotonic -> 1 | monotonic ==   | monotonic -> 0  | 0    |
        | 0          | monotonic 1 | monotonic 0    | monotonic == 0 | monotonic 0     | 0    |
        | -1 to 0    | 1           | COMPLEX!       | no change      | ONLY FINITE INT | 0    |
        | -1         | 1           | COMPLEX!       | no change      | ONLY FINITE INT | -1   |
        | -inf to -1 | 1           | COMPLEX!       | no change      | ONLY FINITE INT | -inf |
        | -inf       | 1           | inf            | no change -inf | inf, neg if odd | inf  |

        | POW <= 0   | 0           | 0 to -1        | -1             | -1 to -inf      | -inf |
        |------------|-------------|----------------|----------------|-----------------|------|
        | inf        | monotonic 1 | monotonic 0    | monotonic 0    | monotonic 0     | 0    |
        | 1 to inf   | monotonic 1 | monotonic      | monotonic      | monotonic       | 0    |
        | 1          | monotonic 1 | monotonic 1    | monotonic 1    | monotonic 1     | 1    |
        | 0 to 1     | monotonic 1 | monotonic      | monotonic      | monotonic       | inf  |
        | 0          | monotonic 1 | ±inf           | ±inf           | ±inf            | ±inf |
        | -1 to 0    | 1           | COMPLEX!       | reciprocal     | ONLY FINITE INT | -inf |
        | -1         | 1           | COMPLEX!       | reciprocal     | ONLY FINITE INT | -1   |
        | -inf to -1 | 1           | COMPLEX!       | reciprocal     | ONLY FINITE INT | 0    |
        | -inf       | 1           | 0              | reciprocal 0   | 0               | 0    |
        """
        if modulo is not None:
            if isinstance(power, Real):
                if math.isinf(power):
                    raise TypeError('pow() 3rd argument not allowed unless all arguments are integral')
                power = MultiInterval(power)
            if not isinstance(power, MultiInterval):
                raise TypeError(power)

            if isinstance(modulo, Real):
                if math.isinf(modulo):
                    raise TypeError('pow() 3rd argument not allowed unless all arguments are integral')
                modulo = MultiInterval(modulo)
            if not isinstance(modulo, MultiInterval):
                raise TypeError(modulo)

            # modular exponentiation only valid on integers
            if not self.is_integral or not power.is_integral or not modulo.is_integral:
                raise TypeError('pow() 3rd argument not allowed unless all arguments are integral')

            # modular exponentiation only valid for non-negative integer powers
            if not power.is_non_negative:
                raise ValueError('pow() 2nd argument cannot be negative when 3rd argument specified')

            # with a bespoke implementation of pow()
            # it would be possible to group by (base, mod) then use the highest exp
            out = MultiInterval()
            for base in self.degenerate_points:
                for exp in power.degenerate_points:
                    for mod in modulo.degenerate_points:
                        out.update(pow(int(float(base)), int(float(exp)), int(float(mod))))
            return out

        if isinstance(power, Real):
            if math.isinf(power):
                raise NotImplementedError  # todo: many special cases
            else:
                power = MultiInterval(power)

        if not isinstance(power, MultiInterval):
            raise TypeError(power)

        elif self.is_empty or power.is_empty:
            return MultiInterval()

        elif self.is_positive:
            return self.apply_monotonic_binary_function(operator.pow, power)

        elif self == 0:
            # contains -inf to inf
            if not power.is_non_negative:
                return MultiInterval(start=-math.inf,
                                     end=math.inf,
                                     start_closed=not INFINITY_IS_NOT_FINITE,
                                     end_closed=not INFINITY_IS_NOT_FINITE)

            # contains 1
            elif 0 in power:
                if not power.is_non_positive:
                    return MultiInterval.merge(0, 1)
                else:
                    return MultiInterval(1)

            # only 0
            else:
                assert power.is_positive
                return MultiInterval(0)

        elif self.is_negative:
            raise NotImplementedError  # todo: many special cases

        else:
            return MultiInterval.merge(self.positive ** power, self[0] ** power, self.negative ** power)

    def __rpow__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        raise NotImplementedError

    # INTERVAL ARITHMETIC: INTEGERS ONLY

    def __lshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.lshift, other)

    def __rlshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.lshift, other, right_hand_side=True)

    def __rshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.rshift, other)

    def __rrshift__(self, other: Union['MultiInterval', Real]) -> 'MultiInterval':
        return self.apply_monotonic_binary_function(operator.rshift, other, right_hand_side=True)

    # INTERVAL ARITHMETIC: UNARY

    def reciprocal(self) -> 'MultiInterval':
        if 0 in self:
            return MultiInterval(start=-math.inf,
                                 end=math.inf,
                                 start_closed=not INFINITY_IS_NOT_FINITE,
                                 end_closed=not INFINITY_IS_NOT_FINITE)

        elif self.is_empty:
            return MultiInterval()

        out = MultiInterval()
        for idx in range(0, len(self.endpoints), 2):
            start, start_epsilon = self.endpoints[idx]
            end, end_epsilon = self.endpoints[idx + 1]

            if start == 0:
                assert start_epsilon > 0
                out.endpoints.append((1 / end, -end_epsilon))
                out.endpoints.append((math.inf, -1))

            elif end == 0:
                assert end_epsilon < 0
                out.endpoints.append((-math.inf, 1))
                out.endpoints.append((1 / start, -start_epsilon))

            else:
                out.endpoints.append((1 / end, -end_epsilon))
                out.endpoints.append((1 / start, -start_epsilon))
        out.merge_adjacent(sort=True)
        return out

    def __neg__(self):
        return self.copy().mirror()

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        return self.copy().abs()

    def __invert__(self):
        return self.copy().invert()

    def exp(self):
        return self.apply_monotonic_unary_function(math.exp)

    def log(self, base: float = math.e):
        return self.apply_monotonic_unary_function(lambda num: math.log(num, base=base))

    def __round__(self, n_digits: int = 0):  # towards nearest integer
        return self.apply_monotonic_unary_function(lambda num: round(num, ndigits=n_digits))

    def __trunc__(self):  # towards 0.0
        return self.apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.trunc(num))

    def __floor__(self):  # towards -inf
        return self.apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.floor(num))

    def __ceil__(self):  # towards +inf
        return self.apply_monotonic_unary_function(lambda num: num if math.isinf(num) else math.ceil(num))

    # TYPE CASTING

    def __bool__(self) -> bool:
        return not self.is_empty

    def __complex__(self) -> complex:
        if self.is_degenerate and self.is_contiguous:
            return complex(float(self))
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to int')

    def __float__(self) -> float:
        if self.is_degenerate and self.is_contiguous:
            return float(self.infimum)
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to float')

    def __int__(self) -> int:
        if self.is_degenerate and self.is_contiguous:
            return int(float(self))
        else:
            raise ValueError('cannot cast non-degenerate MultiInterval to complex')

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        def _interval_to_str(start_tuple, end_tuple, fancy_inf=False):
            assert start_tuple <= end_tuple, (start_tuple, end_tuple)

            # unpack
            start, start_epsilon = start_tuple
            end, end_epsilon = end_tuple
            assert start_epsilon in {1, 0}, (start_tuple, end_tuple)
            assert end_epsilon in {0, -1}, (start_tuple, end_tuple)

            # degenerate interval
            if start == end:
                assert start_epsilon == 0
                assert end_epsilon == 0
                return f'[{start}]'

            # print the mathematically standard but logically inconsistent way
            if fancy_inf:
                # left side
                if start == -math.inf:
                    assert start_epsilon == 0
                    start = '-∞'
                    _left_bracket = '('
                else:
                    _left_bracket = '(' if start_epsilon else '['

                # right side
                if end == math.inf:
                    assert end_epsilon == 0
                    end = '∞'
                    _right_bracket = ')'
                else:
                    _right_bracket = ')' if end_epsilon else ']'

                return f'{_left_bracket}{start}, {end}{_right_bracket}'

            # for the sake of consistency, a closed endpoint at ±inf will be square instead of round
            else:
                return f'{"(" if start_epsilon else "["}{start}, {end}{")" if end_epsilon else "]"}'

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

            return f'{{ {" , ".join(str_intervals)} }}'


def random_multi_interval(start, end, n, decimals=2, neg_inf=0.25, pos_inf=0.25):
    _points = set()
    if random.random() < neg_inf and n > 1:
        _points.add(-math.inf)
    if random.random() < pos_inf and n > 1:
        _points.add(math.inf)
    while len(_points) < 2 * n:
        if decimals:
            _points.add(round(start + (end - start) * random.random(), decimals))
        else:
            _points.add(int(start + (end - start) * random.random()))

    _endpoints = sorted(_points)

    out = MultiInterval()
    for idx in range(0, len(_endpoints), 2):
        x = random.random()

        # degenerate interval
        if (not (math.isinf(_endpoints[idx]) and INFINITY_IS_NOT_FINITE) and
                (x < 0.2 or _endpoints[idx] == _endpoints[idx + 1])):
            out.endpoints.append((_endpoints[idx], 0))
            out.endpoints.append((_endpoints[idx], 0))

        # closed interval
        elif x < 0.4 \
                and not (math.isinf(_endpoints[idx]) and INFINITY_IS_NOT_FINITE) \
                and not (math.isinf(_endpoints[idx + 1]) and INFINITY_IS_NOT_FINITE):
            out.endpoints.append((_endpoints[idx], 0))
            out.endpoints.append((_endpoints[idx + 1], 0))

        # closed-open interval
        elif x < 0.6 and not (math.isinf(_endpoints[idx]) and INFINITY_IS_NOT_FINITE):
            out.endpoints.append((_endpoints[idx], 0))
            out.endpoints.append((_endpoints[idx + 1], -1))

        # open-closed interval
        elif x < 0.8 and not (math.isinf(_endpoints[idx + 1]) and INFINITY_IS_NOT_FINITE):
            out.endpoints.append((_endpoints[idx], 1))
            out.endpoints.append((_endpoints[idx + 1], 0))

        # open interval
        else:
            out.endpoints.append((_endpoints[idx], 1))
            out.endpoints.append((_endpoints[idx + 1], -1))

    # noinspection PyProtectedMember
    out._consistency_check()
    return out


if __name__ == '__main__':
    t = time.time()
    for _ in range(100):
        print()
        i = random_multi_interval(-100, 100, random.randint(0, 5), 0)
        j = random_multi_interval(-100, 100, random.randint(0, 5), 0)
        print(i, i.closed_hull, i.reciprocal())
        print(j, j.closed_hull, j.reciprocal())
        print('union:                ', i.union(j), i.union(j) in j, j in i.union(j))
        print('intersection:         ', i.intersection(j), i.intersection(j) in i, i.intersection(j) in j)
        print('difference:           ', i.difference(j), i.difference(j) in i, i.difference(j) in j)
        print('symmetric_difference: ', i.symmetric_difference(j))
        print('overlapping:          ', i.overlapping(j))
        print('abs:                  ', abs(i))
        print('i[-50:50]', i[-50:50])
        print('j[4:]', j[4:])
        print('i[:0]', i[:0])
        print('j[:]', j[:])
    print(time.time() - t)
