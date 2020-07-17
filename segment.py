import bisect
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union


@dataclass(order=True, unsafe_hash=True, frozen=True)
class Segment:
    """
    a line in 2d space, starting and ending at integers (inclusive of both endpoints)
    displacement from start to end must be non-negative, but can be zero
    """
    start: int
    end: int

    def __post_init__(self):
        if not isinstance(self.start, int):
            if not isinstance(self.end, int):
                raise TypeError((self.start, self.end))
            raise TypeError(self.start)
        if not isinstance(self.end, int):
            raise TypeError(self.end)

        if self.start > self.end:
            raise ValueError((self.start, self.end))

    @property
    def length(self) -> int:
        return self.end - self.start

    def __contains__(self, other: Union[int, 'Segment']) -> bool:
        if isinstance(other, int):
            return self.start <= other <= self.end

        elif isinstance(other, Segment):
            return self.start <= other.start and other.end <= self.end

        else:
            raise TypeError(other)

    def overlaps(self, other: Union[int, 'Segment']) -> bool:
        if isinstance(other, int):
            return self.start <= other <= self.end

        elif isinstance(other, Segment):
            return self.start <= other.end and other.start <= self.end

        else:
            raise TypeError(other)

    def adjacent_to(self, other: Union[int, 'Segment'], merge_adjacent_distance: int = 1) -> bool:
        if isinstance(other, int):
            return self.start - merge_adjacent_distance <= other <= self.end + merge_adjacent_distance

        elif isinstance(other, Segment):
            return self.start - merge_adjacent_distance <= other.end \
                   and other.start <= self.end + merge_adjacent_distance

        else:
            raise TypeError(other)

    def intersect(self, other: Union[int, 'Segment']) -> Optional['Segment']:
        if isinstance(other, int):
            if other in self:
                return Segment(other, other)

        elif isinstance(other, Segment):
            if self.overlaps(other):
                return Segment(max(self.start, other.start), min(self.end, other.end))

        else:
            raise TypeError(other)

    def join(self, other: 'Segment', merge_adjacent_distance: int = 1) -> Optional['Segment']:
        if isinstance(other, Segment):
            # you can only join segments if they are are adjacent or overlapping
            assert self.adjacent_to(other, merge_adjacent_distance=merge_adjacent_distance)
            return Segment(min(self.start, other.start), max(self.end, other.end))

        else:
            raise TypeError(other)

    def __repr__(self):
        return f'Segment({repr(self.start)}, {repr(self.end)})'


@dataclass
class MultiSegment:
    """
    a bunch of non-overlapping lines in 2d space, starting and ending at integers

    WARNING: never been tested, logic hasn't even been double-checked
    """

    _segments: List[int] = field(default_factory=list)

    @property
    def segments(self) -> List[Segment]:
        self._consistency_check()
        return [Segment(self._segments[i], self._segments[i + 1]) for i in range(0, len(self._segments), 2)]

    @property
    def n_segments(self) -> int:
        return len(self._segments) // 2

    @property
    def total_length(self) -> int:
        return sum(self._segments[i + 1] - self._segments[i] for i in range(0, len(self._segments), 2))

    @property
    def min(self):
        if len(self._segments) > 0:
            return self._segments[0]

    @property
    def max(self):
        if len(self._segments) > 0:
            return self._segments[-1]

    def __repr__(self):
        return f'MultiSegment[{repr(self.segments)}'

    def __contains__(self, other: Union[int, Segment, 'MultiSegment']) -> bool:
        if isinstance(other, int):
            return bisect.bisect_left(self._segments, other) % 2 == 1

        elif isinstance(other, Segment):

            left_bound = bisect.bisect_right(self._segments, other.start)
            right_bound = bisect.bisect_left(self._segments, other.end, lo=left_bound)
            assert 0 <= left_bound <= right_bound, other

            # within an existing segment
            return right_bound == left_bound and left_bound % 2 == 1

        elif isinstance(other, MultiSegment):
            self_idx = 0
            for other_idx in range(0, len(other._segments), 2):
                while self._segments[self_idx + 1] < other._segments[other_idx]:
                    self_idx += 2
                if self._segments[self_idx] > other._segments[other_idx]:
                    return False
                if self._segments[self_idx + 1] < other._segments[other_idx + 1]:
                    return False

            return True

        else:
            raise TypeError(other)

    def _consistency_check(self) -> None:
        # length must be even
        assert len(self._segments) % 2 == 0, len(self._segments)

        # must be sorted
        if len(self._segments) > 0:
            prev = self._segments[0]
            for elem in self._segments[1:]:
                assert isinstance(elem, int)
                assert elem >= prev, (elem, self._segments)
                prev = elem

    def add(self, segment: Segment, merge_adjacent_distance: int = 1) -> 'MultiSegment':
        self._consistency_check()

        # self is empty, replace
        if len(self._segments) == 0:
            self._segments = [segment.start, segment.end]
            return self

        assert isinstance(merge_adjacent_distance, int)
        assert merge_adjacent_distance >= 0

        left_bound = bisect.bisect_left(self._segments, segment.start - merge_adjacent_distance)
        right_bound = bisect.bisect_right(self._segments, segment.end + merge_adjacent_distance, lo=left_bound)
        assert 0 <= left_bound <= right_bound, segment

        # either within or between segments
        if right_bound == left_bound:
            # insert between segments
            if left_bound % 2 == 0:
                self._segments = self._segments[:left_bound] + \
                                 [segment.start, segment.end] + \
                                 self._segments[right_bound:]

            # contained within existing segment, do nothing
            else:
                pass

        # in-place replacement, one side
        elif right_bound - left_bound == 1:
            if left_bound % 2 == 0:
                self._segments[left_bound] = min(segment.start, self._segments[left_bound])
            else:
                self._segments[right_bound - 1] = max(segment.end, self._segments[right_bound - 1])

        # in-place replacement, both sides
        elif right_bound - left_bound == 2 and left_bound % 2 == 0:
            assert right_bound % 2 == 0, segment
            self._segments[left_bound] = min(segment.start, self._segments[left_bound])
            self._segments[right_bound - 1] = max(segment.end, self._segments[right_bound - 1])

        # merge multiple things
        else:
            assert right_bound - left_bound >= 2
            if left_bound % 2 == 0:
                self._segments[left_bound] = min(segment.start, self._segments[left_bound])
                left_bound += 1
            if right_bound % 2 == 0:
                self._segments[right_bound - 1] = max(segment.end, self._segments[right_bound - 1])
                right_bound -= 1
            self._segments = self._segments[:left_bound] + self._segments[right_bound:]

        # allow operator chaining
        self._consistency_check()
        return self

    def _intersect(self, segment: Segment) -> 'MultiSegment':
        self._consistency_check()

        # self is empty
        if len(self._segments) == 0:
            return self

        left_bound = bisect.bisect_left(self._segments, segment.start)
        right_bound = bisect.bisect_right(self._segments, segment.end, lo=left_bound)
        assert 0 <= left_bound <= right_bound, segment

        # either within or between segments
        if right_bound == left_bound:
            # within an existing segment
            if left_bound % 2 == 1:
                self._segments = [segment.start, segment.end]

            # between existing segments, erase
            else:
                self._segments = []

        # in-place replacement, one side
        elif right_bound - left_bound == 1:
            if left_bound % 2 == 0:
                self._segments = [self._segments[left_bound], segment.end]
            else:
                self._segments = [segment.start, self._segments[right_bound - 1]]

        # in-place replacement, both sides
        elif right_bound - left_bound == 2 and left_bound % 2 == 0:
            assert right_bound % 2 == 0, segment
            self._segments = [self._segments[left_bound], self._segments[right_bound - 1]]

        # merge multiple things
        else:
            assert right_bound - left_bound >= 2
            if left_bound % 2 == 1:
                self._segments[left_bound - 1] = segment.start
                left_bound -= 1
            if right_bound % 2 == 1:
                self._segments[right_bound] = segment.end
                right_bound += 1
            self._segments = self._segments[left_bound:right_bound]

        # allow operator chaining
        self._consistency_check()
        return self

    def remove(self, segment: Segment) -> 'MultiSegment':
        self._consistency_check()

        # self is empty or nothing to remove
        if len(self._segments) == 0 or segment.end - segment.start == 0:
            return self

        left_bound = bisect.bisect_left(self._segments, segment.start)
        right_bound = bisect.bisect_right(self._segments, segment.end, lo=left_bound)
        assert 0 <= left_bound <= right_bound, segment

        # either within or between segments
        if right_bound == left_bound:
            # split an existing segment
            if left_bound % 2 == 1:
                self._segments = self._segments[:left_bound] + \
                                 [segment.start, segment.end] + \
                                 self._segments[right_bound:]

            # between existing segments, do nothing
            else:
                pass

        # in-place replacement, one side
        elif right_bound - left_bound == 1:
            if left_bound % 2 == 0:
                self._segments[right_bound - 1] = segment.end
            else:
                self._segments[left_bound] = segment.start

        # in-place replacement, both sides
        elif right_bound - left_bound == 2 and left_bound % 2 == 1:
            assert right_bound % 2 == 1, segment
            self._segments[left_bound] = segment.start
            self._segments[right_bound - 1] = segment.end

        # merge multiple things
        else:
            assert right_bound - left_bound >= 2
            if left_bound % 2 == 1:
                self._segments[left_bound] = segment.start
                left_bound += 1
            if right_bound % 2 == 1:
                self._segments[right_bound - 1] = segment.end
                right_bound -= 1
            self._segments = self._segments[:left_bound] + self._segments[right_bound:]

        # allow operator chaining
        self._consistency_check()
        return self

    def xor(self, segment: Segment) -> 'MultiSegment':
        self._consistency_check()

        # self is empty
        if len(self._segments) == 0:
            self._segments = [segment.start, segment.end]
            return self

        # left_bound = bisect.bisect_left(self._segments, segment.start)
        # right_bound = bisect.bisect_right(self._segments, segment.end, lo=left_bound)
        # assert 0 <= left_bound <= right_bound, segment
        #
        # # either within or between segments
        # if right_bound == left_bound:
        #     self._segments = self._segments[:left_bound] + \
        #                      [segment.start, segment.end] + \
        #                      self._segments[right_bound:]
        #
        # # in-place replacement, one side
        # elif right_bound - left_bound == 1:
        #     if left_bound % 2 == 0:
        #         self._segments[right_bound - 1] = segment.end
        #     else:
        #         self._segments[left_bound] = segment.start
        #
        # # in-place replacement, both sides
        # elif right_bound - left_bound == 2 and left_bound % 2 == 1:
        #     assert right_bound % 2 == 1, segment
        #     self._segments[left_bound] = segment.start
        #     self._segments[right_bound - 1] = segment.end
        #
        # # merge multiple things
        # else:
        #     assert right_bound - left_bound >= 2
        #     if left_bound % 2 == 1:
        #         self._segments[left_bound] = segment.start
        #         left_bound += 1
        #     if right_bound % 2 == 1:
        #         self._segments[right_bound - 1] = segment.end
        #         right_bound -= 1
        #     self._segments = self._segments[:left_bound] + self._segments[right_bound:]
        #
        # # allow operator chaining
        # self._consistency_check()
        # return self
        # todo
        raise NotImplementedError

    def merge_adjacent(self, merge_adjacent_distance: int) -> 'MultiSegment':
        assert isinstance(merge_adjacent_distance, int)
        assert merge_adjacent_distance >= 0

        _segments = []
        idx = 0
        _len = len(self._segments)

        # blank
        if _len == 0:
            return self

        _end = self._segments[0]
        while idx < _len:
            _segments.append(self._segments[idx])
            _end = max(_end, self._segments[idx + 1])
            idx += 2

            while idx < _len and self._segments[idx] <= _end + merge_adjacent_distance:
                _end = max(_end, self._segments[idx + 1])
                idx += 2

            _segments.append(_end)

        self._segments = _segments

        # allow operator chaining
        self._consistency_check()
        return self

    def update_multi(self, others: Iterable['MultiSegment'], merge_adjacent_distance: int = 1) -> 'MultiSegment':
        self._consistency_check()

        _segments = self._segments.copy()

        for other in others:
            assert isinstance(other, MultiSegment)
            other._consistency_check()
            _segments.extend(other._segments)

        segment_tuples = sorted((_segments[i], _segments[i + 1]) for i in range(0, len(_segments), 2))
        self._segments = [elem for segment_tuple in segment_tuples for elem in segment_tuple]
        self.merge_adjacent(merge_adjacent_distance=merge_adjacent_distance)

        # allow operator chaining
        self._consistency_check()
        return self

    def update(self, other: Union['MultiSegment', Segment], merge_adjacent_distance: int = 1) -> 'MultiSegment':
        self._consistency_check()

        # have to process the other multi-segment
        if isinstance(other, MultiSegment):
            other._consistency_check()
            if len(other._segments) == 2:
                self.add(Segment(*other._segments), merge_adjacent_distance=merge_adjacent_distance)
            elif len(other._segments) > 2:
                _segments = self._segments + other._segments
                segment_tuples = sorted((_segments[i], _segments[i + 1]) for i in range(0, len(_segments), 2))
                self._segments = [elem for segment_tuple in segment_tuples for elem in segment_tuple]
                self.merge_adjacent(merge_adjacent_distance=merge_adjacent_distance)

        # it's actually a segment, just add it
        elif isinstance(other, Segment):
            self.add(other, merge_adjacent_distance=merge_adjacent_distance)

        # unexpected type
        else:
            raise TypeError(other)

        # allow operator chaining
        self._consistency_check()
        return self

    def intersection_update(self, other: Union['MultiSegment', Segment]):
        self._consistency_check()

        # have to process the other multi-segment
        if isinstance(other, MultiSegment):
            other._consistency_check()
            if len(self._segments) == 0:
                self._segments = []
            elif len(other._segments) == 2:
                self._intersect(Segment(*other._segments))
            elif len(other._segments) > 2:
                # todo
                raise NotImplementedError

        # it's actually a segment, just truncate self
        elif isinstance(other, Segment):
            self._intersect(other)

        # unexpected type
        else:
            raise TypeError(other)

        # allow operator chaining
        self._consistency_check()
        return self

    def difference_update(self, other: Union['MultiSegment', Segment]):
        self._consistency_check()

        # have to process the other multi-segment
        if isinstance(other, MultiSegment):
            other._consistency_check()
            if len(self._segments) == 0:
                pass
            elif len(other._segments) == 2:
                self.remove(Segment(*other._segments))
            elif len(other._segments) > 2:
                # todo
                raise NotImplementedError

        # it's actually a segment, just remove
        elif isinstance(other, Segment):
            self.remove(other)

        # unexpected type
        else:
            raise TypeError(other)

        # allow operator chaining
        self._consistency_check()
        return self

    def symmetric_difference_update(self, other: Union['MultiSegment', Segment]):
        self._consistency_check()

        # have to process the other multi-segment
        if isinstance(other, MultiSegment):
            other._consistency_check()
            if len(self._segments) == 0:
                pass
            elif len(other._segments) == 2:
                self.xor(Segment(*other._segments))
            elif len(other._segments) > 2:
                # todo
                raise NotImplementedError

        # it's actually a segment, just remove
        elif isinstance(other, Segment):
            self.xor(other)

        # unexpected type
        else:
            raise TypeError(other)

        # allow operator chaining
        self._consistency_check()
        return self

    def copy(self):
        self._consistency_check()
        return MultiSegment().update(self, merge_adjacent_distance=0)

    def union(self, other: Union['MultiSegment', Segment], merge_adjacent_distance: int = 1) -> 'MultiSegment':
        return self.copy().update(other, merge_adjacent_distance=merge_adjacent_distance)

    def intersection(self, other: Union['MultiSegment', Segment]) -> 'MultiSegment':
        return self.copy().intersection_update(other)

    def difference(self, other: Union['MultiSegment', Segment]) -> 'MultiSegment':
        return self.copy().difference_update(other)

    def symmetric_difference(self, other: Union['MultiSegment', Segment]) -> 'MultiSegment':
        return self.copy().symmetric_difference_update(other)


if __name__ == '__main__':
    a = MultiSegment()
    print(a.segments)
    a.add(Segment(2, 4), merge_adjacent_distance=1)
    print(a.segments)
    a.add(Segment(6, 8), merge_adjacent_distance=1)
    print(a.segments)
    a.add(Segment(5, 5), merge_adjacent_distance=1)
    print(a.segments)

    b = MultiSegment()
    print(b.segments)
    b.add(Segment(0, 1), merge_adjacent_distance=1)
    print(b.segments)
    b.add(Segment(1, 3), merge_adjacent_distance=1)
    print(b.segments)
    b.add(Segment(1, 9), merge_adjacent_distance=1)
    print(b.segments)
    b.add(Segment(2, 8), merge_adjacent_distance=1)
    print(b.segments)

    a.update(b)
    print(a.segments)
