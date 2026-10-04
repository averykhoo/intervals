"""v2's replacement for CONSISTENCY_CHECK: the __debug__ assert in MultiInterval._wrap (run with and without -O)"""
import sys
sys.path[:0] = ['C:/Users/user/PycharmProjects/intervals']
from intervals import MultiInterval
from intervals.cuts import Cut, Side
bad = (Cut(3, Side.BELOW), Cut(1, Side.ABOVE))
print('__debug__ =', __debug__)
try:
    m = MultiInterval._wrap(bad)
    print('_wrap(invalid) accepted (check skipped):', m.cuts)
except AssertionError as e:
    print('_wrap(invalid) -> AssertionError (check ran)')
try:
    MultiInterval.from_cuts(bad)
except ValueError as e:
    print('from_cuts(invalid) -> ValueError regardless of -O')
