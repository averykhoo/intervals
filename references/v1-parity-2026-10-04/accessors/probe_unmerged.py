"""v1 is_contiguous on a hand-built unmerged endpoint list { [1] , (1, 2] } vs v2 (always normalized)"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *
from intervals.cuts import below, above
a = v1.MultiInterval(); a.endpoints = [(1, 0), (1, 0), (1, 1), (2, 0)]
print('v1 is_contiguous', a.is_contiguous, 'is_degenerate', a.is_degenerate, 'degenerate_points', a.degenerate_points, 'contiguous_intervals', len(a.contiguous_intervals), 'cardinality', a.cardinality)
b = v2.MultiInterval.from_pieces([(1, 1), (1, 2, False, True)])
print('v2 from_pieces', b, 'is_contiguous', b.is_contiguous, 'is_degenerate', b.is_degenerate, 'degenerate_points', b.degenerate_points, 'pieces', len(b.pieces), 'size', b.size)
try:
    v2.MultiInterval.from_cuts([below(1), above(1), above(1), above(2)])
except ValueError as e:
    print('v2 from_cuts unmerged -> ValueError:', e)
assert b.is_contiguous is True and a.is_contiguous is True
