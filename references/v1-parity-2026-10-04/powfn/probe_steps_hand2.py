import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math
from fractions import Fraction as F
from common import *
a1, a2 = round(V1(F(1, 3)), 400), quiet(lambda: round(V2(F(1, 3)), 400))
print('round 1/3 400: equal sets', pieces1(a1) == pieces2(a2), 'value == round(F(1,3),400):', pieces2(a2)[0][0] == round(F(1, 3), 400))
b1, b2 = math.floor(V1(1e300)), quiet(lambda: math.floor(V2(1e300)))
print('floor 1e300: v1', type(pieces1(b1)[0][0]).__name__, 'v2', pieces2(b2), type(pieces2(b2)[0][0]).__name__, 'math.floor(1e300) in v2:', math.floor(1e300) in b2, '1e300 in v2:', 1e300 in b2)
c2 = quiet(lambda: math.floor(V2(1e300, 1e300 * 1.0000001)))
print('floor [1e300, 1.0000001e300]:', c2)
print('numpy floor 1000-value case n pieces:', len(quiet(lambda: math.floor(V2(F(1, 2), F(1999, 2)))).pieces))
