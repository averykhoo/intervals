"""v1 inplace= on apply_monotonic_*; v2 has no mutation (immutable)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from common import *
m = V1(0, 1); r = m.apply_monotonic_unary_function(lambda x: 2 * x, inplace=True)
print('v1 unary inplace returns self:', r is m, 'm now', str(m))
m = V1(0, 1); r = m.apply_monotonic_binary_function(lambda x, y: x + y, 5, inplace=True)
print('v1 binary inplace returns self:', r is m, 'm now', str(m))
m = V1(0, 1); r = m.apply_monotonic_unary_function(lambda x: 2 * x)
print('v1 default copies:', r is not m, 'm still', str(m))
a = V2(0, 1)
try:
    a._cuts = ()
except AttributeError as e:
    print('v2 immutable:', e)
x = a; x = x * 2
print('v2 rebinding spelling x = x * 2:', x, 'original', a)
