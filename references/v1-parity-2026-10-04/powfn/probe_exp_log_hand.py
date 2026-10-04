"""exp and log(base): hand cases"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math
from fractions import Fraction as F
from common import *

def show(label, f1, f2):
    r1, e1 = run(f1); r2, e2 = run(f2)
    print(f'{label:34s} v1: {e1 or s1(r1)!s:52s} v2: {e2 or r2!s}')

for label, f1, f2 in [
    ('exp [0]', lambda: V1(0).exp(), lambda: V2(0).exp()),
    ('exp [0,1]', lambda: V1(0, 1).exp(), lambda: V2(0, 1).exp()),
    ('exp [0.0,1.0]', lambda: V1(0.0, 1.0).exp(), lambda: V2(0.0, 1.0).exp()),
    ('exp (0,1)', lambda: V1(start=0, end=1, start_closed=False, end_closed=False).exp(), lambda: V2(0, 1, start_closed=False, end_closed=False).exp()),
    ('exp (-inf, 0]', lambda: V1(start=-math.inf, end=0, start_closed=False).exp(), lambda: V2(-math.inf, 0, start_closed=False).exp()),
    ('exp [-inf, 0] v2 only', lambda: None, lambda: V2(-math.inf, 0).exp()),
    ('exp [1, inf)', lambda: V1(start=1, end=math.inf, end_closed=False).exp(), lambda: V2(1, math.inf, end_closed=False).exp()),
    ('exp [700, 710] (overflow)', lambda: V1(700, 710).exp(), lambda: V2(700, 710).exp()),
    ('exp [-1000]', lambda: V1(-1000).exp(), lambda: V2(-1000).exp()),
    ('exp F(1,2)', lambda: V1(F(1, 2)).exp(), lambda: V2(F(1, 2)).exp()),
    ('exp {[0],[1,2]}', lambda: V1(0).union(V1(1, 2)).exp(), lambda: (V2(0) | V2(1, 2)).exp()),
    ('exp empty', lambda: V1().exp(), lambda: V2().exp()),
    ('math.exp(A)', lambda: math.exp(V1(0)), lambda: math.exp(V2(0))),
    ('log [1,8] default', lambda: V1(1, 8).log(), lambda: V2(1, 8).log()),
    ('log [1,8] base 2', lambda: V1(1, 8).log(2), lambda: V2(1, 8).log(2)),
    ('log [1,8] base=2', lambda: V1(1, 8).log(base=2), lambda: V2(1, 8).log(base=2)),
    ('log [1,100] base 10', lambda: V1(1, 100).log(10), lambda: V2(1, 100).log(10)),
    ('log [1,8] base 0.5', lambda: V1(1, 8).log(0.5), lambda: V2(1, 8).log(0.5)),
    ('log [1,8] base F(1,2)', lambda: V1(1, 8).log(F(1, 2)), lambda: V2(1, 8).log(F(1, 2))),
    ('log [1,9] base 3.0', lambda: V1(1, 9).log(3.0), lambda: V2(1, 9).log(3.0)),
    ('log [1,8] base 1', lambda: V1(1, 8).log(1), lambda: V2(1, 8).log(1)),
    ('log [1,8] base -2', lambda: V1(1, 8).log(-2), lambda: V2(1, 8).log(-2)),
    ('log [1,8] base 0', lambda: V1(1, 8).log(0), lambda: V2(1, 8).log(0)),
    ('log [1,8] base inf', lambda: V1(1, 8).log(math.inf), lambda: V2(1, 8).log(math.inf)),
    ('log [1,8] base MI(2)', lambda: V1(1, 8).log(V1(2)), lambda: V2(1, 8).log(V2(2))),
    ('log [1,8] base MI[2,4]', lambda: V1(1, 8).log(V1(2, 4)), lambda: V2(1, 8).log(V2(2, 4))),
    ('log [1,8] base True', lambda: V1(1, 8).log(True), lambda: V2(1, 8).log(True)),
    ('log [1,8] base "2"', lambda: V1(1, 8).log('2'), lambda: V2(1, 8).log('2')),
    ('log [0, 1]', lambda: V1(0, 1).log(), lambda: V2(0, 1).log()),
    ('log [-1, 1]', lambda: V1(-1, 1).log(), lambda: V2(-1, 1).log()),
    ('log [-2, -1]', lambda: V1(-2, -1).log(), lambda: V2(-2, -1).log()),
    ('log (0, inf)', lambda: V1(start=0, end=math.inf, start_closed=False, end_closed=False).log(), lambda: V2(0, math.inf, start_closed=False, end_closed=False).log()),
    ('log empty', lambda: V1().log(), lambda: V2().log()),
    ('math.log(A)', lambda: math.log(V1(1)), lambda: math.log(V2(1))),
]:
    show(label, f1, f2)
