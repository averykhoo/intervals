"""hand-picked __pow__ cases: print v1 and v2 side by side"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math
from fractions import Fraction as F
from common import *

def show(label, f1, f2):
    r1, e1 = run(f1)
    r2, e2 = run(f2)
    print(f'{label:42s} v1: {e1 or r1!s:40s} v2: {e2 or r2!s}')

cases = [
    ('[1,2] ** 2', lambda: V1(1, 2) ** 2, lambda: V2(1, 2) ** 2),
    ('[1,2] ** -1', lambda: V1(1, 2) ** -1, lambda: V2(1, 2) ** -1),
    ('[3] ** -1 (exact 1/3)', lambda: V1(3) ** -1, lambda: V2(3) ** -1),
    ('(1,2] ** -2', lambda: V1(start=1, end=2, start_closed=False) ** -2, lambda: V2(1, 2, start_closed=False) ** -2),
    ('[1,4] ** 0.5', lambda: V1(1, 4) ** 0.5, lambda: V2(1, 4) ** 0.5),
    ('[1,4] ** F(1,2)', lambda: V1(1, 4) ** F(1, 2), lambda: V2(1, 4) ** F(1, 2)),
    ('[2,3] ** 0', lambda: V1(2, 3) ** 0, lambda: V2(2, 3) ** 0),
    ('[2,3] ** 2.0', lambda: V1(2, 3) ** 2.0, lambda: V2(2, 3) ** 2.0),
    ('[1/2,2] ** [-1,1]', lambda: V1(0.5, 2) ** V1(-1, 1), lambda: V2(F(1, 2), 2) ** V2(-1, 1)),
    ('[2,3] ** [2] (interval)', lambda: V1(2, 3) ** V1(2), lambda: V2(2, 3) ** V2(2)),
    ('[2,3] ** {[1],[3]}', lambda: V1(2, 3) ** V1(1).union(V1(3)), lambda: V2(2, 3) ** (V2(1) | V2(3))),
    ('[0] ** 2', lambda: V1(0) ** 2, lambda: V2(0) ** 2),
    ('[0] ** 0', lambda: V1(0) ** 0, lambda: V2(0) ** 0),
    ('[0] ** -1', lambda: V1(0) ** -1, lambda: V2(0) ** -1),
    ('[0] ** 0.5', lambda: V1(0) ** 0.5, lambda: V2(0) ** 0.5),
    ('[0] ** [0,2]', lambda: V1(0) ** V1(0, 2), lambda: V2(0) ** V2(0, 2)),
    ('[0] ** [-1,1]', lambda: V1(0) ** V1(-1, 1), lambda: V2(0) ** V2(-1, 1)),
    ('[0] ** [-2,0]', lambda: V1(0) ** V1(-2, 0), lambda: V2(0) ** V2(-2, 0)),
    ('[0] ** [0]', lambda: V1(0) ** V1(0), lambda: V2(0) ** V2(0)),
    ('[0,2] ** 2', lambda: V1(0, 2) ** 2, lambda: V2(0, 2) ** 2),
    ('[0,2] ** -1', lambda: V1(0, 2) ** -1, lambda: V2(0, 2) ** -1),
    ('[0,4] ** 0.5', lambda: V1(0, 4) ** 0.5, lambda: V2(0, 4) ** 0.5),
    ('[0,2] ** [1,2]', lambda: V1(0, 2) ** V1(1, 2), lambda: V2(0, 2) ** V2(1, 2)),
    ('[-3,-1] ** 2', lambda: V1(-3, -1) ** 2, lambda: V2(-3, -1) ** 2),
    ('[-3,-1] ** 3', lambda: V1(-3, -1) ** 3, lambda: V2(-3, -1) ** 3),
    ('[-3,-1] ** -1', lambda: V1(-3, -1) ** -1, lambda: V2(-3, -1) ** -1),
    ('[-3,1] ** 2', lambda: V1(-3, 1) ** 2, lambda: V2(-3, 1) ** 2),
    ('[-3,1] ** 3', lambda: V1(-3, 1) ** 3, lambda: V2(-3, 1) ** 3),
    ('[-3,1] ** 0.5', lambda: V1(-3, 1) ** 0.5, lambda: V2(-3, 1) ** 0.5),
    ('[-2] ** 2', lambda: V1(-2) ** 2, lambda: V2(-2) ** 2),
    ('[1,2] ** inf', lambda: V1(1, 2) ** math.inf, lambda: V2(1, 2) ** math.inf),
    ('[1/2] ** inf', lambda: V1(0.5) ** math.inf, lambda: V2(F(1, 2)) ** math.inf),
    ('[2,3] ** -inf', lambda: V1(2, 3) ** -math.inf, lambda: V2(2, 3) ** -math.inf),
    ('[2,inf) ** 2', lambda: V1(start=2, end=math.inf, end_closed=False) ** 2, lambda: V2(2, math.inf, end_closed=False) ** 2),
    ('[2,inf) ** -1', lambda: V1(start=2, end=math.inf, end_closed=False) ** -1, lambda: V2(2, math.inf, end_closed=False) ** -1),
    ('(0,1] ** -1', lambda: V1(start=0, end=1, start_closed=False) ** -1, lambda: V2(0, 1, start_closed=False) ** -1),
    ('(0,1] ** [1,2]', lambda: V1(start=0, end=1, start_closed=False) ** V1(1, 2), lambda: V2(0, 1, start_closed=False) ** V2(1, 2)),
    ('empty ** 2', lambda: V1() ** 2, lambda: V2() ** 2),
    ('[1,2] ** empty', lambda: V1(1, 2) ** V1(), lambda: V2(1, 2) ** V2()),
    ('[1,2] ** True', lambda: V1(1, 2) ** True, lambda: V2(1, 2) ** True),
    ('[1,2] ** "2"', lambda: V1(1, 2) ** '2', lambda: V2(1, 2) ** '2'),
    ('[2] ** 100', lambda: V1(2) ** 100, lambda: V2(2) ** 100),
    ('[2.0] ** 1100 (overflow)', lambda: V1(2.0) ** 1100, lambda: V2(2.0) ** 1100),
    ('[1.1, 2.0] ** 3', lambda: V1(1.1, 2.0) ** 3, lambda: V2(1.1, 2.0) ** 3),
    ('{[1],[2,3]} ** 2', lambda: V1(1).union(V1(2, 3)) ** 2, lambda: (V2(1) | V2(2, 3)) ** 2),
]
for c in cases:
    show(*c)
