"""step functions: hand cases (infinities, ndigits values and types, big sets, empty)"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math
from fractions import Fraction as F
import numpy as np
from common import *

def show(label, f1, f2):
    r1, e1 = run(f1); r2, e2 = run(f2)
    print(f'{label:34s} v1: {e1 or s1(r1)!s:44s} v2: {e2 or r2!s}'[:220])

inf = math.inf
cases = [
    ('floor [-1.5, 1.5]', lambda: math.floor(V1(-1.5, 1.5)), lambda: math.floor(V2(-1.5, 1.5))),
    ('floor (0.5, 1)', lambda: math.floor(V1(start=0.5, end=1, start_closed=False, end_closed=False)), lambda: math.floor(V2(0.5, 1, start_closed=False, end_closed=False))),
    ('floor [0, inf)', lambda: math.floor(V1(start=0, end=inf, end_closed=False)), lambda: math.floor(V2(0, inf, end_closed=False))),
    ('floor (-inf, inf)', lambda: math.floor(V1(start=-inf, end=inf, start_closed=False, end_closed=False)), lambda: math.floor(V2(-inf, inf, start_closed=False, end_closed=False))),
    ('floor [-inf, 0] v2 only', lambda: None, lambda: math.floor(V2(-inf, 0))),
    ('floor [inf] v2 only', lambda: None, lambda: math.floor(V2(inf))),
    ('ceil (0.5, 2.5]', lambda: math.ceil(V1(start=0.5, end=2.5, start_closed=False)), lambda: math.ceil(V2(0.5, 2.5, start_closed=False))),
    ('trunc (-2.5, 2.5)', lambda: math.trunc(V1(start=-2.5, end=2.5, start_closed=False, end_closed=False)), lambda: math.trunc(V2(-2.5, 2.5, start_closed=False, end_closed=False))),
    ('trunc [-0.5, 0.5]', lambda: math.trunc(V1(-0.5, 0.5)), lambda: math.trunc(V2(-0.5, 0.5))),
    ('trunc (-inf, -1]', lambda: math.trunc(V1(start=-inf, end=-1, start_closed=False)), lambda: math.trunc(V2(-inf, -1, start_closed=False))),
    ('floor [0, 5000] (cap)', lambda: math.floor(V1(0, 5000)), lambda: math.floor(V2(0, 5000))),
    ('floor [0.5, 999.5] (1000 vals)', lambda: math.floor(V1(0.5, 999.5)), lambda: math.floor(V2(F(1, 2), F(1999, 2)))),
    ('floor empty', lambda: math.floor(V1()), lambda: math.floor(V2())),
    ('round [0.5, 2.5] default', lambda: round(V1(0.5, 2.5)), lambda: round(V2(0.5, 2.5))),
    ('round [1/2, 5/2] default', lambda: round(V1(F(1, 2), F(5, 2))), lambda: round(V2(F(1, 2), F(5, 2)))),
    ('round(A, 0) [0.5, 2.5]', lambda: round(V1(0.5, 2.5), 0), lambda: round(V2(0.5, 2.5), 0)),
    ('round(A, None) [0.5, 2.5]', lambda: round(V1(0.5, 2.5), None), lambda: round(V2(0.5, 2.5), None)),
    ('round(A, None) [0, inf)', lambda: round(V1(start=0, end=inf, end_closed=False), None), lambda: round(V2(0, inf, end_closed=False), None)),
    ('round [0, inf)', lambda: round(V1(start=0, end=inf, end_closed=False)), lambda: round(V2(0, inf, end_closed=False))),
    ('round [2.675] 2', lambda: round(V1(2.675), 2), lambda: round(V2(2.675), 2)),
    ('round [2.675, 2.676] 2', lambda: round(V1(2.675, 2.676), 2), lambda: round(V2(2.675, 2.676), 2)),
    ('round [0.125, 0.135] 2 exact', lambda: round(V1(F(1, 8), F(27, 200)), 2), lambda: round(V2(F(1, 8), F(27, 200)), 2)),
    ('round [1234, 1789] -2', lambda: round(V1(1234, 1789), -2), lambda: round(V2(1234, 1789), -2)),
    ('round [1234, 1789] -5', lambda: round(V1(1234, 1789), -5), lambda: round(V2(1234, 1789), -5)),
    ('round [1/3] 400', lambda: round(V1(F(1, 3)), 400), lambda: round(V2(F(1, 3)), 400)),
    ('round [0.1, 0.2] 400 (float)', lambda: round(V1(0.1, 0.2), 400), lambda: round(V2(0.1, 0.2), 400)),
    ('round [0.1] 30 (float)', lambda: round(V1(0.1), 30), lambda: round(V2(0.1), 30)),
    ('round(A, 1.0)', lambda: round(V1(0.5, 2.5), 1.0), lambda: round(V2(0.5, 2.5), 1.0)),
    ('round(A, True)', lambda: round(V1(0.5, 2.5), True), lambda: round(V2(0.5, 2.5), True)),
    ('round(A, np.int64(1))', lambda: round(V1(0.5, 0.7), np.int64(1)), lambda: round(V2(0.5, 0.7), np.int64(1))),
    ('round(A, "1")', lambda: round(V1(0.5, 2.5), '1'), lambda: round(V2(0.5, 2.5), '1')),
    ('A.round(1) method', lambda: V1(0.5, 0.7).__round__(1), lambda: V2(0.5, 0.7).round(1)),
    ('round_ties_away [0.5, 2.5] v2', lambda: None, lambda: V2(0.5, 2.5).round_ties_away()),
    ('floor [-0.0, 0.5]', lambda: math.floor(V1(-0.0, 0.5)), lambda: math.floor(V2(-0.0, 0.5))),
    ('floor [10**20 + 0.5 as F]', lambda: math.floor(V1(F(10**20 * 2 + 1, 2))), lambda: math.floor(V2(F(10**20 * 2 + 1, 2)))),
    ('floor [1e300, 1e300] float', lambda: math.floor(V1(1e300)), lambda: math.floor(V2(1e300))),
    ('floor A.floor() method', lambda: None, lambda: V2(0.5, 2.5).floor()),
]
for c in cases:
    show(*c)
# sabotage: claim floor (0.5, 1) is {1}: must differ
assert pieces2(math.floor(V2(0.5, 1, start_closed=False, end_closed=False))) != [(1, 1, True, True)]
print('sabotage caught')
