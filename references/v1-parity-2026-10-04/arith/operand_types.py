"""operand types on each side of + - * /: numpy scalars, numpy arrays, gmpy2, bool, int/float/Fraction mixes"""
import sys, os, warnings, operator, math
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
import gmpy2
warnings.simplefilter('ignore')
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

def res(f):
    try:
        r = f()
        if isinstance(r, np.ndarray):
            return f'ndarray{r.shape} dtype={r.dtype} [{", ".join(str(x) for x in r.ravel())}]'
        return f'{type(r).__name__} {r}'
    except Exception as e:
        return f'RAISES {type(e).__name__}: {str(e)[:70]}'

S1, S2 = V1(F(1), F(2), end_closed=False), V2(F(1), F(2), end_closed=False)
scalars = {'np.float64(0.5)': np.float64(0.5), 'np.int64(3)': np.int64(3), 'np.float32(0.1)': np.float32(0.1),
           'np.bool_(True)': np.bool_(True), 'gmpy2.mpq(1,3)': gmpy2.mpq(1, 3), 'gmpy2.mpz(2)': gmpy2.mpz(2),
           'gmpy2.mpfr(0.5)': gmpy2.mpfr(0.5), 'int 3': 3, 'float 0.5': 0.5, 'F(1,3)': F(1, 3),
           'np.array([1,2])': np.array([1, 2])}
for sname, s in scalars.items():
    for oname, op in OPS.items():
        r1 = res(lambda: op(S1, s)); r2 = res(lambda: op(S2, s))
        l1 = res(lambda: op(s, S1)); l2 = res(lambda: op(s, S2))
        print(f'[1,2) {oname} {sname:17s} v1: {r1}\n{"":28s} v2: {r2}')
        print(f'{sname:17s} {oname} [1,2) v1: {l1}\n{"":28s} v2: {l2}')
# exact check: does v1 keep np.int64/F exact and does v2 agree as sets?
for s in (np.int64(3), F(1, 3), gmpy2.mpq(1, 3), np.float32(0.1)):
    a = S1 * s; b = S2 * s
    print('set-compare [1,2) *', repr(s), 'finite diffs:', finite_diff(a, b) if isinstance(a, V1) else 'n/a')
