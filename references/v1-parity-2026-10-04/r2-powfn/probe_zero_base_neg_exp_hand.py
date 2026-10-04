"""hand cases: bases mixing [0, b] pieces with interval exponents holding negatives; v1 vs v2, with warnings"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import warnings
from fractions import Fraction as F
from common import *
import intervals

def show(label, b, e, expect):
    r1, x1 = run(lambda: mk1(b) ** mk1(e))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r2 = mk2(b) ** mk2(e)
    ws = sorted({m.category.__name__ for m in w})
    ok = str(r2) == expect
    print(f'{label:34s} v1: {x1 or s1(r1)!s:52s} v2: {r2!s:28s} {ws} {"OK" if ok else "!= expected " + expect}')
    return ok

C = lambda a, b: (F(a), F(b), True, True)
O0 = lambda b: (F(0), F(b), False, True)
P = lambda a: (F(a), F(a), True, True)
res = [
    show('[0,4] ** [-1/2,1/2]', [C(0, 4)], [C(F(-1, 2), F(1, 2))], '[0, inf)'),
    show('(0,4] ** [-1/2,1/2]', [O0(4)], [C(F(-1, 2), F(1, 2))], '(0, inf)'),
    show('[0,4] ** [-1,-1/2]', [C(0, 4)], [C(-1, F(-1, 2))], '[1/4, inf)'),
    show('[0,1/4] ** [-1,0]', [C(0, F(1, 4))], [C(-1, 0)], '[1, inf)'),
    show('[0,1/4] ** (-1,0)', [C(0, F(1, 4))], [(F(-1), F(0), False, False)], '(1, inf)'),
    show('{[0,1/4],[4,9]} ** [-1/2]', [C(0, F(1, 4)), C(4, 9)], [P(F(-1, 2))], '{ [1/3, 1/2] , [2, inf) }'),
    show('{[0],[4,9]} ** [-1/2,1/2]', [P(0), C(4, 9)], [C(F(-1, 2), F(1, 2))], '{ [0] , [1/3, 3] }'),
    show('{[0],[4,9]} ** [-1/2]', [P(0), C(4, 9)], [P(F(-1, 2))], '[1/3, 1/2]'),
    show('{[0,1],[4]} ** {[-2,-1],[1,2]}', [C(0, 1), P(4)], [C(-2, -1), C(1, 2)], '{ [0, 1] , [1/16, 1/4] , [4, 16] }'.replace('{ [0, 1] , [1/16, 1/4] , [4, 16] }', '{ [0, 1] , [4, 16] }') if False else '[0, inf)'),
    show('[-1,4] ** [-1/2,1/2]', [C(-1, 4)], [C(F(-1, 2), F(1, 2))], '[0, inf)'),
]
print('all as expected' if all(res) else 'SOME DIFFER')
# sabotage: a deliberately wrong expectation must print a difference
assert not show('SABOTAGE [0,4] ** [-1/2,1/2]', [C(0, 4)], [C(F(-1, 2), F(1, 2))], '(0, inf)')
print('sabotage caught')
