"""hand-picked edge cases of + - * / (and reflected), v1 vs v2"""
import sys, os, warnings, operator, math
from fractions import Fraction as F
from decimal import Decimal
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def show(label, f1, f2):
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter('always')
        try:
            r1 = f1(); 
            if isinstance(r1, V1): r1._consistency_check()
            r1 = f'{r1!s} endpoints={r1.endpoints}' if isinstance(r1, V1) else repr(r1)
        except Exception as e:
            r1 = f'RAISES {type(e).__name__}: {e}'
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter('always')
        try:
            r2 = f2(); r2 = str(r2) if isinstance(r2, V2) else repr(r2)
        except Exception as e:
            r2 = f'RAISES {type(e).__name__}: {e}'
    ws1 = [x.category.__name__ for x in w1]; ws2 = [x.category.__name__ for x in w2]
    print(f'{label:45s} | v1: {r1} {ws1 or ""}\n{"":45s} | v2: {r2} {ws2 or ""}')

a1, a2 = V1(0, 1), V2(0, 1)
b1, b2 = V1(2, 3, start_closed=False, end_closed=False), V2(2, 3, start_closed=False, end_closed=False)
show('[0,1]*(2,3)', lambda: a1 * b1, lambda: a2 * b2)
show('(2,3)*[0,1]', lambda: b1 * a1, lambda: b2 * a2)
n1, n2 = V1(-math.inf, 0, start_closed=False, end_closed=False), V2(-math.inf, 0, start_closed=False, end_closed=False)
p1, p2 = V1(0, math.inf, start_closed=False, end_closed=False), V2(0, math.inf, start_closed=False, end_closed=False)
show('(-inf,0)+(0,inf)', lambda: n1 + p1, lambda: n2 + p2)
show('(0,inf)+(-inf,0)', lambda: p1 + n1, lambda: p2 + n2)
show('(-inf,0)-(-inf,0)', lambda: n1 - n1, lambda: n2 - n2)
show('(0,inf)-(0,inf)', lambda: p1 - p1, lambda: p2 - p2)
show('(-inf,0)*(0,inf)', lambda: n1 * p1, lambda: n2 * p2)
show('[0]*(0,inf)', lambda: V1(0) * p1, lambda: V2(0) * p2)
show('0*(0,inf) (scalar left)', lambda: 0 * p1, lambda: 0 * p2)
show('(0,inf)/(0,inf)', lambda: p1 / p1, lambda: p2 / p2)
# empty operands
e1, e2 = V1(), V2()
show('empty + [0,1]', lambda: e1 + a1, lambda: e2 + a2)
show('[0,1] + empty', lambda: a1 + e1, lambda: a2 + e2)
show('empty + 1', lambda: e1 + 1, lambda: e2 + 1)
show('1 + empty', lambda: 1 + e1, lambda: 1 + e2)
show('1 - empty', lambda: 1 - e1, lambda: 1 - e2)
show('[0,1] / empty', lambda: a1 / e1, lambda: a2 / e2)
show('empty / [0,1]', lambda: e1 / a1, lambda: e2 / a2)
show('1 / empty', lambda: 1 / e1, lambda: 1 / e2)
show('empty * empty', lambda: e1 * e1, lambda: e2 * e2)
# division by scalar zero
show('[1,2] / 0', lambda: V1(1, 2) / 0, lambda: V2(1, 2) / 0)
show('[1,2] / 0.0', lambda: V1(1, 2) / 0.0, lambda: V2(1, 2) / 0.0)
show('[-1,2] / 0', lambda: V1(-1, 2) / 0, lambda: V2(-1, 2) / 0)
show('[0] / 0', lambda: V1(0) / 0, lambda: V2(0) / 0)
show('empty / 0', lambda: e1 / 0, lambda: e2 / 0)
show('[1,2] / [0]', lambda: V1(1, 2) / V1(0), lambda: V2(1, 2) / V2(0))
show('[1,2] / [0,1]', lambda: V1(1, 2) / V1(0, 1), lambda: V2(1, 2) / V2(0, 1))
show('[1,2] / [-1,1]', lambda: V1(1, 2) / V1(-1, 1), lambda: V2(1, 2) / V2(-1, 1))
show('1 / [-1,1]', lambda: 1 / V1(-1, 1), lambda: 1 / V2(-1, 1))
show('3 / [1,2]', lambda: 3 / V1(1, 2), lambda: 3 / V2(1, 2))
show('3 - [1,2]', lambda: 3 - V1(1, 2), lambda: 3 - V2(1, 2))
show('[1,2] - 3', lambda: V1(1, 2) - 3, lambda: V2(1, 2) - 3)
show('2 * [1,2)', lambda: 2 * V1(1, 2, end_closed=False), lambda: 2 * V2(1, 2, end_closed=False))
show('-1 * [1,2)', lambda: -1 * V1(1, 2, end_closed=False), lambda: -1 * V2(1, 2, end_closed=False))
show('0 * [1,2)', lambda: 0 * V1(1, 2, end_closed=False), lambda: 0 * V2(1, 2, end_closed=False))
show('[1,2) * 0', lambda: V1(1, 2, end_closed=False) * 0, lambda: V2(1, 2, end_closed=False) * 0)
show('inf * [1,2]', lambda: math.inf * V1(1, 2), lambda: math.inf * V2(1, 2))
show('[1,2] + inf', lambda: V1(1, 2) + math.inf, lambda: V2(1, 2) + math.inf)
show('[1,2] + nan', lambda: V1(1, 2) + math.nan, lambda: V2(1, 2) + math.nan)
# operand types
show('[1,2] + 1 (int)', lambda: V1(1, 2) + 1, lambda: V2(1, 2) + 1)
show('[1,2] + 0.1 (float)', lambda: V1(1, 2) + 0.1, lambda: V2(1, 2) + 0.1)
show('[1,2] + F(1,3)', lambda: V1(1, 2) + F(1, 3), lambda: V2(1, 2) + F(1, 3))
show('[1,2] / 3 (int: exact?)', lambda: V1(1, 2) / 3, lambda: V2(1, 2) / 3)
show('[1,2] + True (bool)', lambda: V1(1, 2) + True, lambda: V2(1, 2) + True)
show('True * [1,2] (bool)', lambda: True * V1(1, 2), lambda: True * V2(1, 2))
show('[1,2] + Decimal(1)', lambda: V1(1, 2) + Decimal(1), lambda: V2(1, 2) + Decimal(1))
show('Decimal(1) + [1,2]', lambda: Decimal(1) + V1(1, 2), lambda: Decimal(1) + V2(1, 2))
show("[1,2] + '1' (str)", lambda: V1(1, 2) + '1', lambda: V2(1, 2) + '1')
show("'1' + [1,2] (str)", lambda: '1' + V1(1, 2), lambda: '1' + V2(1, 2))
show('[1,2] + None', lambda: V1(1, 2) + None, lambda: V2(1, 2) + None)
show('[1,2] + complex', lambda: V1(1, 2) + 1j, lambda: V2(1, 2) + 1j)
show('[1,2] / "x"', lambda: V1(1, 2) / 'x', lambda: V2(1, 2) / 'x')
show('"x" / [1,2]', lambda: 'x' / V1(1, 2), lambda: 'x' / V2(1, 2))
show('[1,2] / None', lambda: V1(1, 2) / None, lambda: V2(1, 2) / None)
show('None / [1,2]', lambda: None / V1(1, 2), lambda: None / V2(1, 2))
show('[1,2] / Decimal(2)', lambda: V1(1, 2) / Decimal(2), lambda: V2(1, 2) / Decimal(2))
show('Decimal(2) / [1,2]', lambda: Decimal(2) / V1(1, 2), lambda: Decimal(2) / V2(1, 2))
# float exactness / rounding
show('[0.1] + [0.2]', lambda: V1(0.1) + V1(0.2), lambda: V2(0.1) + V2(0.2))
show('[0.1, 0.3] * 3', lambda: V1(0.1, 0.3) * 3, lambda: V2(0.1, 0.3) * 3)
show('[1e308] * 10 (overflow)', lambda: V1(1e308) * 10, lambda: V2(1e308) * 10)
show('[1e308, 2e308) ... [1e308,1.5e308] + 1e308', lambda: V1(1e308, 1.5e308) + 1e308, lambda: V2(1e308, 1.5e308) + 1e308)
show('[-0.0] + 1', lambda: V1(-0.0) + 1, lambda: V2(-0.0) + 1)
show('OutwardMI [0.1]+[0.2]', lambda: 'n/a', lambda: v2.OutwardMultiInterval(0.1) + v2.OutwardMultiInterval(0.2))
# deliberately wrong expectation, so this file can fail
assert str(V2(0, 1) * V2(2, 3, start_closed=False, end_closed=False)) == '[0, 3)'
try:
    assert str(V2(0, 1) * V2(2, 3, start_closed=False, end_closed=False)) == '(0, 3)'  # v1's answer
    print('SELF-CHECK FAILED: wrong expectation not caught')
except AssertionError:
    print('self-check: a wrong expectation is caught')
