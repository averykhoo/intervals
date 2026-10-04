from common import *
import numpy as np
from decimal import Decimal
from probe_arith_helpers import run

a = I(1, False, 3, True)
A = to_v2(a)
rows = []
for name, s in [('np.float64(2)', np.float64(2)), ('np.int64(2)', np.int64(2)), ('Fraction(1,2)', F(1, 2)), ('2.5', 2.5), ('True', True),
                ('Decimal(2)', Decimal(2)), ('None', None), ('complex 2j', 2j)]:
    for sym, f in [('+', lambda x, y: x + y), ('*', lambda x, y: x * y), ('/', lambda x, y: x / y), ('-', lambda x, y: x - y)]:
        l1, l2 = run(lambda: f(a, s)), run(lambda: f(A, s))
        r1, r2 = run(lambda: f(s, a)), run(lambda: f(s, A))
        def fmt(r):
            return r[0] if r[0] != 'ok' else ('ok ' + type(r[1]).__name__ + ' ' + str(r[1])[:40])
        same_l = (l1[0] == 'ok') == (l2[0] == 'ok')
        same_r = (r1[0] == 'ok') == (r2[0] == 'ok')
        print(f'{name:14s} {sym}  I op s: v1 {fmt(l1):45s} v2 {fmt(l2):45s} | s op I: v1 {fmt(r1):50s} v2 {fmt(r2)}')
        if l1[0] == 'ok' and l2[0] == 'ok' and isinstance(l1[1], I) and isinstance(l2[1], M):
            same_reals(f'{name}{sym}', l1[1], l2[1])
        if r1[0] == 'ok' and r2[0] == 'ok' and isinstance(r1[1], I) and isinstance(r2[1], M):
            same_reals(f'r{name}{sym}', r1[1], r2[1])
    print(f'{name:14s} in: v1 {run(lambda: s in a)} v2 {run(lambda: s in A)}')
check('SELFTEST expected mismatch', False)
report_end(__file__)
