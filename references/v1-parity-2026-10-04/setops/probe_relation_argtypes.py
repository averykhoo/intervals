"""non-MI non-number arguments to issubset/issuperset/isdisjoint/overlaps"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
A1, A2 = M1(0, 2), M2(0, 2)
for name, a in [('str', '[0, 1]'), ('list', [0, 1]), ('tuple', (0, 1)), ('set', {1}), ('None', None), ('True', True)]:
    for op in ['issubset', 'issuperset', 'isdisjoint', 'overlaps']:
        out = []
        for A in (A1, A2):
            try: out.append(str(getattr(A, op)(a)))
            except Exception as e: out.append(f'raise {type(e).__name__}')
        print(f'{op}({name}): v1={out[0]:22s} v2={out[1]:22s}{"" if out[0] == out[1] else "   <-- DIFF"}')
print('sabotage caught:', 'raise TypeError' != str(A2.issubset(M2(0, 5))))
