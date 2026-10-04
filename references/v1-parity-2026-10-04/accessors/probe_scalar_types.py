"""accessors on -0.0 and numpy scalar ends; comparisons with numpy scalars"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *
import numpy as np
rows = []
for lo, hi in [(-0.0, 1), (-1, -0.0), (np.float64(0.5), np.float64(2.0)), (np.int64(1), np.int64(3)), (np.float32(0.1), 1), (np.int64(2), np.int64(2))]:
    a = v1.MultiInterval(lo, hi) if lo != hi else v1.MultiInterval(lo)
    b = v2.MultiInterval(lo, hi) if lo != hi else v2.MultiInterval(lo)
    for n1, n2 in [('infimum', 'inf'), ('supremum', 'sup'), ('is_integral', 'is_integral'), ('is_positive', 'is_positive'),
                   ('is_non_negative', 'is_non_negative'), ('degenerate_points', 'degenerate_points'), ('cardinality', 'size')]:
        r1, r2 = getattr(a, n1), getattr(b, n2)
        if n1 == 'cardinality':
            same = (r1[0], r1[1], r1[2]) == (r2.rays, r2.length, 2 * r2.points)
        else:
            same = r1 == r2
        print(f'{lo!r:>22} {hi!r:>22} {n1:18} v1={r1!r} ({type(r1).__name__})  v2={r2!r} ({type(r2).__name__})  {"same" if same else "DIFFER"}')
    for x in [np.float64(lo), np.int64(3)]:
        r1 = a == x; r1l = a < x
        r2 = b == v2.MultiInterval(x); r2l = b.sort_key < v2.MultiInterval(x).sort_key
        print(f'   compare with {x!r}: v1 == {r1} < {r1l} | v2 == {r2} sort_key< {r2l}', 'same' if (r1, r1l) == (r2, r2l) else 'DIFFER')
# sabotage: float32(0.1) must not equal 0.1 exactly in v2 (exact value kept)
assert v2.MultiInterval(np.float32(0.1)).inf != 0.1
