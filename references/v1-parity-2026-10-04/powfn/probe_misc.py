import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math
import numpy as np
from fractions import Fraction as F
from common import *
A1 = V1(start=0, end=math.inf, end_closed=False)
print('v1 A.__round__(None):', run(lambda: A1.__round__(None))[1])
for label, f1, f2 in [
    ('[1,2] ** np.float64(2.0)', lambda: V1(1, 2) ** np.float64(2.0), lambda: V2(1, 2) ** np.float64(2.0)),
    ('[1,2] ** np.int64(-1)', lambda: V1(1, 2) ** np.int64(-1), lambda: V2(1, 2) ** np.int64(-1)),
    ('[1,4] ** np.float64(0.5)', lambda: V1(1, 4) ** np.float64(0.5), lambda: V2(1, 4) ** np.float64(0.5)),
    ('np.float64(2) ** [1,3]', lambda: np.float64(2) ** V1(1, 3), lambda: np.float64(2) ** V2(1, 3)),
    ('[1,2] ** nan', lambda: V1(1, 2) ** math.nan, lambda: V2(1, 2) ** math.nan),
    ('[1,2].exp() then .log() roundtrip', lambda: None, lambda: V2(1, 2).exp().log()),
]:
    r1, e1 = run(f1); r2, e2 = run(f2)
    print(f'{label:34s} v1: {e1 or (s1(r1) if r1 is not None else None)!s:40s} v2: {e2 or r2!s}')
