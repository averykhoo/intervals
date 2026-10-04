"""reciprocal hand cases: int ends, infinite ends, zero ends, empty"""
import sys, os, warnings, math
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
def show(label, f1, f2):
    out = []
    for f in (f1, f2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                r = f(); r = f'{r}' + (f' endpoints={r.endpoints}' if isinstance(r, V1) else '')
            except Exception as e:
                r = f'RAISES {type(e).__name__}: {e}'
        out.append(f'{r} {[x.category.__name__ for x in w] or ""}')
    print(f'{label:28s} v1: {out[0]}\n{"":28s} v2: {out[1]}')
show('[1,3] ints', lambda: V1(1, 3).reciprocal(), lambda: V2(1, 3).reciprocal())
show('1/3 in v1([1,3].recip)?', lambda: F(1, 3) in V1(1, 3).reciprocal(), lambda: F(1, 3) in V2(1, 3).reciprocal())
show('(0,1]', lambda: V1(0, 1, start_closed=False).reciprocal(), lambda: V2(0, 1, start_closed=False).reciprocal())
show('[-1,0)', lambda: V1(-1, 0, end_closed=False).reciprocal(), lambda: V2(-1, 0, end_closed=False).reciprocal())
show('[1,inf)', lambda: V1(1, math.inf, end_closed=False).reciprocal(), lambda: V2(1, math.inf, end_closed=False).reciprocal())
show('(-inf,-1]', lambda: V1(-math.inf, -1, start_closed=False).reciprocal(), lambda: V2(-math.inf, -1, start_closed=False).reciprocal())
show('(0,inf)', lambda: V1(0, math.inf, start_closed=False, end_closed=False).reciprocal(), lambda: V2(0, math.inf, start_closed=False, end_closed=False).reciprocal())
show('[0]', lambda: V1(0).reciprocal(), lambda: V2(0).reciprocal())
show('[0,1]', lambda: V1(0, 1).reciprocal(), lambda: V2(0, 1).reciprocal())
show('empty', lambda: V1().reciprocal(), lambda: V2().reciprocal())
show('[0.1]', lambda: V1(0.1).reciprocal(), lambda: V2(0.1).reciprocal())
show('[3] (int)', lambda: V1(3).reciprocal(), lambda: V2(3).reciprocal())
show('{(-1,0) u (0,1)}', lambda: V1(-1, 0, start_closed=False, end_closed=False).union(V1(0, 1, start_closed=False, end_closed=False)).reciprocal(),
     lambda: V2.parse('{ (-1, 0) , (0, 1) }').reciprocal())
