"""a user function with a nan (inf - inf) corner: v1 apply_monotonic_binary_function vs v2 internal applicator, raw and with
the OpDescriptor contract ('fn ... None where it has no value') honoured by a wrapper"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from oracle import *
from intervals import applicator
from intervals.applicator import OpDescriptor
warnings.simplefilter('ignore')
f = lambda x, y: x - F(2, 3) * y + 0.1
def g(x, y):
    r = f(x, y)
    return None if isinstance(r, float) and math.isnan(r) else r
def run(label, c):
    try: print(f'{label}: {c()}')
    except Exception as e: print(f'{label}: RAISES {type(e).__name__}: {e}')
for pa, pb in [([(-INF, F(1, 2), False, True)], [(-INF, 0, False, True)]),
               ([(-INF, F(1, 2), False, True)], [(0, 1, True, True)]),
               ([(0, INF, True, False)], [(0, INF, True, False)]),
               ([(-INF, INF, True, True)], [(5, 5, True, True)])]:
    a1, b1, a2, b2 = mk1(pa), mk1(pb), mk2(pa), mk2(pb)
    print('==', fmt(pa), 'x', fmt(pb))
    run('  v1', lambda: a1.apply_monotonic_binary_function(f, b1))
    run('  v2 internal raw fn', lambda: V2.from_cuts(applicator.apply_binary(OpDescriptor('u', f), a2.cuts, b2.cuts)))
    run('  v2 internal fn->None at nan', lambda: V2.from_cuts(applicator.apply_binary(OpDescriptor('u', g), a2.cuts, b2.cuts)))
