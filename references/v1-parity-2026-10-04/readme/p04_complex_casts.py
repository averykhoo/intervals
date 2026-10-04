# README: "does not and will never support complex numbers"; v1 casts a degenerate set to complex/float/int
from common import *
warnings.simplefilter('ignore')
def run(f):
    try:
        r = f(); return 'NotImplemented' if r is NotImplemented else repr(r)
    except Exception as e: return f'{type(e).__name__}: {str(e)[:70]}'
for name, f1, f2 in [
    ('complex([2])', lambda: complex(V1(2)), lambda: complex(MI(2))),
    ('float([2])', lambda: float(V1(2)), lambda: float(MI(2))),
    ('int([2])', lambda: int(V1(2)), lambda: int(MI(2))),
    ('float([1,2])', lambda: float(V1(1, 2)), lambda: float(MI(1, 2))),
    ('complex input', lambda: V1(1, 2) + 1j, lambda: MI(1, 2) + 1j),
    ('V(1j)', lambda: V1(1j), lambda: MI(1j)),
    ('bool(empty)', lambda: bool(V1()), lambda: bool(MI())),
    ('bool([0])', lambda: bool(V1(0)), lambda: bool(MI(0))),
]:
    print(f'{name:15s} v1: {run(f1):50s} v2: {run(f2)}')
