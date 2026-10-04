import sys, warnings, datetime
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import interval as v1i
import multi_interval as v1
import intervals as v2
from intervals import MultiInterval as MI
import numpy as np

def run(f):
    try:
        r = f()
        return ('ok', r)
    except Exception as e:
        return (type(e).__name__, str(e)[:60])

def pieces(m):
    return [tuple(p) for p in m.pieces()] if hasattr(m, 'pieces') else str(m)

flags = ['no', 'yes', '', None, 0, 1, 2, [], [0], np.bool_(False), np.bool_(True), True, False, 'False']
print('--- v1 Interval(0, start_open=f, 1, end_closed=True) vs v2 MI(0,1,start_closed=not-f) ---')
for f in flags:
    a = run(lambda: v1i.Interval(0, f, 1, True))
    print(repr(f), 'v1 Interval:', a[0], '|', end=' ')
    b = run(lambda: v1.MultiInterval(0, 1, start_closed=f))
    print('v1 MI start_closed:', b[0], (b[1].endpoints if b[0]=='ok' else b[1]), '|', end=' ')
    c = run(lambda: MI(0, 1, start_closed=f))
    print('v2 MI start_closed:', c[0], str(c[1]))
print('--- membership of 0 under start_closed="False" (string) ---')
m = MI(0, 1, start_closed='False')
print('v2 MI(0,1,start_closed="False") =', m, '0 in it:', 0 in m)
print('--- other v2 entry points ---')
for name, f in [
    ('from_pieces x', lambda: MI.from_pieces([(0, 1, 'x', 'x')])),
    ('from_pieces None', lambda: MI.from_pieces([(0, 1, None, None)])),
    ('Builder.add_piece', lambda: v2.kernel.Builder().add_piece(0, 1, 'no', 'no').build() if hasattr(v2,'kernel') else None),
    ('point a/b', lambda: MI(5, start_closed='a', end_closed='b')),
    ('point no/no', lambda: MI(5, start_closed='no', end_closed='no')),
    ('point False/0', lambda: MI(5, start_closed=False, end_closed=0)),
    ('Outward no', lambda: v2.OutwardMultiInterval(0, 1, start_closed='no')),
    ('DateTimeInterval no', lambda: v2.DateTimeInterval(datetime.datetime(2024,1,1), datetime.datetime(2024,1,2), start_closed='no')),
]:
    print(name, run(f))
# v1 MI point
print('v1 MI point no/no', run(lambda: v1.MultiInterval(5, start_closed='no', end_closed='no').endpoints))
# sabotage: assert a wrong expectation is caught
r = run(lambda: MI(0, 1, start_closed='no'))
caught = not (r[0] == 'TypeError')
print('sabotage (expect v2 TypeError, flag if not):', 'CAUGHT' if caught else 'missed')
