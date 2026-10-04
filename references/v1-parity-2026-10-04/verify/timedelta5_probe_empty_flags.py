import sys, warnings, datetime
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m
import time_interval as v1t
import intervals as v2
from intervals.time_interval import TimeDeltaInterval as T2, DateTimeInterval as D2

def run(f):
    try:
        r = f()
        return ('ok', r)
    except Exception as e:
        return ('raise', type(e).__name__ + ':' + str(e))

def desc(tag, r):
    if r[0] == 'raise': return r
    x = r[1]
    inner = getattr(x, 'interval', None) or getattr(x, '_mi', None) or x
    return ('ok', 'empty' if inner.is_empty else 'NONEMPTY', type(x).__module__ + '.' + type(x).__name__)

wrong = 0
for sc in (True, False):
    for ec in (True, False):
        kw = dict(start_closed=sc, end_closed=ec)
        rows = {
          'v1 MI': desc('', run(lambda: v1m.MultiInterval(**kw))),
          'v2 MI': desc('', run(lambda: v2.MultiInterval(**kw))),
          'v1 TD': desc('', run(lambda: v1t.TimeDeltaInterval(**kw))),
          'v2 TD': desc('', run(lambda: T2(**kw))),
          'v1 DT': desc('', run(lambda: v1t.DateTimeInterval(**kw))),
          'v2 DT': desc('', run(lambda: D2(**kw))),
        }
        print(kw)
        for k, v in rows.items(): print('   ', k, v)
        # self-check: wrong expectation that v1 always equals v2 must be caught at least once
        if rows['v1 TD'][0] != rows['v2 TD'][0]: wrong += 1
print('mismatching flag combos (v1 TD vs v2 TD):', wrong)

# the composition giving v1's raise is not a capability; does the v2 empty set equal v1's empty in every allowed combo?
e2 = T2()
print('v2 T() == T(start_closed=False):', e2 == T2(start_closed=False), '== T(False,False):', e2 == T2(start_closed=False, end_closed=False))
# v2 still refuses a half-open point (the analogous meaningful check), as v1
h = datetime.timedelta(hours=1)
print('v1 half-open point:', run(lambda: v1t.TimeDeltaInterval(h, start_closed=False)))
print('v2 half-open point:', run(lambda: T2(h, start_closed=False)))
print('v2 end without start:', run(lambda: T2(None, h)))
print('v1 end without start:', run(lambda: v1t.TimeDeltaInterval(None, h)))
