import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import time_interval as v1t, multi_interval as v1
import intervals as v2
from intervals.time_interval import TimeDeltaInterval as T2, DateTimeInterval as D2
T1 = v1t.TimeDeltaInterval
e1 = T1()
print('v1 TDI defines __bool__/__len__:', '__bool__' in vars(T1), '__len__' in vars(T1), 'DTI:', '__bool__' in vars(v1t.DateTimeInterval), '__len__' in vars(v1t.DateTimeInterval))
print('v1 empty TDI: bool', bool(e1), 'is_empty', getattr(e1, 'is_empty', 'n/a'), 'inner bool', bool(e1.interval))
print('v1 empty MultiInterval bool', bool(v1.MultiInterval()))
e2 = T2()
print('v2 empty TDI: bool', bool(e2), 'is_empty', e2.is_empty)
x1 = T1(dt.timedelta(0), dt.timedelta(0)); x2 = T2(dt.timedelta(0), dt.timedelta(0))
print('zero point: v1', bool(x1), 'v2', bool(x2))
print('v1 empty DTI bool', bool(v1t.DateTimeInterval()), 'v2', bool(D2()))
assert bool(e2) is False  # a wrong expectation (True) would be caught
