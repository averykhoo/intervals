from common import *
import numpy as np

def outcome(f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = f()
        return ('ok', r, [str(x.message) for x in w])
    except Exception as e:
        return ('raise', type(e).__name__, str(e)[:80])

H = dt.timedelta(hours=1)
cases = {
  'empty': (lambda: T1(), lambda: T2()),
  'point': (lambda: T1(H), lambda: T2(H)),
  'zero point': (lambda: T1(dt.timedelta(0)), lambda: T2(dt.timedelta(0))),
  'neg point': (lambda: T1(-H), lambda: T2(-H)),
  'closed': (lambda: T1(-H, 2*H), lambda: T2(-H, 2*H)),
  'open-start': (lambda: T1(-H, 2*H, start_closed=False), lambda: T2(-H, 2*H, start_closed=False)),
  'open-end': (lambda: T1(-H, 2*H, end_closed=False), lambda: T2(-H, 2*H, end_closed=False)),
  'open both': (lambda: T1(-H, 2*H, start_closed=False, end_closed=False), lambda: T2(-H, 2*H, start_closed=False, end_closed=False)),
  'end==start closed': (lambda: T1(H, H), lambda: T2(H, H)),
  'end==start open-end': (lambda: T1(H, H, end_closed=False), lambda: T2(H, H, end_closed=False)),
  'end==start open-start': (lambda: T1(H, H, start_closed=False), lambda: T2(H, H, start_closed=False)),
  'reversed': (lambda: T1(2*H, H), lambda: T2(2*H, H)),
  'end only': (lambda: T1(None, H), lambda: T2(None, H)),
  'single open-start': (lambda: T1(H, start_closed=False), lambda: T2(H, start_closed=False)),
  'single open both': (lambda: T1(H, start_closed=False, end_closed=False), lambda: T2(H, start_closed=False, end_closed=False)),
  'empty open flags': (lambda: T1(start_closed=False), lambda: T2(start_closed=False)),
  'empty both open': (lambda: T1(start_closed=False, end_closed=False), lambda: T2(start_closed=False, end_closed=False)),
  'pd.Timedelta': (lambda: T1(pd.Timedelta(hours=1), pd.Timedelta(hours=2)), lambda: T2(pd.Timedelta(hours=1), pd.Timedelta(hours=2))),
  'pd.Timedelta mixed': (lambda: T1(pd.Timedelta(hours=1), 2*H), lambda: T2(pd.Timedelta(hours=1), 2*H)),
  'pd.Timedelta ns': (lambda: T1(pd.Timedelta(1, unit='ns'), pd.Timedelta(3, unit='ns')), lambda: T2(pd.Timedelta(1, unit='ns'), pd.Timedelta(3, unit='ns'))),
  'timedelta.max': (lambda: T1(dt.timedelta.max), lambda: T2(dt.timedelta.max)),
  'timedelta.min..max': (lambda: T1(dt.timedelta.min, dt.timedelta.max), lambda: T2(dt.timedelta.min, dt.timedelta.max)),
  'int bound': (lambda: T1(1, 2), lambda: T2(1, 2)),
  'float bound': (lambda: T1(1.0), lambda: T2(1.0)),
  'str bound': (lambda: T1('1h'), lambda: T2('1h')),
  'datetime bound': (lambda: T1(dt.datetime(2020, 1, 1)), lambda: T2(dt.datetime(2020, 1, 1))),
  'np.timedelta64': (lambda: T1(np.timedelta64(3, 's')), lambda: T2(np.timedelta64(3, 's'))),
  'NaT': (lambda: T1(pd.NaT), lambda: T2(pd.NaT)),
  'None None': (lambda: T1(None, None), lambda: T2(None, None)),
  'start_closed=None': (lambda: T1(H, 2*H, start_closed=None), lambda: T2(H, 2*H, start_closed=None)),
  '-0 microsecond': (lambda: T1(-dt.timedelta(0), H), lambda: T2(-dt.timedelta(0), H)),
}
for name, (f1, f2) in cases.items():
    o1, o2 = outcome(f1), outcome(f2)
    c1 = canon1(o1[1], exact=True) if o1[0] == 'ok' else o1
    c2 = canon2(o2[1]) if o2[0] == 'ok' else o2
    if o1[0] == 'ok' and o2[0] == 'ok':
        same = check(name, c1, c2)
    else:
        same = check(name, o1[0], o2[0]) and (o1[0] == 'ok' or o1[1] == o2[1])
    print(f"{'SAME' if same else 'DIFF'} {name}: v1={c1 if o1[0]=='ok' else o1} {o1[2] if o1[0]=='ok' and o1[2] else ''} | v2={c2 if o2[0]=='ok' else o2}")
# exactness: timedelta.max reading
print('v1 tdmax seconds', repr(T1(dt.timedelta.max).interval.endpoints[0][0]), 'exact', dt.timedelta.max.days*86400+dt.timedelta.max.seconds+Fraction(dt.timedelta.max.microseconds, US))
print("v1 tdmax infimum", outcome(lambda: T1(dt.timedelta.max).infimum), " v2 inf", T2(dt.timedelta.max).inf == dt.timedelta.max)
print("v1 1ns total_seconds", pd.Timedelta(1, unit="ns").total_seconds(), "v2 inf_seconds", T2(pd.Timedelta(1, unit="ns")).inf_seconds)
# sabotage: a deliberately wrong expectation must be caught
assert not check('sabotage', canon1(T1(H, 2*H)), canon2(T2(H, 3*H)))
FAILS.pop()
report('probe_ctor')
