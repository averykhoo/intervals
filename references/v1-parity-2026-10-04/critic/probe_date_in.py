"""re-check datetime row '__contains__ of a date (whole day a subset) [EQUAL]': its loop check has `or True`, and the
second loop draws random-microsecond ends, so no set ever ends at a day's last microsecond. hand edge cases here."""
import sys, datetime as dt, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
D = dt.datetime; d0 = dt.date(2024, 1, 1)
cases = [
    ('[d 00:00, d 23:59:59.999999] from datetimes', (D(2024,1,1), D(2024,1,1,23,59,59,999999)), {}),
    ('[d 00:00, d+1 00:00:00.000001)', (D(2024,1,1), D(2024,1,2,0,0,0,1)), {'end_closed': False}),
    ('[d 00:00, d+1 00:00) built half-open (no snap: end has us=0 -> v1 snaps)', (D(2024,1,1), D(2024,1,2)), {'end_closed': False}),
    ('[d 00:00:00.000001, d+1 12:00:00.5]', (D(2024,1,1,0,0,0,1), D(2024,1,2,12,0,0,500000)), {}),
    ('DTI(d) itself', (d0,), {}),
]
diff = 0
for name, args, kw in cases:
    a1, a2 = v1t.DateTimeInterval(*args, **kw), v2t.DateTimeInterval(*args, **kw)
    r1, r2 = d0 in a1, d0 in a2
    diff += r1 != r2
    print(f'{name}: v1 {r1}  v2 {r2}', '' if r1 == r2 else '<-- DIFFER')
print('differences:', diff)
assert diff > 0, 'sabotage: expected at least one day-edge difference; the old sweep reported 0'
