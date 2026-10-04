"""re-check timedelta row '__str__ (non-negative) [EQUAL]': its probe compares after .replace(', ', ' ')."""
import sys, datetime as dt, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
td = dt.timedelta
cases = [(td(hours=1),), (td(hours=1), td(hours=2)), (td(days=1, hours=2), td(days=3)), (td(0), td(seconds=1.5)),
         (td(days=2),), (td(hours=1), td(hours=2))]
same = 0
for c in cases:
    s1, s2 = str(v1t.TimeDeltaInterval(*c)), str(v2t.TimeDeltaInterval(*c))
    same += s1 == s2
    print(repr(s1), '|', repr(s2), '' if s1 == s2 else '<-- DIFFER')
u1 = v1t.TimeDeltaInterval(td(hours=1), td(hours=2)).union(v1t.TimeDeltaInterval(td(days=1, hours=3), td(days=1, hours=4)))
u2 = v2t.TimeDeltaInterval(td(hours=1), td(hours=2)).union(v2t.TimeDeltaInterval(td(days=1, hours=3), td(days=1, hours=4)))
print(repr(str(u1)), '|', repr(str(u2)), '' if str(u1) == str(u2) else '<-- DIFFER')
print('identical texts', same + (str(u1) == str(u2)), 'of', len(cases) + 1)
assert str(u1) != str(u1) + 'x'
