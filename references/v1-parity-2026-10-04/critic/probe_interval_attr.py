"""re-check datetime row '.interval attribute / building from raw seconds [EQUIVALENT_RENAMED]': its probe adds a
hard-coded 28800 (this machine's UTC+8) to v1's .interval. v1's .interval is POSIX (UTC) seconds of the local reading;
which v2 spelling gives the same number without knowing the offset?"""
import sys, datetime as dt, random, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
rng = random.Random(5)
ok_naive = ok_aware = n = 0
for _ in range(300):
    t = dt.datetime(1990, 1, 1) + dt.timedelta(days=rng.randrange(15000), seconds=rng.randrange(86400), microseconds=rng.randrange(1, 10**6))
    v1s = F(v1t.DateTimeInterval(t).interval.endpoints[0][0])
    v2_naive = v2t.DateTimeInterval(t).seconds.inf
    v2_aware = v2t.DateTimeInterval(t.astimezone()).seconds.inf      # local zone attached, as v1's timestamp() assumes
    n += 1; ok_naive += abs(v2_naive - v1s) < F(1, 10**5); ok_aware += abs(v2_aware - v1s) < F(1, 10**5)
print('n', n, ' v2 naive .seconds == v1 .interval:', ok_naive, ' v2 DTI(t.astimezone()).seconds == v1 .interval:', ok_aware)
assert ok_naive == 0 and ok_aware == n
