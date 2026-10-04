"""iteration and len. v1 MultiInterval has no __iter__/__len__, but __getitem__ takes numbers, so the
legacy sequence protocol runs A[0], A[1], ... forever (IndexError is never raised). v2: pieces."""
import sys, itertools, random, warnings, datetime as dt
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import time_interval as v1t
import intervals as v2
import intervals.time_interval as v2t

def tryit(f):
    try: return f()
    except Exception as e: return f'{type(e).__name__}: {e}'

A1, A2 = v1.MultiInterval(1, 2), v2.MultiInterval(1, 2)
print('v1 hasattr __iter__/__len__:', hasattr(A1, '__iter__'), hasattr(A1, '__len__'))
print('v1 iter(A) first 5:', [str(p) for p in itertools.islice(iter(A1), 5)])
# never ends: draw 20000 items and show it still has not raised StopIteration
it = iter(v1.MultiInterval(1, 2)); k = 0
for _ in itertools.islice(it, 20000): k += 1
print('v1 iter(A) still going after', k, 'items; next():', str(next(it)))
print('v1 iter(empty) first 3:', [str(p) for p in itertools.islice(iter(v1.MultiInterval()), 3)])
print('v1 len(A):', tryit(lambda: len(A1)))
print('v1 x in A uses __contains__ (not iteration):', 1.5 in A1, 3 in A1)
print('v1 contiguous_intervals (v1 piece accessor):', [str(p) for p in v1.MultiInterval.merge(A1, v1.MultiInterval(5)).contiguous_intervals])
print('v2 list(A):', [str(p) for p in A2], 'len', len(A2), 'list(empty):', list(v2.MultiInterval()), 'len(empty)', len(v2.MultiInterval()))

# random sweep: v2 iteration == v1 contiguous_intervals (the thing v1 users had for "the pieces")
rng = random.Random(4)
bad = 0; n = 0
for _ in range(300):
    m1, m2 = v1.MultiInterval(), v2.MultiInterval()
    for _ in range(rng.randint(0, 4)):
        s = rng.randint(-6, 6); t = s + rng.randint(0, 3)
        sc, tc = (True, True) if s == t else (rng.random() < .5, rng.random() < .5)
        m1 = m1.union(v1.MultiInterval(s, t, start_closed=sc, end_closed=tc) if s != t else v1.MultiInterval(s))
        m2 = m2 | (v2.MultiInterval(s, t, start_closed=sc, end_closed=tc) if s != t else v2.MultiInterval(s))
    p1 = [(p.infimum, p.infimum_is_closed, p.supremum, p.supremum_is_closed) for p in m1.contiguous_intervals]
    p2 = [(p.inf, p.inf_closed, p.sup, p.sup_closed) for p in m2]
    n += 1
    if p1 != p2 or len(m2) != len(p1) or tuple(m2) != m2.pieces: bad += 1; print('DIFF', p1, p2)
print('sweep', n, 'v1 contiguous_intervals != v2 iter:', bad)

# time layer
d0, d1 = dt.datetime(2024, 1, 1, 6), dt.datetime(2024, 1, 1, 7, 30, 1, 5)
D1, D2 = v1t.DateTimeInterval(d0, d1), v2t.DateTimeInterval(d0, d1)
print('v1 DTI list():', tryit(lambda: list(D1)), '| v1 DTI len():', tryit(lambda: len(D1)))
print('v2 DTI list():', [str(p) for p in D2], 'len', len(D2))
T1, T2 = v1t.TimeDeltaInterval(dt.timedelta(0), dt.timedelta(1)), v2t.TimeDeltaInterval(dt.timedelta(0), dt.timedelta(1))
print('v1 TDI list():', tryit(lambda: list(T1)), '| v1 TDI len():', tryit(lambda: len(T1)))
print('v2 TDI list():', [str(p) for p in T2], 'len', len(T2))
print('v1 DTI contiguous_intervals:', [str(p) for p in D1.contiguous_intervals])

# sabotage: v1's iteration must not be mistaken for the pieces
assert [str(p) for p in itertools.islice(iter(A1), 1)] != [str(p) for p in A2], 'sabotage: v1 iter would look like pieces'
assert bad == 0
print('sabotage caught')
