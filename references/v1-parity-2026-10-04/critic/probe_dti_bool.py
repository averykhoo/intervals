import sys, datetime as dt, pickle, copy
sys.path[:0] = ['.', 'archive/v1']
import time_interval as v1t
import intervals.time_interval as v2t
t = dt.datetime(2024, 5, 1, 10, 30, 15, 5)
for name, a1, a2 in [('empty', v1t.DateTimeInterval(), v2t.DateTimeInterval()),
                     ('point', v1t.DateTimeInterval(t), v2t.DateTimeInterval(t))]:
    print(name, 'bool v1', bool(a1), 'v2', bool(a2))
    for f in ('pickle', 'deepcopy'):
        try:
            r1 = (pickle.loads(pickle.dumps(a1)) if f == 'pickle' else copy.deepcopy(a1)) == a1
        except Exception as e: r1 = repr(e)
        try:
            r2 = (pickle.loads(pickle.dumps(a2)) if f == 'pickle' else copy.deepcopy(a2)) == a2
        except Exception as e: r2 = repr(e)
        print(' ', f, 'roundtrip==: v1', r1, 'v2', r2)
# sabotage self-check: the comparison must be able to fail
assert bool(v1t.DateTimeInterval()) != bool(v2t.DateTimeInterval()), 'expected the truthiness difference'
