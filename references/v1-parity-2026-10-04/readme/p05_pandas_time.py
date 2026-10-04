# README: "DateTimeInterval and TimeDeltaInterval ... behaves somewhat like datetime.datetime and datetime.timedelta
# merged with MultiInterval ... also accepts pandas.Timestamp and pandas.Timedelta"
from common import *
import datetime as dt
import pandas as pd
import time_interval as v1t
from intervals import DateTimeInterval as DTI, TimeDeltaInterval as TDI
warnings.simplefilter('ignore')
EPOCH = dt.datetime(1970, 1, 1)
def run(f):
    try:
        r = f(); return 'NotImplemented' if r is NotImplemented else r
    except Exception as e: return f'{type(e).__name__}: {str(e)[:70]}'

def v1_dt_pieces(a):
    """v1 DateTimeInterval -> list of (lo_dt, hi_dt, lc, hc) as naive local datetimes (v1 used timestamp())"""
    e = a.interval.endpoints; out = []
    for i in range(0, len(e), 2):
        out.append((dt.datetime.fromtimestamp(e[i][0]), dt.datetime.fromtimestamp(e[i+1][0]), e[i][1] == 0, e[i+1][1] == 0))
    return out
def v2_dt_pieces(b):
    return [(p.inf, p.sup, p.inf_closed, p.sup_closed) for p in b.pieces]
def v1_td_pieces(a):
    e = a.interval.endpoints
    return [(dt.timedelta(seconds=e[i][0]), dt.timedelta(seconds=e[i+1][0]), e[i][1] == 0, e[i+1][1] == 0) for i in range(0, len(e), 2)]
def v2_td_pieces(b):
    return [(p.inf, p.sup, p.inf_closed, p.sup_closed) for p in b.pieces]
def close(p, q):  # microsecond-rounded float of v1 vs exact v2: allow 2 us
    if len(p) != len(q): return False
    return all(abs(x[0]-y[0]) <= dt.timedelta(microseconds=2) and abs(x[1]-y[1]) <= dt.timedelta(microseconds=2) and x[2:] == y[2:] for x, y in zip(p, q))

rng = random.Random(11)
def rts():  # random pandas Timestamp with non-zero microsecond (v1 snaps an end with zero microseconds)
    return pd.Timestamp(2024, 3, rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59), rng.randint(1, 999999))
def rtd():
    return pd.Timedelta(seconds=rng.randint(0, 100000), microseconds=rng.randint(1, 999999))
stats = dict(ctor=0, contains=0, add=0, sub_dt=0, union=0, td_ctor=0, td_mul=0)
bad = []
N = 200
for _ in range(N):
    a, b = sorted([rts(), rts()]); c, d = sorted([rts(), rts()]); t = rts(); td = rtd()
    lc, hc = rng.random() < .5, rng.random() < .5
    A1 = v1t.DateTimeInterval(a, b, start_closed=lc, end_closed=hc); A2 = DTI(a, b, start_closed=lc, end_closed=hc)
    B1 = v1t.DateTimeInterval(c, d); B2 = DTI(c, d)
    stats['ctor'] += close(v1_dt_pieces(A1), v2_dt_pieces(A2)) or bad.append(('ctor', a, b))
    stats['contains'] += ((t in A1) == (t in A2)) or bad.append(('in', a, b, t))
    stats['add'] += close(v1_dt_pieces(A1 + td), v2_dt_pieces(A2 + td)) or bad.append(('add', a, b, td))
    stats['sub_dt'] += close(v1_td_pieces(A1 - t), v2_td_pieces(A2 - t)) or bad.append(('sub', a, b, t))
    stats['union'] += close(v1_dt_pieces(A1.union(B1)), v2_dt_pieces(A2 | B2)) or bad.append(('union', a, b, c, d))
    t1, t2 = sorted([rtd(), rtd()])
    T1 = v1t.TimeDeltaInterval(t1, t2); T2 = TDI(t1, t2)
    stats['td_ctor'] += close(v1_td_pieces(T1), v2_td_pieces(T2)) or bad.append(('tdctor', t1, t2))
    stats['td_mul'] += close(v1_td_pieces(T1 * 3), v2_td_pieces(T2 * 3)) or bad.append(('tdmul', t1, t2))
print(f'pandas sweep N={N}:', stats); print('  bad:', bad[:4])
# sabotage: close() must catch a 1 ms shift and a flag flip
p = v2_dt_pieces(DTI(pd.Timestamp('2024-01-01 00:00:00.5'), pd.Timestamp('2024-01-02 00:00:00.5')))
assert not close(p, [(p[0][0] + dt.timedelta(milliseconds=1),) + p[0][1:]]) and not close(p, [p[0][:2] + (False, True)])

print('--- hand cases (v1 | v2)')
ts0, ts1 = pd.Timestamp('2024-01-01 09:00'), pd.Timestamp('2024-01-01 17:00')
for name, f1, f2 in [
  ('DTI(09:00, 17:00) sup', lambda: v1t.DateTimeInterval(ts0, ts1).supremum, lambda: DTI(ts0, ts1).sup),
  ('DTI(NaT)', lambda: str(v1t.DateTimeInterval(pd.NaT)), lambda: DTI(pd.NaT)),
  ('DTI(ts, NaT)', lambda: str(v1t.DateTimeInterval(ts0, pd.NaT)), lambda: DTI(ts0, pd.NaT)),
  ('ns Timestamp inf', lambda: v1t.DateTimeInterval(pd.Timestamp('2024-01-01 00:00:00.000000001')).interval.endpoints[0][0] % 1, lambda: DTI(pd.Timestamp('2024-01-01 00:00:00.000000001')).inf_seconds % 1),
  ('TDI(pd.Timedelta(1ns))', lambda: v1t.TimeDeltaInterval(pd.Timedelta(1, 'ns')).interval.endpoints, lambda: TDI(pd.Timedelta(1, 'ns')).seconds),
  ('pd.Timedelta + DTI', lambda: str(pd.Timedelta(hours=1) + v1t.DateTimeInterval(pd.Timestamp('2024-01-01 09:00:00.5'))), lambda: str(pd.Timedelta(hours=1) + DTI(pd.Timestamp('2024-01-01 09:00:00.5')))),
  ('pd.Timestamp - DTI', lambda: str(pd.Timestamp('2024-01-02') - v1t.DateTimeInterval(pd.Timestamp('2024-01-01 09:00:00.5'))), lambda: str(pd.Timestamp('2024-01-02') - DTI(pd.Timestamp('2024-01-01 09:00:00.5')))),
  ('ts in DTI', lambda: pd.Timestamp('2024-01-01 12:00') in v1t.DateTimeInterval(ts0, ts1), lambda: pd.Timestamp('2024-01-01 12:00') in DTI(ts0, ts1)),
  ('TDI / pd.Timedelta', lambda: v1t.TimeDeltaInterval(pd.Timedelta(hours=1), pd.Timedelta(hours=3)) / pd.Timedelta(minutes=30), lambda: TDI(pd.Timedelta(hours=1), pd.Timedelta(hours=3)) / pd.Timedelta(minutes=30)),
  ('to_pandas', lambda: 'n/a', lambda: DTI(ts0, ts1).to_pandas()),
]:
    print(f'{name:24s} v1: {str(run(f1)):55s} v2: {run(f2)}')
