from common import *
import pickle, copy
def outcome(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__)
H = dt.timedelta(hours=1)
A1, A2 = T1(H, 2*H), T2(H, 2*H)
DT = dt.datetime(2024, 1, 1)
for lab, f1, f2 in [
    ('bool(empty)', lambda: bool(T1()), lambda: bool(T2())),
    ('bool(non-empty)', lambda: bool(A1), lambda: bool(A2)),
    ('len', lambda: len(A1), lambda: len(A2)),
    ('iter', lambda: list(A1), lambda: len(list(A2))),
    ('pickle round trip', lambda: canon1(pickle.loads(pickle.dumps(A1))) == canon1(A1), lambda: pickle.loads(pickle.dumps(A2)) == A2),
    ('copy.copy', lambda: canon1(copy.copy(A1)) == canon1(A1), lambda: copy.copy(A2) == A2),
    ('deepcopy', lambda: canon1(copy.deepcopy(A1)) == canon1(A1), lambda: copy.deepcopy(A2) == A2),
    ('T < datetime', lambda: A1 < DT, lambda: A2 < DT),
    ('T == datetime', lambda: A1 == DT, lambda: A2 == DT),
    ('T.union(datetime)', lambda: A1.union(DT), lambda: A2.union(DT)),
    ('T.union(D)', lambda: A1.union(D1(DT)), lambda: A2.union(D2(DT))),
    ('datetime in T', lambda: DT in A1, lambda: DT in A2),
    ('T.overlaps(5)', lambda: A1.overlaps(5), lambda: A2.overlaps(5)),
    ('T.issubset(MultiInterval)', lambda: A1.issubset(v1m.MultiInterval(0, 10000)), lambda: A2.issubset(M2(0, 10000))),
    ('T < pd.Timedelta', lambda: A1 < pd.Timedelta(hours=3), lambda: (A2 < pd.Timedelta(hours=3)).certainly),
    ('T == pd.Timedelta', lambda: T1(H) == pd.Timedelta(hours=1), lambda: T2(H) == pd.Timedelta(hours=1)),
    ('POS_INF bound', lambda: 'n/a in v1', lambda: str(T2(H, POS_INF))),
    ('getitem slice', lambda: A1[H:2*H], lambda: str(A2[H:H*3/2])),
]:
    print(f'{lab}: v1 {outcome(f1)} | v2 {outcome(f2)}')
