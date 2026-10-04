from common import *
import numpy as np
def outcome(f):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__, str(e)[:90])
H = dt.timedelta(hours=1)
A1, A2 = T1(H, 2 * H), T2(H, 2 * H)
print('T*0.1 v1 inf', (A1 * 0.1).infimum, '| v2 T*Fraction(1,10) inf', (A2 * Fraction(1, 10)).inf, '| v2 inf_seconds of T*0.1', (A2 * 0.1).inf_seconds)
print('T*np.timedelta64(3,ns): v1 str', outcome(lambda: str(A1 * np.timedelta64(3, 'ns'))), 'endpoints', (A1 * np.timedelta64(3, 'ns')).interval.endpoints)
print('T/(-3/4): v1 sup', (A1 / Fraction(-3, 4)).supremum, '| v2 sup', outcome(lambda: (A2 / Fraction(-3, 4)).sup), 'sup_seconds', (A2 / Fraction(-3, 4)).sup_seconds)
print('T1(3.5s,4s open)*0:', outcome(lambda: T1(td(Fraction(7,2)), td(4), start_closed=False, end_closed=False) * 0), '| v2', str(T2(td(Fraction(7,2)), td(4), start_closed=False, end_closed=False) * 0))
print('exact: {x*0 : x in (3.5, 4)} = {0}')
print('T - D(interval): v1', outcome(lambda: type(A1 - D1(dt.datetime(2024,1,1,0,0,0,1), dt.datetime(2024,1,2,0,0,0,1))).__name__), '| v2', outcome(lambda: A2 - D2(dt.datetime(2024,1,1), dt.datetime(2024,1,2))))
print('T - Timestamp: v1', outcome(lambda: type(A1 - pd.Timestamp(2024,1,1)).__name__), '| v2', outcome(lambda: A2 - pd.Timestamp(2024,1,1)))
print('T - date: v1', outcome(lambda: type(A1 - dt.date(2024,1,1)).__name__), '| v2', outcome(lambda: A2 - dt.date(2024,1,1)))
r = A1 - dt.datetime(2024, 1, 1)
print('v1 T - dt raw seconds', r.interval.endpoints, '(= duration seconds minus a LOCAL unix timestamp)')
print('pd.Timedelta % T workaround: v2 T(x) % A', T2(pd.Timedelta(hours=1)) % T2(H, 2*H), '| v1 has no %:', outcome(lambda: A1 % A1))
print('T // 2: v1', outcome(lambda: A1 // 2), '| v2', outcome(lambda: A2 // 2))
print('-T: v1', outcome(lambda: -A1), '| v2', -A2)
print('Timestamp + T type: v1', type(pd.Timestamp(2024,1,1) + A1).__name__, '| v2', type(pd.Timestamp(2024,1,1) + A2).__name__)
print('pd.Timedelta + T type: v1', type(pd.Timedelta(hours=1) + A1).__name__, '| v2', type(pd.Timedelta(hours=1) + A2).__name__)
print('T * Fraction(1,3) inf: v1', (A1 * Fraction(1,3)).infimum, '| v2', outcome(lambda: (A2 * Fraction(1, 3)).inf))
print('np.float64(2) * T: v1', type(np.float64(2) * A1).__name__, str(np.float64(2) * A1), '| v2', type(np.float64(2) * A2).__name__, str(np.float64(2) * A2))
