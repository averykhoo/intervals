"""TimeDeltaInterval: (1) pickle, copy.copy, copy.deepcopy round trips (empty, point, multi-piece, negative), checked by
exact structure AND ==, with a shallow-copy independence check (v1 has in-place methods; v2 is immutable);
(2) T * r, r * T, T / r for r in int, float, np.float64, np.int64, np.float32, Fraction (and bool), against an exact
oracle on the seconds: v2 exact; v1 within float rounding; read-outs (inf/sup, v1 infimum/supremum) compared where the
exact product is a whole microsecond. `sab`: the oracle for one case is off by 1 us, must be caught."""
import sys, random, warnings, datetime as dt, pickle, copy, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import numpy as np
import time_interval as v1t
import intervals.time_interval as v2t
SAB = len(sys.argv) > 1
td = dt.timedelta
def fx(v): return F(float(v)) if isinstance(v, np.floating) else F(v)     # v1 can hold np.float32 ends
def st1(m): e = m.endpoints; return [(fx(e[i][0]), e[i][1] == 0, fx(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]
def st2(m): return [(F(p.inf), bool(p.inf_closed), F(p.sup), bool(p.sup_closed)) for p in m]
rng = random.Random(1039)
def rset():
    ds = sorted({td(seconds=rng.randrange(-10**6, 10**6), microseconds=rng.randrange(10**6)) for _ in range(2 * rng.randint(1, 3))})
    ps = []
    for i in range(0, len(ds) - 1, 2):
        lo, hi = ds[i], ds[i + 1]
        if rng.random() < .2: hi = lo
        ps.append((lo, hi, True if lo == hi else rng.random() < .5, True if lo == hi else rng.random() < .5))
    return ps
def build(ps):
    a1, a2 = v1t.TimeDeltaInterval(), v2t.TimeDeltaInterval()
    for lo, hi, lc, hc in ps:
        a1 = a1.union(v1t.TimeDeltaInterval(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else v1t.TimeDeltaInterval(lo))
        a2 = a2 | (v2t.TimeDeltaInterval(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else v2t.TimeDeltaInterval(lo))
    return a1, a2
# (1) pickle / copy / deepcopy
c = dict(n=0, v1_pickle=0, v1_copy=0, v1_deep=0, v2_pickle=0, v2_copy=0, v2_deep=0, v1_shallow_shares=0, v2_type_kept=0)
sets = [[], [(td(hours=1), td(hours=1), True, True)]] + [rset() for _ in range(200)]
for ps in sets:
    a1, a2 = build(ps); c['n'] += 1
    for name, f in (('pickle', lambda x: pickle.loads(pickle.dumps(x))), ('copy', copy.copy), ('deep', copy.deepcopy)):
        b1, b2 = f(a1), f(a2)
        c['v1_' + name] += bool(b1 == a1) and st1(b1.interval) == st1(a1.interval) and b1 is not a1
        c['v2_' + name] += (b2 == a2) is True and st2(b2.seconds) == st2(a2.seconds) and type(b2) is type(a2)
    s1 = copy.copy(a1); before = st1(a1.interval); s1.update(v1t.TimeDeltaInterval(td(days=999)))
    c['v1_shallow_shares'] += st1(a1.interval) != before       # mutating the shallow copy changed the original?
    c['v2_type_kept'] += type(pickle.loads(pickle.dumps(a2))).__name__ == 'TimeDeltaInterval'
print(c)
other = build([(td(0), td(1), True, True)])[1]
assert (pickle.loads(pickle.dumps(other)) == build([(td(0), td(2), True, True)])[1]) is False   # == can fail
assert all(c[k] == c['n'] for k in ('v1_pickle', 'v1_copy', 'v1_deep', 'v2_pickle', 'v2_copy', 'v2_deep', 'v2_type_kept'))
# (2) scaling
def factors():
    q = rng.choice([F(3, 2), F(-1, 4), F(2), F(0), F(1, 10), F(7, 3), F(-5)])
    return {'int': int(rng.randint(-4, 4)), 'float': float(q), 'np.float64': np.float64(float(q)), 'np.int64': np.int64(rng.randint(-4, 4)),
            'np.float32': np.float32(float(q)), 'Fraction': q}
def exact(x): return F(float(x)) if isinstance(x, (float, np.floating)) else F(int(x)) if isinstance(x, (int, np.integer)) else F(x)
def oracle(ps, r, op):
    out = []
    for lo, hi, lc, hc in ps:
        a, b = F(lo // td(microseconds=1), 10**6), F(hi // td(microseconds=1), 10**6)
        u, w = (a * r, b * r) if op != '/' else (a / r, b / r)
        if r == 0: out.append((F(0), True, F(0), True)); continue
        out.append((u, lc, w, hc) if u <= w else (w, hc, u, lc))
    out.sort(); return out
s = {}; ex = []
for it in range(250):
    ps = rset()
    if not ps: continue
    a1, a2 = build(ps)
    for kind, r in factors().items():
        R = exact(r)
        for op in ('T*r', 'r*T', 'T/r'):
            if op == 'T/r' and R == 0: continue
            want = oracle(ps, R, '/' if op == 'T/r' else '*')
            if R == 0: want = [(F(0), True, F(0), True)]
            if SAB and it == 3 and kind == 'int': want = [(want[0][0] + F(1, 10**6),) + want[0][1:]] + want[1:]
            f = {'T*r': lambda x: x * r, 'r*T': lambda x: r * x, 'T/r': lambda x: x / r}[op]
            try:
                g1 = st1(f(a1).interval)
                if not isinstance(f(a1), v1t.TimeDeltaInterval): g1 = 'type ' + type(f(a1)).__name__
            except Exception as e: g1 = 'RAISES ' + type(e).__name__
            try: r2 = f(a2); g2 = st2(r2.seconds) if isinstance(r2, v2t.TimeDeltaInterval) else 'type ' + type(r2).__name__
            except Exception as e: g2 = 'RAISES ' + type(e).__name__
            k = (kind, op); t = s.setdefault(k, {'n': 0, 'v2 exact': 0, 'v1 close': 0, 'readout same': 0, 'readout n': 0, 'v2 readout raises': 0})
            t['n'] += 1; t['v2 exact'] += g2 == want
            if not isinstance(g1, str) and len(g1) == len(want) and all(abs(x[0] - y[0]) <= abs(y[0]) * F(1, 2**50) + F(1, 10**9) and abs(x[2] - y[2]) <= abs(y[2]) * F(1, 2**50) + F(1, 10**9) and x[1] == y[1] and x[3] == y[3] for x, y in zip(g1, want)):
                t['v1 close'] += 1
            elif len(ex) < 4: ex.append((kind, op, ps[:1], r, g1 if isinstance(g1, str) else g1[:1], want[:1]))
            if g2 == want and want[0][0] * 10**6 == int(want[0][0] * 10**6):
                t['readout n'] += 1
                try: t['readout same'] += f(a1).infimum == f(a2).inf
                except Exception: pass
            elif g2 == want:
                try: f(a2).inf
                except ValueError: t['v2 readout raises'] += 1
bad = 0
for k, t in s.items():
    print(k, t); bad += t['v2 exact'] != t['n']
for e in ex: print('  v1 off:', e)
for op in ('T*r', 'T/r'):
    x1 = v1t.TimeDeltaInterval(td(seconds=1000, microseconds=1)); x1 = x1 * np.float32(3.0) if op == 'T*r' else x1 / np.float32(3.0)
    x2 = v2t.TimeDeltaInterval(td(seconds=1000, microseconds=1)); x2 = x2 * np.float32(3.0) if op == 'T*r' else x2 / np.float32(3.0)
    e1 = x1.interval.endpoints[0][0]
    try: ro1 = str(x1.infimum)
    except Exception as e: ro1 = 'RAISES ' + type(e).__name__ + ': ' + str(e)[:60]
    try: ro2 = str(x2.inf)
    except Exception as e: ro2 = 'RAISES ' + type(e).__name__
    print(f'[1000.000001 s] {op} np.float32(3): v1 end {e1!r} ({type(e1).__name__}), read-out {ro1}; v2 end {x2.inf_seconds}, read-out {ro2}')
print('bool factor: v1', end=' ')
try: print((v1t.TimeDeltaInterval(td(1)) * True).infimum, end=' ')
except Exception as e: print('RAISES', type(e).__name__, end=' ')
try: print('v2', (v2t.TimeDeltaInterval(td(1)) * True).inf)
except Exception as e: print('v2 RAISES', type(e).__name__, str(e)[:80])
assert bad == 0
