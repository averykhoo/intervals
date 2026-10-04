"""round(A, n), math.trunc/floor/ceil(A): v1 endpoint-wise vs v2 values attained vs an exact oracle.
checks per case: (1) every sampled x in A has python's f(x) inside v2's result ("what a user gets");
(2) v2's result is exactly the oracle's attained set (no extra values); (3) v1 holds every attained value
(soundness) and how much it adds."""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random, time
CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
from fractions import Fraction as F
from common import *

def preimage(name, k, n):
    """grid index k -> (lo, hi, lo_closed, hi_closed) of {x : f(x) = k * unit}"""
    if name == 'floor':
        return (F(k), F(k + 1), True, False)
    if name == 'ceil':
        return (F(k - 1), F(k), False, True)
    if name == 'trunc':
        return (F(k), F(k + 1), True, False) if k > 0 else (F(k - 1), F(k), False, True) if k < 0 else (F(-1), F(1), False, False)
    u = F(1) if n is None else F(10) ** -n
    g, h = k * u, u / 2
    return (g - h, g + h, k % 2 == 0, k % 2 == 0)

def meets(p, q):
    lo = max((p[0], not p[2]), (q[0], not q[2]))
    hi = min((p[1], p[3]), (q[1], q[3]))
    return lo[0] < hi[0] or (lo[0] == hi[0] and not lo[1] and hi[1])

def attained(name, n, pieces):
    unit = F(1) if n is None else F(10) ** -n
    out = set()
    for p in pieces:
        q = (F(p[0]), F(p[1]), p[2], p[3])
        a, b = math.floor(q[0] / unit) - 2, math.ceil(q[1] / unit) + 2
        for k in range(a, b + 1):
            if meets(preimage(name, k, n), q):
                out.add(k * unit)
    return out

def pyf(name, n, x):
    if name == 'round':
        return round(x, n)
    return getattr(math, name)(x)

def apply1(name, n, A):
    return round(A, n) if name == 'round' else getattr(math, name)(A)

rng = random.Random(2026)
def rand_set(kind):
    k = rng.randint(1, 2)
    if kind == 'float':
        vals = sorted(round(rng.uniform(-6, 6), rng.choice([1, 2, 3])) for _ in range(2 * k))
    else:
        vals = sorted(F(rng.randint(-60, 60), rng.choice([1, 2, 4, 10, 20, 100])) for _ in range(2 * k))
    ps = []
    for i in range(k):
        lo, hi = vals[2 * i], vals[2 * i + 1]
        if ps and lo <= ps[-1][1]:
            continue
        if lo == hi or rng.random() < 0.1:
            ps.append((lo, lo, True, True))
        else:
            ps.append((lo, hi, rng.random() < .5, rng.random() < .5))
    return ps

def samples(ps):
    xs = []
    for lo, hi, lc, hc in ps:
        if lc: xs.append(lo)
        if hc: xs.append(hi)
        for _ in range(25):
            t = F(rng.randint(1, 999), 1000)
            xs.append(lo + (hi - lo) * t)
        if isinstance(lo, float):
            xs += [x for x in (math.nextafter(lo, math.inf), math.nextafter(hi, -math.inf)) if lo < x < hi]
    return xs

def as_values(ps):
    return {F(p[0]) for p in ps if p[0] == p[1]}, [p for p in ps if p[0] != p[1]]

stats = {}
def tally(k):
    stats[k] = stats.get(k, 0) + 1
EX = {}
N = 0
for name, ns in (('floor', [None]), ('ceil', [None]), ('trunc', [None]), ('round', [None, 0, 1, 2, -1])):
    for n in ns:
        for kind in ('float', 'exact'):
            t0 = time.time()
            for _ in range(CASES):
                ps = rand_set(kind)
                N += 1
                A1, A2 = mk1(ps), mk2(ps)
                r2, e2 = run(lambda: apply1(name, n, A2))
                if name == 'round' and n is None:
                    r1, e1 = run(lambda: round(A1))       # v1 default n_digits=0
                else:
                    r1, e1 = run(lambda: apply1(name, n, A1))
                tag = f'{name}({n}) {kind}'
                if e2:
                    tally(f'{tag}: v2 raises {e2}'); continue
                # (1) what a user gets is in v2's result
                miss = [x for x in samples(ps) if pyf(name, n, x) not in r2]
                if miss:
                    tally(f'{tag}: v2 MISSES a python value'); EX.setdefault(f'{tag} miss', (ps, miss[:3], str(r2)))
                # (2) v2 is exactly the attained set (exact inputs only: float rounding of 0.1 grids makes values floats)
                truth = attained(name, n, [(F(p[0]), F(p[1]), p[2], p[3]) for p in ps])
                pts, rest = as_values(pieces2(r2))
                if len(truth) > 1000:
                    hull_ok = all((t in r2) if kind == "exact" else (float(t) in r2) for t in truth)
                    tally(f'{tag}: >1000 values, v2 hull (HullWarning) {"holds" if hull_ok else "MISSES"} them all')
                elif kind == 'exact':
                    if rest or pts != truth:
                        tally(f'{tag}: v2 != oracle'); EX.setdefault(f'{tag} v2!=oracle', (ps, sorted(truth), str(r2)))
                    else:
                        tally(f'{tag}: v2 == oracle')
                else:
                    near = all(any(abs(p - t) <= F(1, 10 ** 12) for t in truth) for p in pts) and \
                           all(any(abs(p - t) <= F(1, 10 ** 12) for p in pts) for t in truth) and not rest
                    tally(f'{tag}: v2 {"==" if near else "!="} oracle (to float rounding)')
                    if not near:
                        EX.setdefault(f'{tag} v2!=oracle', (ps, sorted(truth), str(r2)))
                # (3) v1
                if e1:
                    tally(f'{tag}: v1 raises {e1.split(":")[0]}'); EX.setdefault(f'{tag} v1 raises', (ps, e1)); continue
                if not well_formed1(r1):
                    tally(f'{tag}: v1 MALFORMED'); EX.setdefault(f'{tag} v1 malformed', (ps, r1.endpoints)); continue
                p1 = pieces1(r1)
                lost = [t for t in truth if not (contains_pieces(p1, t) or contains_pieces(p1, float(t)))]
                if lost:
                    tally(f'{tag}: v1 LOSES an attained value'); EX.setdefault(f'{tag} v1 loses', (ps, s1(r1), sorted(map(str, lost))))
                elif any(lo != hi for lo, hi, _, _ in p1):
                    tally(f'{tag}: v1 sound, adds non-attained values (hull)')
                else:
                    tally(f'{tag}: v1 sound and exact')
print('cases', N)
for k, v in sorted(stats.items()):
    print(f'{v:5d}  {k}')
print()
for k, v in EX.items():
    print(k, '\n   ', v)
# sabotage: a wrong oracle (floor where ceil is asked) must be caught
assert attained('floor', None, [(F(1, 2), F(1, 2), True, True)]) != {F(1)}
assert F(1) not in mk2([(F(1, 2), F(1, 2), True, True)]).floor()
print('sabotage caught')
