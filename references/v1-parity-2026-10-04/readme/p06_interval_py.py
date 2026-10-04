# README: "interval.Interval: the usual contiguous intervals ... usable set functions ... most numeric operations;
# interval.MultipleInterval: operations on groups of Interval objects" -- dropped (archived); the workaround is MultiInterval
from common import *
import interval as v1i
warnings.simplefilter('ignore')
def mi_of(xs):  # list of v1i.Interval -> v2
    return MI.from_pieces((i.start, i.end, not i.start_open, i.end_closed) for i in xs)
def rand_iv(rng):
    a, b = sorted(Fraction(rng.randint(-12, 12), 2) for _ in range(2))
    if a == b: return v1i.Interval(a, False, a, True)
    return v1i.Interval(a, rng.random() < .5, b, rng.random() < .5)
def brute_member(y, op, A, B):
    return None
rng = random.Random(5)
st = dict(union=0, intersection=0, difference=0, symmetric_difference=0); err = {k: 0 for k in st}; ex = []
N = 300
for _ in range(N):
    xs = [rand_iv(rng) for _ in range(rng.randint(1, 3))]; ys = [rand_iv(rng) for _ in range(rng.randint(1, 3))]
    A1, B1 = v1i.MultipleInterval(*xs), v1i.MultipleInterval(*ys); A2, B2 = mi_of(xs), mi_of(ys)
    truth = {'union': lambda p: p in A2 or p in B2, 'intersection': lambda p: p in A2 and p in B2,
             'difference': lambda p: p in A2 and p not in B2, 'symmetric_difference': lambda p: (p in A2) != (p in B2)}
    for k in st:
        try:
            r1 = mi_of(getattr(A1.copy(), k)(B1).intervals)
        except Exception as e:
            err[k] += 1; ex.append((k, str(A2), str(B2), repr(e)[:60])); continue
        r2 = getattr(A2, k)(B2)
        pts = probes_of(A2, B2)
        ok2 = all((p in r2) == truth[k](p) for p in pts)
        st[k] += (r1 == r2) and ok2
        if r1 != r2: ex.append((k, str(A2), str(B2), str(r1), str(r2), ok2))
print(f'MultipleInterval set ops, N={N}: agree {st}, v1 raised {err}')
for e in ex[:5]: print('  ', e)
# numeric ops on Interval vs v2 single piece, exact corners
ok = {'+': 0, '-': 0, '*': 0}; bad = []
for _ in range(N):
    x, y = rand_iv(rng), rand_iv(rng)
    for s, f in (('+', lambda a, b: a + b), ('-', lambda a, b: a - b), ('*', lambda a, b: a * b)):
        try: r1 = mi_of([f(x, y)])
        except Exception as e: bad.append((s, x, y, repr(e)[:50])); continue
        r2 = f(mi_of([x]), mi_of([y]))
        if r1 == r2: ok[s] += 1
        else: bad.append((s, str(mi_of([x])), str(mi_of([y])), str(r1), str(r2)))
print(f'Interval + - * vs v2, N={N}: agree {ok}; disagreements {len(bad)}')
for b in bad[:5]: print('  ', b)
# sabotage
assert mi_of([v1i.Interval(0, False, 1, False)]) != MI(0, 1)
