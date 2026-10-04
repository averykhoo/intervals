# README: "the __pow__ and __rpow__ operations are not closed ... negative number raised to fractional powers
# result in complex numbers ... if modulo is specified, the same problems ... special cases are enumerable"
from common import *
warnings.simplefilter('ignore')

def run(f):
    try:
        r = f()
        if r is NotImplemented: return 'NotImplemented'
        return r
    except Exception as e:
        return f'{type(e).__name__}: {str(e)[:60]}'

def exp_pos_int(ps, n):
    out = []
    for lo, hi, lc, hc in ps:
        if n == 0: out.append((1, 1, True, True))
        elif n > 0: out.append((lo ** n, hi ** n, lc, hc))
        else:
            top = math.inf if lo == 0 else lo ** n
            out.append((hi ** n, top, hc, lc if lo != 0 else lc))
    return MI.from_pieces(out)

rng = random.Random(7)
n = agree = v1w = v2w = 0; ex = []
for _ in range(400):
    ps = []
    for _ in range(rng.randint(1, 2)):
        a, b = sorted(Fraction(rng.randint(1, 16), 4) for _ in range(2))
        ps.append((a, b, True, True) if a == b else (a, b, rng.random() < .5, rng.random() < .5))
    if rng.random() < .2:  # open at 0
        ps.append((0, Fraction(1, 8), False, rng.random() < .5))
    k = rng.randint(-3, 3)
    a1 = V1()
    for lo, hi, lc, hc in ps: a1.update(V1(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else V1(lo))
    a2 = MI.from_pieces(ps)
    r1, r2, want = run(lambda: a1 ** k), a2 ** k, exp_pos_int(ps, k)
    n += 1
    if isinstance(r1, V1): r1 = conv(r1)
    ok1, ok2 = (r1 == want), (r2 == want)
    agree += (r1 == r2); v1w += not ok1; v2w += not ok2
    if not ok1 or not ok2: ex.append((str(a2), k, str(r1), str(r2), str(want)))
print(f'positive base ** int: cases {n} agree {agree} v1_wrong {v1w} v2_wrong {v2w}')
for e in ex[:6]: print('  ', e)
# sabotage: a wrong expectation is caught
assert exp_pos_int([(1, 2, True, False)], 2) != MI(1, 4)

print('--- hand cases: v1 | v2 | exact')
cases = [
 ('(1,2] ** [0,1]', lambda: V1(1, 2, start_closed=False) ** V1(0, 1), lambda: MI(1, 2, start_closed=False) ** MI(0, 1), '[1, 2]: y=0 gives 1 for every x'),
 ('[2,3] ** [1,2]', lambda: V1(2, 3) ** V1(1, 2), lambda: MI(2, 3) ** MI(1, 2), '[2, 9]'),
 ('[1/2,2] ** [-1,2]', lambda: V1(Fraction(1,2), 2) ** V1(-1, 2), lambda: MI(Fraction(1,2), 2) ** MI(-1, 2), '[1/4, 4]'),
 ('[1/2,2) ** (1,2)', lambda: V1(Fraction(1,2), 2, end_closed=False) ** V1(1, 2, start_closed=False, end_closed=False), lambda: MI(Fraction(1,2), 2, end_closed=False) ** MI(1, 2, start_closed=False, end_closed=False), '(1/4, 4): 1/4 needs x=1/2,y=2 (y open)'),
 ('[2,4] ** 0.5', lambda: V1(2, 4) ** 0.5, lambda: MI(2, 4) ** 0.5, '[sqrt2, 2]'),
 ('[0] ** 2', lambda: V1(0) ** 2, lambda: MI(0) ** 2, '[0]'),
 ('[0] ** 0', lambda: V1(0) ** 0, lambda: MI(0) ** 0, 'python 0**0 = 1'),
 ('[0] ** -1', lambda: V1(0) ** -1, lambda: MI(0) ** -1, 'python raises; no value'),
 ('[0] ** [0,1]', lambda: V1(0) ** V1(0, 1), lambda: MI(0) ** MI(0, 1), '{0,1} with python 0**0=1; 1788 pow: [0]'),
 ('[0] ** [-1,1]', lambda: V1(0) ** V1(-1, 1), lambda: MI(0) ** MI(-1, 1), '0**neg: no value'),
 ('[-1,2] ** 2', lambda: V1(-1, 2) ** 2, lambda: MI(-1, 2) ** 2, '[0, 4]'),
 ('[-3,-1] ** 2', lambda: V1(-3, -1) ** 2, lambda: MI(-3, -1) ** 2, '[1, 9]'),
 ('[-3,-1] ** 3', lambda: V1(-3, -1) ** 3, lambda: MI(-3, -1) ** 3, '[-27, -1]'),
 ('[-8,-1] ** (1/3)', lambda: V1(-8, -1) ** (1/3), lambda: MI(-8, -1) ** (1/3), 'real cube root exists; python -8**(1/3) is complex'),
 ('[0,2] ** [1,2]', lambda: V1(0, 2) ** V1(1, 2), lambda: MI(0, 2) ** MI(1, 2), '[0, 4]'),
 ('[2,3] ** inf', lambda: V1(2, 3) ** math.inf, lambda: MI(2, 3) ** math.inf, '[inf] as python 2**inf'),
 ('2 ** [1,3] (__rpow__)', lambda: 2 ** V1(1, 3), lambda: 2 ** MI(1, 3), '[2, 8]'),
 ('pow([2,3],2,5)', lambda: pow(V1(2, 3), 2, 5), lambda: pow(MI(2, 3), 2, 5), 'v1 enumerates degenerate points only'),
 ('pow({2,3},2,5)', lambda: pow(V1.merge({2, 3}), 2, 5), lambda: pow(MI(2) | MI(3), 2, 5), '{4 % 5, 9 % 5} = {4}'),
]
for name, f1, f2, note in cases:
    r1, r2 = run(f1), run(f2)
    print(f'{name:22s} v1: {str(r1):42s} v2: {str(r2):42s} | {note}')

# DROPPED 3-arg pow: the workaround, run against v1 on integral sets
ok = 0; bad = []
for _ in range(200):
    bs = set(rng.sample(range(-5, 20), rng.randint(1, 4))); es = set(rng.sample(range(0, 6), rng.randint(1, 3)))
    ms = set(rng.sample(range(1, 9), rng.randint(1, 2)))
    r1 = run(lambda: pow(V1.merge(bs), V1.merge(es), V1.merge(ms)))
    B, E, M = (MI.from_pieces((v, v) for v in s) for s in (bs, es, ms))
    r2 = MI().union(*(MI(pow(int(b), int(e), int(m))) for b in B.degenerate_points for e in E.degenerate_points for m in M.degenerate_points))
    if isinstance(r1, V1) and conv(r1) == r2: ok += 1
    else: bad.append((bs, es, ms, str(r1), str(r2)))
print(f'3-arg pow workaround vs v1: {ok}/200 agree', bad[:3])
