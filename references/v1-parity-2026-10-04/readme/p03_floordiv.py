# README: "neither are __divmod__, __floordiv__, or __rfloordiv__ ... [0, inf) // 1 == range(infinity)"
from common import *
warnings.simplefilter('ignore')
def run(f):
    try:
        r = f(); return 'NotImplemented' if r is NotImplemented else r
    except Exception as e: return f'{type(e).__name__}: {str(e)[:60]}'

def floordiv_exact(ps, m):
    """integers floor(x/m) over the pieces (m > 0 scalar)"""
    out = set()
    for lo, hi, lc, hc in ps:
        for q in range(math.floor(lo / m) - 1, math.floor(hi / m) + 2):
            # exists x in piece with q <= x/m < q+1  <=> piece meets [q m, (q+1) m)
            a, b = q * m, (q + 1) * m
            l = max((lo, not lc), (a, False)); h = min((hi, hc), (b, False))
            if l[0] < h[0] or (l[0] == h[0] and not l[1] and h[1]): out.add(q)
    return MI.from_pieces((q, q) for q in out)

rng = random.Random(3)
n = agree = v1w = v2w = 0; ex = []
for _ in range(400):
    ps = []
    for _ in range(rng.randint(1, 2)):
        a, b = sorted(Fraction(rng.randint(-20, 20), 2) for _ in range(2))
        ps.append((a, b, True, True) if a == b else (a, b, rng.random() < .5, rng.random() < .5))
    m = rng.choice([1, 2, 3, Fraction(3, 2)])
    a1 = V1()
    for lo, hi, lc, hc in ps: a1.update(V1(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else V1(lo))
    a2 = MI.from_pieces(ps)
    want = floordiv_exact(ps, m)
    r1 = run(lambda: a1 // m); r2 = a2 // m
    if isinstance(r1, V1): r1 = conv(r1)
    n += 1; agree += r1 == r2; v1w += r1 != want; v2w += r2 != want
    if r1 != want: ex.append((str(a2), m, str(r1), str(r2), str(want)))
print(f'A // m: cases {n} agree {agree} v1_wrong {v1w} v2_wrong {v2w}')
for e in ex[:4]: print('  ', e)
assert floordiv_exact([(0, 10, True, True)], 4) != MI(0, 2)  # sabotage: v1's endpoint-floor answer is caught
for name, f1, f2 in [
    ('[0,10] // 4', lambda: V1(0, 10) // 4, lambda: MI(0, 10) // 4),
    ('[1,2) // 1', lambda: V1(1, 2, end_closed=False) // 1, lambda: MI(1, 2, end_closed=False) // 1),
    ('[0,inf) // 1', lambda: V1(0, math.inf, end_closed=False) // 1, lambda: MI(0, math.inf, end_closed=False) // 1),
    ('7 // [2,5]', lambda: 7 // V1(2, 5), lambda: 7 // MI(2, 5)),
    ('[1,3] // [1,2]', lambda: V1(1, 3) // V1(1, 2), lambda: MI(1, 3) // MI(1, 2)),
    ('divmod([0,10], 4)', lambda: divmod(V1(0, 10), 4), lambda: tuple(map(str, divmod(MI(0, 10), 4)))),
    ('divmod(7, [2,5])', lambda: divmod(7, V1(2, 5)), lambda: tuple(map(str, divmod(7, MI(2, 5))))),
]:
    print(f'{name:18s} v1: {str(run(f1)):50s} v2: {run(f2)}')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); MI(0, math.inf, end_closed=False) // 1
    print('warnings for [0,inf)//1:', [type(x.message).__name__ for x in w])
