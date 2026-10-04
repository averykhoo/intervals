from common import *

def outcome(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__)

rng = random.Random(4)
for i in range(400):
    sa = rand_pieces(rng)
    a1, a2 = build1(sa), build2(sa)
    d = rng.randint(0, 6) * Fraction(1, 4)
    for dist in (td(d), pd.Timedelta(td(d))):
        r1 = a1.expand(dist); r2 = a2.expand(dist)
        check(('expand', sa, d), canon1(r1), canon2(r2))
        check(('expand orig untouched', sa, d), canon1(a1), canon2(a2))
        c = a1.copy(); r = c.expand(dist, inplace=True)
        check(('expand inplace', sa, d), (r is c, canon1(c)), (True, canon2(r2)))
        # oracle: x in expand(A, d) iff exists y in A within d of x, keeping closedness: brute-force on grid
        want = frozenset(x for x in GRID if any(
            (lo - d < x or (lc and lo - d == x)) and (x < hi + d or (hc and x == hi + d)) for lo, lc, hi, hc in canon2(a2)))
        check(('expand oracle', sa, d), oracle_set(canon2(r2)), want)
H = dt.timedelta(hours=1)
print('negative: v1', outcome(lambda: T1(H, 2*H).expand(-H)), 'v2', outcome(lambda: T2(H, 2*H).expand(-H)))
print('number: v1', outcome(lambda: T1(H, 2*H).expand(5)), 'v2', outcome(lambda: T2(H, 2*H).expand(5)))
print('inplace kw: v2', outcome(lambda: T2(H, 2*H).expand(H, inplace=True)))
print('empty: v1', canon1(T1().expand(H)), 'v2', canon2(T2().expand(H)))
print('merging: v1', T1(H, 2*H).union(T1(3*H, 4*H)).expand(H/2), 'v2', T2(H, 2*H).union(T2(3*H, 4*H)).expand(H/2))
print('open adjacency: v1', T1(H, 2*H, end_closed=False).union(T1(3*H, 4*H, start_closed=False)).expand(H/2), 'v2', T2(H, 2*H, end_closed=False).union(T2(3*H, 4*H, start_closed=False)).expand(H/2))
assert not check('sabotage', canon1(T1(H).expand(H)), canon2(T2(H)))
FAILS.pop()
report('probe_expand')
