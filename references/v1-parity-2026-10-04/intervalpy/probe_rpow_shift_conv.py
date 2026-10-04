from common import *
from collections import Counter
from probe_arith_helpers import run, exact_interval

def approx_same(i1, R, tol=1e-9):
    if len(R) != 1:
        return False
    def close(x, y):
        if math.isinf(x) or math.isinf(y):
            return x == y
        return abs(x - y) <= tol * max(1, abs(x), abs(y))
    return close(float(i1.start), float(R.inf)) and close(float(i1.end), float(R.sup)) and \
        i1.start_closed == R.inf_closed and (i1.end_closed == R.sup_closed or math.isinf(i1.end))

# __rpow__: b ** Interval
rng = random.Random(9)
st = Counter(); ex = {}
for _ in range(1500):
    a = exact_interval(rng); A = to_v2(a)
    b = rng.choice([2, 3, F(1, 2), 0.5, 1, 0, -2, -1, F(-1, 2), 10])
    v1r = run(lambda: b ** a); R = b ** A
    if v1r[0] != 'ok':
        o = 'v1raise:' + v1r[0] + ':' + v1r[1][:40]
    else:
        o = 'agree~' if approx_same(v1r[1], R) else 'DIFFER'
    key = ('b>0' if b > 0 else ('b=0' if b == 0 else 'b<0'), 'deg' if a.is_degenerate else 'iv', o)
    st[key] += 1; ex.setdefault(key, (b, a, v1r, R))
for k in sorted(st, key=str):
    print(k, st[k])
for k, e in sorted(ex.items(), key=str):
    if k[-1] != 'agree~':
        print('  example', k, e)
print('0 ** [0]:', run(lambda: 0 ** I(0, False, 0, True)), 0 ** M(0), '| M(0) ** 0:', M(0) ** 0)
print('0 ** [-2,-1]:', run(lambda: 0 ** I(-2, False, -1, True)), 0 ** M(-2, -1))
print('(-2) ** [2]:', run(lambda: (-2) ** I(2, False, 2, True)), (-2) ** M(2), '| workaround M(-2) ** 2:', M(-2) ** 2)
print('(-2) ** [3]:', run(lambda: (-2) ** I(3, False, 3, True)), (-2) ** M(3), '| workaround M(-2) ** 3:', M(-2) ** 3)
print('[-3,1] ** [2] (degenerate Interval exponent):', run(lambda: I(-3, False, 1, True) ** I(2, False, 2, True)), M(-3, 1) ** M(2), '| workaround A ** int(B):', M(-3, 1) ** int(M(2)))
print('inf ** A:', run(lambda: INF ** I(1, False, 2, True)), INF ** M(1, 2))

# shifts
for a, n in [(I(1, False, 3, True), 2), (I(-3, True, 5, False), 1), (I(1, False, 3, True), I(0, False, 2, True)), (I(0, False, 0, True), 3)]:
    l1, r1 = run(lambda: a << n), run(lambda: a >> n)
    A = to_v2(a); N = to_v2(n) if isinstance(n, I) else n
    l2 = run(lambda: A << N)
    wl = A * 2 ** N
    wr = A // 2 ** N
    print('shift', a, n, '| v1 <<', l1, ' v2 <<', l2[0], ' workaround A*2**n:', wl, '| v1 >>', r1, ' A//2**n:', wr)
    if l1[0] == 'ok':
        check('<< workaround', same_reals('<<', l1[1], wl))
print('float shift v1:', run(lambda: I(1.5, False, 3, True) << 1), '| v2 M(1.5,3)*2:', M(1.5, 3) * 2)
print('v2 >> refused:', run(lambda: M(1, 3) >> 1))

# float() / int()
for a in [I(2, False, 2, True), I(F(7, 2), False, F(7, 2), True), I(-F(7, 2), False, -F(7, 2), True), I(10**20 + 1, False, 10**20 + 1, True), I(0, False, 1, True)]:
    A = to_v2(a)
    f1, f2 = run(lambda: float(a)), run(lambda: float(A))
    i1, i2 = run(lambda: int(a)), run(lambda: int(A))
    print('conv', a, '| float', f1, f2, '| int', i1, i2)
    check('float conv', f1 == f2 or (f1[0] != 'ok' and f2[0] != 'ok'), (a, f1, f2))
print('int of 10**20+1: exact is', 10**20 + 1)
print('float(empty) v2:', run(lambda: float(M())), '| float of two points v2:', run(lambda: float(M(1) | M(2))))

# repr / str
for a in [I(0, False, 1, True), I(0, True, 1, False), I(2, False, 2, True), I(-INF, True, 3, True), I(-INF, True, INF, False), I(F(1, 3), False, 0.5, True)]:
    A = to_v2(a)
    print(f'repr v1 {a!r:55s} v2 {A!r:45s} | str v1 {str(a):14s} v2 {str(A)}')
    check('repr evals back', eval(repr(A), {'MultiInterval': M}) == A)
    check('str parses back', M.parse(str(A)) == A)
    check('str parse of v1 str', run(lambda: M.parse(str(a).replace('∞', 'inf'))) == ('ok', A), (str(a),))
print('v2 parses v1 str with the infinity sign?', run(lambda: M.parse('(-∞, 3]')))
check('SELFTEST expected mismatch', approx_same(I(1, False, 2, True), M(1, 2, end_closed=False)))
report_end(__file__)
