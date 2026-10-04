"""MultiInterval.merge: every input form and n_overlaps mode, v1 vs v2 compositions vs brute force"""
from common import *
import random, itertools
from functools import reduce
inf = math.inf

# ---------- pure-python brute force over piece lists (independent of both libraries)
def in_pieces(pcs, x):
    return any((x > lo or (x == lo and lc)) and (x < hi or (x == hi and hc)) for lo, hi, lc, hc in pcs)

# ---------- v2 compositions
def v2_union(sets):
    return M2().union(*sets)
def v2_atleast(sets, k):
    if k > len(sets):
        return M2()
    return M2().union(*(reduce(lambda a, b: a & b, c) for c in itertools.combinations(sets, k)))
def v2_exactly(sets, k):
    if k == 0:
        u = v2_union(sets)
        return u.hull.difference(u) if u else M2()   # v1's n_overlaps=0: the gaps inside the hull
    return v2_atleast(sets, k).difference(v2_atleast(sets, k + 1))
def v2_merge(sets, n_overlaps=None):
    if n_overlaps is None:
        return v2_union(sets)
    ks = {n_overlaps} if isinstance(n_overlaps, int) else set(n_overlaps)
    return M2().union(*(v2_exactly(sets, k) for k in ks))

# ---------- random inputs: each input is a list of pieces; rendered in one of v1's forms
VALS = [-5, -3, -2, -1, 0, 1, 2, 3, 4, 6, F(1, 2), F(-7, 3), 1.5, -0.25]
def rand_input():
    """returns (v1_form, v2_set, pieces)"""
    form = random.choice(['mi', 'num', 'set', 'list1', 'list2', 'tuple2', 'mi', 'mi'])
    if form == 'num':
        x = random.choice(VALS); return x, M2(x), [(x, x, True, True)]
    if form == 'set':
        s = set(random.sample(VALS, random.randint(0, 3)))
        return s, M2.from_pieces((x, x) for x in s), [(x, x, True, True) for x in s]
    if form == 'list1':
        x = random.choice(VALS); return [x], M2(x), [(x, x, True, True)]
    if form in ('list2', 'tuple2'):
        a, b = sorted(random.sample(VALS, 2))
        c = form == 'list2'
        return (([a, b] if c else (a, b)), M2(a, b, start_closed=c, end_closed=c), [(a, b, c, c)])
    # a v1 MultiInterval of up to 3 random pieces (built by v1's own union, then converted)
    pcs = []
    m1 = M1()
    for _ in range(random.randint(0, 3)):
        a, b = sorted(random.sample(VALS, 2))
        if random.random() < .2:
            b = a
        lc, hc = (True, True) if a == b else (random.random() < .5, random.random() < .5)
        if random.random() < .15 and a != b:
            a = -inf; lc = False
        if random.random() < .15 and a != b:
            b = inf; hc = False
        pcs.append((a, b, lc, hc))
        m1 = m1.union(M1(a, b, start_closed=lc, end_closed=hc))
    return m1, v1_to_v2(m1), pcs

def run_sweep(n_cases, mode_gen, label):
    t, tv1, tv2 = Tally(label + ': v1 vs v2'), Tally(label + ': v1 vs brute force'), Tally(label + ': v2 vs brute force')
    crashes = []
    for _ in range(n_cases):
        inputs = [rand_input() for _ in range(random.randint(1, 4))]
        n_ov = mode_gen(len(inputs))
        o1 = outcome(lambda: M1.merge(*[i[0] for i in inputs], n_overlaps=n_ov))
        r2 = v2_merge([i[1] for i in inputs], n_ov)
        if o1[0] == 'raise':
            crashes.append((o1[1], [str(i[0]) if isinstance(i[0], M1) else i[0] for i in inputs], n_ov)); continue
        r1 = v1_to_v2(o1[1])
        allv = [v for i in inputs for p in i[2] for v in p[:2]]
        pts = probe_points(allv)
        ks = None if n_ov is None else ({n_ov} if isinstance(n_ov, int) else set(n_ov))
        def brute(x):
            c = sum(in_pieces(i[2], x) for i in inputs)
            if ks is None:
                return c >= 1
            if 0 in ks and c == 0:
                lo = min(pts, key=lambda p: (not any(in_pieces(i[2], p) for i in inputs), p))
                # inside the hull of the union, not in any input
                inside = [p for p in pts if any(in_pieces(i[2], p) for i in inputs)]
                return bool(inside) and inside[0] < x < inside[-1]
            return c in ks
        t.check(r1 == r2, (str(r1), str(r2), [i[0] if not isinstance(i[0], M1) else str(i[0]) for i in inputs], n_ov))
        tv1.check(not same_set_by_membership(brute, lambda p: p in r1, pts), ('v1', str(r1), n_ov))
        tv2.check(not same_set_by_membership(brute, lambda p: p in r2, pts), ('v2', str(r2), n_ov))
    t.report(3); tv1.report(3); tv2.report(3)
    print(f'[{label}] v1 crashes: {len(crashes)}')
    for c in crashes[:3]:
        print('   ', c)
    return t, crashes

random.seed(42)
print('=== A. n_overlaps=None (union) over every input form')
run_sweep(500, lambda n: None, 'union')
print('=== C. n_overlaps=k (exactly k), k in 1..n')
run_sweep(500, lambda n: random.randint(1, n), 'exactly-k')
print('=== C2. n_overlaps=set of ks')
run_sweep(300, lambda n: set(random.sample(range(1, n + 1), random.randint(1, n))), 'k-set')
print('=== D. n_overlaps=0 (int; the gaps inside the hull)')
run_sweep(300, lambda n: 0, 'zero')

print('=== B. edge table of input forms')
def edge(label, *args, **kw):
    o1 = outcome(lambda: M1.merge(*args, **kw))
    s = outcome(lambda: str(o1[1])) if o1[0] == 'ok' else None
    txt = ('RAISE ' + o1[1]) if o1[0] == 'raise' else (s[1] if s[0] == 'ok' else 'BUILT INVALID OBJECT, endpoints=%r, str() -> %s' % (o1[1].endpoints, s[1]))
    print(f'  {label:45s} v1 -> {txt}')
edge('no args')
edge('list []', [])
edge('list [3, 1] reversed', [3, 1])
edge('list [1,2,3]', [1, 2, 3])
edge('tuple ()', ())
edge('tuple (5,)', (5,))
edge('tuple (1, 1)', (1, 1))
edge('tuple (3, 1) reversed', (3, 1))
edge('tuple (1,2,3)', (1, 2, 3))
edge('list ["a", 2]', ['a', 2])
edge('set {1, "a"}', {1, 'a'})
edge('frozenset {1, 2}', frozenset({1, 2}))
edge('dict', {1: 2})
edge('None', None)
edge('float nan', math.nan)
edge('list [1, inf]', [1, inf])
edge('tuple (1, inf)', (1, inf))
edge('number inf', inf)
edge('list [-0.0, 1]', [-0.0, 1])
edge('n_overlaps=-1', [0, 1], n_overlaps=-1)
edge('n_overlaps=[0]', [0, 1], n_overlaps=[0])
edge('n_overlaps=[1.5]', [0, 1], n_overlaps=[1.5])
edge('n_overlaps=1.0', [0, 1], n_overlaps=1.0)
edge('n_overlaps=True', [0, 1], [0.5, 2], n_overlaps=True)
edge('n_overlaps=5 > n', [0, 1], n_overlaps=5)
print('  v2: from_pieces([(3, 1)]) ->', outcome(lambda: M2.from_pieces([(3, 1)])))
print('  v2: from_pieces([(1, 1, False, False)]) ->', outcome(lambda: str(M2.from_pieces([(1, 1, False, False)]))))
print('  v2: from_pieces([(1, inf, False, False)]) ->', outcome(lambda: str(M2.from_pieces([(1, inf, False, False)]))))
print('  v2: union of points from frozenset ->', M2().union(*frozenset({1, 2})))
print('  v2: M2().union() ->', M2().union(), '| M2.from_pieces([]) ->', M2.from_pieces([]))

print('=== sabotage: a wrong composition must be caught')
random.seed(7)
saved = v2_exactly
def v2_exactly(sets, k):   # deliberately wrong: at-least-k instead of exactly-k
    return v2_atleast(sets, k) if k else saved(sets, k)
t, _ = run_sweep(200, lambda n: random.randint(1, n), 'SABOTAGE exactly-k')
assert t.bad, 'sabotage not caught'
print('sabotage caught')
