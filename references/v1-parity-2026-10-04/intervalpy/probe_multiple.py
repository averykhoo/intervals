from common import *
from collections import Counter
import copy
from probe_arith_helpers import run


def rand_mi(rng, k=None):
    k = rng.randint(0, 4) if k is None else k
    return [rand_interval(rng) for _ in range(k)]


def build(ivs):
    return MI(*ivs), M().union(*[to_v2(i) for i in ivs])


rng = random.Random(21)
st = Counter()
ex = {}


def note(k, o, e):
    st[(k, o)] += 1
    ex.setdefault((k, o), e)


for _ in range(800):
    ia, ib = rand_mi(rng), rand_mi(rng)
    a, A = build(ia)
    b, B = build(ib)
    # construction / normal form: same pieces
    note('init', same_reals('init', a, A) and len(a.intervals) == len(A), (ia,))
    note('init pieces', [to_v2(i) for i in a] == list(A), (ia, a, A))
    note('from_pieces', M.from_pieces([(i.start, i.end, i.start_closed, i.end_closed) for i in ia]) == A, None)
    # length / infimum / supremum
    L1 = run(lambda: a.length)
    note('length', (L1[1] == A.size.length) if L1[0] == 'ok' else 'v1raise', (a, L1, A.size.length))
    note('infimum', (a.infimum is None and not A) or (bool(A) and a.infimum == A.inf), (a, a.infimum))
    note('supremum', (a.supremum is None and not A) or (bool(A) and a.supremum == A.sup), (a, a.supremum))
    # contains: Real, Interval, MultipleInterval
    p = rand_val(rng)
    note('contains real', (p in a) == (p in A), (a, p))
    j = rand_interval(rng)
    note('contains iv', (j in a) == (to_v2(j) in A), (a, j))
    note('contains mi', (b in a) == (B in A), (a, b))
    note('eq', (a == b) == (A == B), (a, b))
    # reciprocal
    r1 = run(lambda: a.reciprocal())
    R = A.reciprocal()
    if r1[0] == 'ok':
        pts = probe_points(*ends_of(R), *ends_of(r1[1]))
        eq = all((q in r1[1]) == (q in R) for q in pts)
        sup = all((q in r1[1]) for q in pts if q in R)
        note('reciprocal', 'equal' if eq else ('v1 superset' if sup else 'OTHER'), (a, r1[1], R))
    else:
        note('reciprocal', 'v1raise ' + r1[0], (a, r1))
    # overlapping
    for adj in (False, True):
        o1 = run(lambda: a.overlapping(b, or_adjacent=adj))
        if adj:
            comp_doc = M().union(*[pc for pc in A if pc.overlaps(B) or pc.adjoins(B)])
            comp_ok = M().union(*[pc for pc in A if any(pc.overlaps(q) or pc.adjoins(q) for q in B)])
        else:
            comp_doc = comp_ok = M().union(*[pc for pc in A if pc.overlaps(B)])
        if o1[0] != 'ok':
            note(f'overlapping adj={adj}', 'v1raise ' + o1[0], (a, b, o1))
            continue
        good = mi_to_v2(o1[1]) == comp_ok
        note(f'overlapping adj={adj}', 'equal' if good else 'DIFFER', (a, b, o1[1], comp_ok))
        if adj:
            note('overlapping adj=True via p.overlaps(B) or p.adjoins(B) (whole B)', mi_to_v2(o1[1]) == comp_doc, (a, b, o1[1], comp_doc))
    note('overlapping real', mi_to_v2(a.overlapping(p)) == M().union(*[pc for pc in A if p in pc]), (a, p))
    # binary set ops, one operand
    for k, f1, f2, pred in [
        ('union', lambda: a.union(b), lambda: A | B, lambda q: (q in a) or (q in b)),
        ('intersection', lambda: a.intersection(b), lambda: A & B, lambda q: (q in a) and (q in b)),
        ('difference', lambda: a.difference(b), lambda: A.difference(B), lambda q: (q in a) and not (q in b)),
        ('symmetric_difference', lambda: a.symmetric_difference(b), lambda: A ^ B, lambda q: (q in a) != (q in b)),
    ]:
        a_before = repr(a)
        r1 = run(f1)
        R = f2()
        pts = probe_points(*ends_of(a), *ends_of(b), *ends_of(R))
        check(k + ' v2', all((q in R) == pred(q) for q in pts), (a, b, R))
        if r1[0] != 'ok':
            note(k, 'v1raise ' + r1[0] + ' ' + r1[1][:30], (a, b, r1))
            continue
        v1ok = all((q in r1[1]) == pred(q) for q in pts)
        note(k, 'v1ok' if v1ok else 'v1WRONG', (a, b, r1[1], R))
        note(k + ' leaves self alone', repr(a) == a_before, (a_before, a))
    # with an Interval and a Real operand
    J = to_v2(j)
    for k, f1, f2, pred in [
        ('union(Real)', lambda: a.union(p), lambda: A | p, lambda q: (q in a) or q == p),
        ('intersection(Real)', lambda: a.intersection(p), lambda: A & p, lambda q: (q in a) and q == p),
        ('difference(Real)', lambda: a.difference(p), lambda: A.difference(p), lambda q: (q in a) and q != p),
        ('symmetric_difference(Real)', lambda: a.symmetric_difference(p), lambda: A ^ p, lambda q: (q in a) != (q == p)),
        ('union(Interval)', lambda: a.union(j), lambda: A | J, lambda q: (q in a) or (q in j)),
        ('intersection(Interval)', lambda: a.intersection(j), lambda: A & J, lambda q: (q in a) and (q in j)),
        ('difference(Interval)', lambda: a.difference(j), lambda: A.difference(J), lambda q: (q in a) and not (q in j)),
        ('symmetric_difference(Interval)', lambda: a.symmetric_difference(j), lambda: A ^ J, lambda q: (q in a) != (q in j)),
    ]:
        r1 = run(f1)
        R = f2()
        pts = probe_points(*ends_of(a), p, *ends_of(j), *ends_of(R))
        check(k + ' v2', all((q in R) == pred(q) for q in pts), (a, p, j, R))
        if r1[0] != 'ok':
            note(k, 'v1raise ' + r1[0] + ' ' + r1[1][:30], (a, p, j, r1))
            continue
        note(k, 'v1ok' if all((q in r1[1]) == pred(q) for q in pts) else 'v1WRONG', (a, p, j, r1[1], R))
    # several operands
    c, C = build(rand_mi(rng))
    for k, f1, f2, pred in [
        ('union*2', lambda: a.union(b, c), lambda: A.union(B, C), lambda q: (q in a) or (q in b) or (q in c)),
        ('intersection*2', lambda: a.intersection(b, c), lambda: A.intersection(B, C), lambda q: (q in a) and (q in b) and (q in c)),
        ('difference*2', lambda: a.difference(b, c), lambda: A.difference(B, C), lambda q: (q in a) and not (q in b) and not (q in c)),
        ('symmetric_difference*2', lambda: a.symmetric_difference(b, c), lambda: A.symmetric_difference(B, C),
         lambda q: ((q in a) + (q in b) + (q in c)) % 2 == 1),
    ]:
        r1 = run(f1)
        R = f2()
        pts = probe_points(*ends_of(a), *ends_of(b), *ends_of(c), *ends_of(R))
        check(k + ' v2', all((q in R) == pred(q) for q in pts), (a, b, c, R))
        if r1[0] != 'ok':
            note(k, 'v1raise ' + r1[0] + ' ' + r1[1][:40], None)
            continue
        note(k, 'v1ok' if all((q in r1[1]) == pred(q) for q in pts) else 'v1WRONG', (a, b, c, r1[1], R))
    # in-place ops: v1 mutates and returns self; v2 rebinds
    for k, v2op in [('update', lambda X, Y: X | Y), ('intersection_update', lambda X, Y: X & Y),
                    ('difference_update', lambda X, Y: X.difference(Y)),
                    ('symmetric_difference_update', lambda X, Y: X ^ Y)]:
        x = a.copy()
        alias = x
        r1 = run(lambda: getattr(x, k)(b))
        if r1[0] != 'ok':
            note(k, 'v1raise ' + r1[0], None)
            continue
        note(k, (r1[1] is alias) and mi_to_v2(alias) == v2op(A, B), (a, b, alias))
    y = a.copy()
    y.update(I(100, False, 101, True))
    note('copy independent', I(100, False, 101, True) not in a.intervals, None)
    note('iter', [to_v2(i) for i in a] == list(A.pieces), None)

for k in sorted(st, key=str):
    print(k, st[k])
for k, e in sorted(ex.items(), key=str):
    if k[1] not in (True, 'v1ok', 'equal') and e is not None:
        print('  example', k, str(e)[:300])
print('empty infimum:', MI().infimum, run(lambda: M().inf))
print('repr/str:', repr(MI(I(0, False, 1, True), I(2, True, 3, False))), '|', str(MI(I(0, False, 1, True), I(2, True, 3, False))), '|',
      repr(M(0, 1) | M(2, 3, start_closed=False, end_closed=False)), str(M(0, 1) | M(2, 3, start_closed=False, end_closed=False)))
print('init with non-Interval:', run(lambda: MI(1)), run(lambda: M().union(1)), run(lambda: M().union('x')))
print('symdiff_update no args:', run(lambda: MI().symmetric_difference_update()), run(lambda: M().symmetric_difference()))
X = M(0, 1)
Y = X
X |= M(2, 3)
print('v2 X |= ... rebinds:', X, 'alias Y', Y)
print('v2 copy.copy:', copy.copy(M(0, 1)))
check('SELFTEST expected mismatch', mi_to_v2(MI(I(0, False, 1, True))) == M(0, 1, end_closed=False))
report_end(__file__)
