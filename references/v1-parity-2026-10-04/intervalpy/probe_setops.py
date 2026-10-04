from common import *
from collections import Counter

def run(f):
    try:
        return 'ok', f()
    except Exception as e:
        return type(e).__name__, str(e)[:60]

def brute(name, v1res, v2res, a, b, pred):
    """v1res: ('ok', Interval) or exception; v2res MultiInterval. pred(p) -> bool exact set membership"""
    pts = probe_points(*ends_of(a), *ends_of(b), *ends_of(v2res))
    v2ok = all((p in v2res) == pred(p) for p in pts)
    check(name + ' v2 vs brute', v2ok, f'a={a} b={b} v2={v2res}')
    if v1res[0] == 'ok':
        v1ok = all((p in v1res[1]) == pred(p) for p in pts)
        return 'v1ok' if v1ok else 'v1WRONG'
    return 'v1raise:' + v1res[0]

rng = random.Random(7)
stats = {k: Counter() for k in ['contains_num', 'contains_iv', 'overlaps', 'overlaps_adj', 'intersect', 'union', 'difference', 'symdiff', 'expand', 'closed_hull', 'invert']}
examples = {}
def note(k, outcome, ex):
    stats[k][outcome] += 1
    examples.setdefault((k, outcome), ex)

for _ in range(1500):
    a = rand_interval(rng)
    b = rand_interval(rng) if rng.random() < 0.8 else I(*(lambda v: (v, False, v, True))(rand_val(rng)))
    A, B = to_v2(a), to_v2(b)
    # scalar membership
    for p in probe_points(*ends_of(a), *ends_of(b))[:12]:
        check('contains num', (p in a) == (p in A), (a, p))
    note('contains_num', 'agree', None)
    # interval containment
    check('contains iv', (b in a) == (B in A), (a, b)); note('contains_iv', (b in a) == (B in A), (a, b))
    # overlaps
    ov = a.overlaps(b)
    check('overlaps', ov == A.overlaps(B), (a, b)); note('overlaps', ov == A.overlaps(B), (a, b))
    adj = a.overlaps(b, or_adjacent=True)
    v2adj = A.overlaps(B) or A.adjoins(B)
    check('overlaps or_adjacent', adj == v2adj, (a, b)); note('overlaps_adj', adj == v2adj, (a, b))
    check('overlaps or_adjacent alt', adj == (A | B).is_contiguous, (a, b))
    # overlaps with a Real
    p = rand_val(rng)
    check('overlaps real', a.overlaps(p) == (p in A), (a, p))
    check('overlaps real adj', a.overlaps(p, or_adjacent=True) == (A.overlaps(M(p)) or A.adjoins(M(p))), (a, p, a.overlaps(p, or_adjacent=True)))
    # intersect / union / difference / symmetric difference
    for k, f1, f2, pred in [
        ('intersect', lambda: a.intersect(b), lambda: A & B, lambda q: (q in a) and (q in b)),
        ('union', lambda: a.union(b), lambda: A | B, lambda q: (q in a) or (q in b)),
        ('difference', lambda: a.difference(b), lambda: A.difference(B), lambda q: (q in a) and not (q in b)),
        ('symdiff', lambda: a.symmetric_difference(b), lambda: A ^ B, lambda q: (q in a) != (q in b)),
    ]:
        o = brute(k, run(f1), f2(), a, b, pred); note(k, o, (a, b, run(f1)))
    # same ops with a plain Real
    for k, f1, f2, pred in [
        ('intersect', lambda: a.intersect(p), lambda: A & p, lambda q: (q in a) and q == p),
        ('union', lambda: a.union(p), lambda: A | p, lambda q: (q in a) or q == p),
        ('difference', lambda: a.difference(p), lambda: A.difference(p), lambda q: (q in a) and q != p),
        ('symdiff', lambda: a.symmetric_difference(p), lambda: A ^ p, lambda q: (q in a) != (q == p)),
    ]:
        o = brute(k + '_real', run(f1), f2(), a, M(p), pred); note(k, 'real:' + o, (a, p, run(f1)))
    # expand
    d = rng.choice([0, 1, F(1, 3), 2.5])
    e1 = run(lambda: a.expand(d)); E2 = A.expand(d)
    if e1[0] == 'ok':
        note('expand', same_reals('expand', e1[1], E2), (a, d))
    else:
        note('expand', 'v1raise', (a, d, e1))
    # closed_hull
    c1 = run(lambda: a.closed_hull()); C2 = A.closed_hull
    if c1[0] == 'ok':
        note('closed_hull', same_reals('closed_hull', c1[1], C2), (a,))
    else:
        o = brute('closed_hull', c1, C2, a, a, lambda q: a.start <= q <= a.end)
        note('closed_hull', o, (a, c1))
    # invert
    i1 = run(lambda: ~a); I2 = ~A
    o = brute('invert', i1, I2, a, a, lambda q: q not in a)
    note('invert', o, (a, i1))
    # v2 complement puts +-inf points in when a does not reach them
    if math.isinf(a.start) or math.isinf(a.end):
        check('invert inf point', (-INF in I2) == (-INF not in A) and (INF in I2) == (INF not in A))

for k, c in stats.items():
    print(k, dict(c))
for (k, o), ex in sorted(examples.items(), key=str):
    if ex is not None and o not in (True, 'agree', 'v1ok', 'real:v1ok'):
        print('  example', k, o, ex)

# edge cases by hand
print('expand negative:', run(lambda: I(0, False, 1, True).expand(-1)), run(lambda: M(0, 1).expand(-1)))
print('expand inf:', run(lambda: I(0, False, 1, True).expand(INF)), run(lambda: M(0, 1).expand(INF)))
print('expand str:', run(lambda: I(0, False, 1, True).expand('1')), run(lambda: M(0, 1).expand('1')))
print('expand ray:', run(lambda: I(-INF, True, 1, True).expand(1)), run(lambda: M(-INF, 1, start_closed=False).expand(1)))
print('closed_hull ray:', run(lambda: I(0, True, INF, False).closed_hull()), M(0, INF, start_closed=False, end_closed=False).closed_hull)
print('closed_hull identity:', I(0, False, 1, True).closed_hull() is I(0, False, 1, True), M(0, 1).closed_hull == M(0, 1))
print('symdiff same start:', run(lambda: I(0, False, 1, True).symmetric_difference(I(0, False, 2, True))), M(0, 1) ^ M(0, 2))
print('invert ray:', run(lambda: ~I(-INF, True, 0, True)), ~M(-INF, 0, start_closed=False))
print('contains str:', run(lambda: 'x' in I(0, False, 1, True)), run(lambda: 'x' in M(0, 1)))
print('overlaps str:', run(lambda: I(0, False, 1, True).overlaps('x')), run(lambda: M(0, 1).overlaps('x')))
# SELFTEST
brute('SELFTEST expected mismatch', ('ok', I(0, False, 1, True)), M(0, 1), I(0, False, 1, True), I(0, False, 1, True), lambda q: 0 < q <= 1)
report_end(__file__)
