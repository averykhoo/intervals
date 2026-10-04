from common import *
def _s(e):
    try:
        return str(e)[:60]
    except Exception:
        return "<str failed>"


def outcome(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__, _s(e))

rng = random.Random(3)
for i in range(400):
    k = rng.randint(0, 3)
    sa = rand_pieces(rng); sos = [rand_pieces(rng) for _ in range(k)]
    a1 = build1(sa); os1 = [build1(s) for s in sos]
    a2 = build2(sa); os2 = [build2(s) for s in sos]
    A = oracle_set(canon2(a2)); OS = [oracle_set(canon2(o)) for o in os2]
    for name in ('union', 'intersection', 'difference', 'symmetric_difference'):
        r1 = outcome(lambda: getattr(a1, name)(*os1))
        r2 = outcome(lambda: getattr(a2, name)(*os2))
        r1c = canon1(r1[1]) if r1[0] == 'ok' else r1[:2]
        r2c = canon2(r2[1]) if r2[0] == 'ok' else r2[:2]
        check((name, k, sa, sos), r1c, r2c)
        # oracle
        if name == 'union': want = A.union(*OS)
        elif name == 'intersection': want = A.intersection(*OS)
        elif name == 'difference': want = A.difference(*OS)
        else:
            want = A
            for o in OS: want = want ^ o
        if r2[0] == 'ok':
            check((name + ' v2 oracle', k, sa, sos), oracle_set(r2c), want)
        if r1[0] == 'ok':
            check((name + ' v1 oracle', k, sa, sos), oracle_set(r1c), want)
        # in-place twin: v1 X_update mutates and returns self; v2 rebinding A = A.X(*others)
        upd = {'union': 'update'}.get(name, name + '_update')
        c1 = a1.copy()
        ru = outcome(lambda: getattr(c1, upd)(*os1))
        check((upd + ' returns self', k), ru[0] != 'ok' or ru[1] is c1, True)
        if ru[0] == 'ok':
            check((upd, k, sa, sos), canon1(c1), r1c)
    # scalar operands
    x = rng.choice(GRID); t = td(x)
    for name in ('union', 'intersection', 'difference', 'symmetric_difference'):
        r1 = outcome(lambda: getattr(a1, name)(t, pd.Timedelta(t)))
        r2 = outcome(lambda: getattr(a2, name)(t, pd.Timedelta(t)))
        check((name + ' scalar', sa, x), canon1(r1[1]) if r1[0] == 'ok' else r1[:2], canon2(r2[1]) if r2[0] == 'ok' else r2[:2])
    # item methods: add / discard / remove / pop / clear
    sb = rand_pieces(rng); b1 = build1(sb); b2 = build2(sb)
    B = oracle_set(canon2(b2))
    c1 = a1.copy(); c1.add(b1)
    check(('add', sa, sb), canon1(c1), canon2(a2 | b2))
    c1 = a1.copy(); c1.discard(b1)
    want_discard = a2 - b2 if False else (a2.difference(b2) if b2 in a2 else a2)
    check(('discard', sa, sb), canon1(c1), canon2(want_discard))
    c1 = a1.copy(); r = outcome(lambda: c1.remove(b1))
    r2 = ('ok',) if b2 in a2 else ('raise', 'KeyError')
    check(('remove outcome', sa, sb), r[:2] if r[0] != 'ok' else ('ok',), r2)
    if r[0] == 'ok':
        check(('remove result', sa, sb), canon1(c1), canon2(a2.difference(b2)))
    c1 = a1.copy(); r = outcome(lambda: c1.pop())
    if a2.is_empty:
        check(('pop empty', sa), r[:2], ('raise', 'KeyError'))
    else:
        last, rest = a2.pieces[-1], T2().union(*a2.pieces[:-1])
        check(('pop value', sa), canon1(r[1]), canon2(last))
        check(('pop rest', sa), canon1(c1), canon2(rest))
    c1 = a1.copy(); c1.clear()
    check(('clear', sa), canon1(c1), canon2(T2()))
from collections import Counter
print(Counter(f[0][0] for f in FAILS))
seen = set()
for f in FAILS:
    if f[0][0] not in seen:
        seen.add(f[0][0]); print('  first', f)
H = dt.timedelta(hours=1)
# hand-picked: three-way symmetric difference with a point in all three
X1 = T1(H, 3*H); X2 = T2(H, 3*H)
print('xor3 v1', X1.symmetric_difference(T1(H, 3*H), T1(H, 3*H)), ' v2', X2.symmetric_difference(T2(H, 3*H), T2(H, 3*H)))
print('difference() no args: v1', outcome(lambda: X1.difference()), ' v2', outcome(lambda: X2.difference()))
print('union() no args: v1', outcome(lambda: str(X1.union())), ' v2', outcome(lambda: str(X2.union())))
print('union(5): v1', outcome(lambda: X1.union(5)), ' v2', outcome(lambda: X2.union(5)))
print('union(NaT): v1', outcome(lambda: X1.union(pd.NaT)), ' v2', outcome(lambda: X2.union(pd.NaT)))
print('v2 has update/add/discard/remove/pop/clear:', [hasattr(X2, n) for n in ('update', 'intersection_update', 'difference_update', 'symmetric_difference_update', 'add', 'discard', 'remove', 'pop', 'clear', 'copy')])
assert not check('sabotage', canon1(X1.union(T1(4*H))), canon2(X2))
FAILS.pop()
report('probe_setops')
print('mismatch label x k:', Counter((f[0][0], f[0][1]) for f in FAILS if isinstance(f[0][1], int)))
print('remove mismatches not empty/empty:', [f for f in FAILS if f[0][0] == 'remove outcome' and (f[0][1] or f[0][2])][:3])
print('v2 symmetric_difference oracle mismatches:', sum(1 for f in FAILS if f[0][0] == 'symmetric_difference v2 oracle'))
