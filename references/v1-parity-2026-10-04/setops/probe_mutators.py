"""add/clear/discard/pop/remove: v1 in-place results vs v2 pure compositions; Builder as the incremental route"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random

def v2_add(A, x): return A | x
def v2_clear(A): return M2()
def v2_discard(A, x): return A.difference(x) if x in A else A
def v2_remove(A, x):
    if x not in A: raise KeyError(x)
    return A.difference(x)
def v2_pop(A):
    if not A: raise KeyError('pop from empty')
    last = A.pieces[-1]
    return last, A.difference(last)

def outcome1(f):
    try: return ('ok', f())
    except Exception as e: return ('raise', type(e).__name__)

rng = random.Random(99)
T = {k: Tally(k) for k in ['add MI', 'add num', 'clear', 'discard MI', 'discard num', 'remove MI', 'remove num', 'pop', 'pop returned', 'Builder vs v1 repeated add']}
v1_raises = {k: [] for k in T}
for _ in range(400):
    pa = rand_pieces(rng); a1, a2 = twin(pa)
    # MI operand: half the time a sub-piece of A (so discard/remove act)
    if a2 and rng.random() < 0.5:
        p = rng.choice(a2.pieces); pb = [(p.inf, p.sup, p.inf_closed, p.sup_closed)]
    else:
        pb = rand_pieces(rng, kmax=2)
    b1, b2 = twin(pb)
    x = rng.choice(GRID) if rng.random() < 0.7 or not a2 else a2.pieces[0].inf if a2.pieces[0].inf_closed and a2.pieces[0].inf != -inf else 0
    for name, f1, f2 in [
        ('add MI', lambda: a1.copy().add(b1), lambda: v2_add(a2, b2)),
        ('add num', lambda: a1.copy().add(x), lambda: v2_add(a2, x)),
        ('clear', lambda: a1.copy().clear(), lambda: v2_clear(a2)),
        ('discard MI', lambda: a1.copy().discard(b1), lambda: v2_discard(a2, b2)),
        ('discard num', lambda: a1.copy().discard(x), lambda: v2_discard(a2, x)),
        ('remove MI', lambda: a1.copy().remove(b1), lambda: v2_remove(a2, b2)),
        ('remove num', lambda: a1.copy().remove(x), lambda: v2_remove(a2, x)),
    ]:
        o1, o2 = outcome1(f1), outcome1(f2)
        lab = (pa, pb if 'MI' in name else x)
        if o1[0] == 'raise' or o2[0] == 'raise':
            T[name].check(lab, None if o1 == o2 else f'v1 {o1} v2 {o2}')
            if o1 != o2: v1_raises[name].append((lab, o1, o2))
        else:
            T[name].check(lab, same_set_v1_v2(o1[1], o2[1]))
    # pop
    y1 = a1.copy()
    o1 = outcome1(lambda: y1.pop()); o2 = outcome1(lambda: v2_pop(a2))
    if o1[0] == 'raise' or o2[0] == 'raise':
        T['pop'].check(pa, None if o1 == o2 else f'v1 {o1} v2 {o2}')
    else:
        T['pop'].check(pa, same_set_v1_v2(y1, o2[1][1]))
        T['pop returned'].check(pa, same_set_v1_v2(o1[1], o2[1][0]))
    # Builder: v1 built by repeated add of pieces, v2 by Builder
    z1 = M1()
    bld = v2.Builder()
    for lo, hi, lc, hc in pa:
        z1.add(M1(lo) if lo == hi else M1(lo, hi, start_closed=lc, end_closed=hc))
        bld.add_piece(lo, hi, lc, hc)
    T['Builder vs v1 repeated add'].check(pa, same_set_v1_v2(z1, M2.from_cuts(bld.build())))
for t in T.values(): t.report(3)
for k, v in v1_raises.items():
    if v: print('raise-diff', k, len(v), v[:2])
# hand cases
for lab, f1, f2 in [
    ('discard([0,5]) from [0,3] (not a subset: v1 no-op)', lambda: str(M1(0, 3).discard(M1(0, 5))), lambda: str(v2_discard(M2(0, 3), M2(0, 5)))),
    ('discard(1) from [0,3]', lambda: str(M1(0, 3).discard(1)), lambda: str(v2_discard(M2(0, 3), 1))),
    ('remove(1) from [0,3]', lambda: str(M1(0, 3).remove(1)), lambda: str(v2_remove(M2(0, 3), 1))),
    ('remove(5) from [0,3]', lambda: str(M1(0, 3).remove(5)), lambda: str(v2_remove(M2(0, 3), 5))),
    ('pop() of empty', lambda: str(M1().pop()), lambda: str(v2_pop(M2()))),
    ('pop() of [0,1]u[2,3] returns', lambda: str(M1(0, 1).union(M1(2, 3)).pop()), lambda: str(v2_pop(M2(0, 1) | M2(2, 3))[0])),
    ('add(3) to empty', lambda: str(M1().add(3)), lambda: str(M2() | 3)),
    ('add(3) to [0,1]', lambda: str(M1(0, 1).add(3)), lambda: str(M2(0, 1) | 3)),
    ('add returns self (chaining)', lambda: (lambda m: m.add(M1(5)) is m)(M1(0, 1)), lambda: 'n/a (immutable)'),
    ('Builder add_point/add_piece', lambda: str(M1(3).union(M1(0, 1, end_closed=False))), lambda: str(M2.from_cuts(v2.Builder().add_point(3).add_piece(0, 1, True, False).build()))),
]:
    o1, o2 = outcome1(f1), outcome1(f2)
    print(f'{lab:55s} v1={o1!s:45.45s} v2={o2!s:45.45s}{"" if o1 == o2 else "   <-- DIFF"}')
s = Tally('sabotage'); s.check('x', same_set_v1_v2(M1(0, 3).discard(M1(1, 2)), M2(0, 3))); assert s.report(0) == 1
