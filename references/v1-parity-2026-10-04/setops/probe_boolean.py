"""union/intersection/difference/symmetric_difference and their *_update forms, 0..4 other args, MI and number args"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random, copy

def truth(op, sets2, p):
    m = [p in s for s in sets2]
    if op == 'union': return any(m)
    if op == 'intersection': return all(m)
    if op == 'difference': return m[0] and not any(m[1:])
    if op == 'symmetric_difference': return sum(m) % 2 == 1

def check_truth(op, res2, sets2):
    for p in points_for(res2, *sets2):
        if (p in res2) != truth(op, sets2, p):
            return f'at {p}: got {p in res2}'
    return None

rng = random.Random(4242)
ops = ['union', 'intersection', 'difference', 'symmetric_difference']
T = {}
v1_vs_truth = {}
for op in ops:
    for k in range(0, 5):
        for numbers in (False, True):
            key = f'{op} k={k} {"with numbers" if numbers else "MI only"}'
            T[key] = Tally(key); v1_vs_truth[key] = []
            for _ in range(120):
                pa = rand_pieces(rng)
                a1, a2 = twin(pa)
                others1, others2, others_desc = [], [], []
                for j in range(k):
                    if numbers and rng.random() < 0.5:
                        x = rng.choice(GRID)
                        others1.append(x); others2.append(x); others_desc.append(x)
                    else:
                        pb = rand_pieces(rng); b1, b2 = twin(pb)
                        others1.append(b1); others2.append(b2); others_desc.append(pb)
                lab = (pa, others_desc)
                before = from_v1(a1)
                for inplace in (False, True):
                    tk = key + (' INPLACE' if inplace else '')
                    if tk not in T: T[tk] = Tally(tk); v1_vs_truth[tk] = []
                    x1 = a1.copy()
                    try:
                        r1 = getattr(x1, op + '_update' if op != 'union' else 'update')(*others1) if inplace else getattr(x1, op)(*others1)
                        if inplace: assert r1 is x1
                    except Exception as e:
                        r1 = f'raise {type(e).__name__}: {e}'
                    if not inplace: assert from_v1(x1) == before  # non-inplace leaves self alone
                    try:
                        r2 = getattr(a2, op)(*others2)
                    except Exception as e:
                        r2 = f'raise {type(e).__name__}: {e}'
                    assert not isinstance(r2, str), (op, lab, r2)
                    ops2 = [a2] + [o if isinstance(o, M2) else M2(o) for o in others2]
                    tr = check_truth(op, r2, ops2)
                    assert tr is None, (op, lab, tr)  # v2 against brute truth
                    if isinstance(r1, str):
                        T[tk].check(lab, f'v1 {r1}')
                        v1_vs_truth[tk].append((lab, r1)); continue
                    d = same_set_v1_v2(r1, r2)
                    T[tk].check(lab, d)
                    if d is not None: v1_vs_truth[tk].append((lab, check_truth(op, from_v1(r1), ops2)))
bad = 0
for t in T.values():
    bad += t.report(show=2)
print('--- v1 vs brute truth where v1 and v2 differ (None means v1 is right on the grid):')
for k, v in v1_vs_truth.items():
    if v: print(' ', k, len(v), v[:2])
# sabotage
s = Tally('sabotage'); a1, a2 = twin([(0, 2, True, True)]); b1, b2 = twin([(1, 3, True, True)])
s.check('x', same_set_v1_v2(a1.union(b1), a2 & b2)); assert s.report(0) == 1
