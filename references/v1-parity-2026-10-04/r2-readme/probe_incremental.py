"""
differential probe: compare.py's run_incremental_sort (and run_incremental_bisect) vs v2 Builder /
repeated union. v1 returns the SORTED, UNMERGED record list; its set is the union of its records.

run from the repo root:
    timeout 110 C:/Users/user/anaconda3/envs/intervals/python.exe -u .scratch/v1-parity/r2-readme/probe_incremental.py
"""
import copy
import random
import sys

sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-readme']
import compare as v1  # noqa: E402
from intervals import kernel  # noqa: E402
from intervals import MultiInterval as MI  # noqa: E402
import sweep_helpers as P  # noqa: E402

stats = {'cases': 0, 'mismatch': 0, 'v1_wrong': 0, 'v2_wrong': 0, 'not_sorted': 0, 'step_mismatch': 0}


def v2_builder_incremental(base, new):
    b = kernel.Builder()
    for p in P.v2_pieces(base):
        b.add_piece(*p)
    steps = []
    for p in P.v2_pieces(new):
        b.add_piece(*p)
        steps.append(MI.from_cuts(b.build()))  # build() does not consume: query after every add
    return MI.from_cuts(b.build()), steps


def v2_union_incremental(base, new):
    mi = MI.from_pieces(P.v2_pieces(base))
    for p in P.v2_pieces(new):
        mi = mi | MI.from_pieces([p])
    return mi


def check(label, base, new, *, sabotage=False):
    stats['cases'] += 1
    all_recs = base + new
    out_sort = v1.run_incremental_sort(copy.deepcopy(base), copy.deepcopy(new))
    out_bis = v1.run_incremental_bisect(copy.deepcopy(base), copy.deepcopy(new))
    if out_sort != sorted(all_recs):
        stats['not_sorted'] += 1
    v2b, steps = v2_builder_incremental(base, new)
    v2u = v2_union_incremental(base, new)
    ok = True
    for x in P.test_points(all_recs):
        want = P.oracle(all_recs, x)
        if sabotage:
            want = not want
        got = {'v1 sort': P.v1_member(out_sort, x), 'v1 bisect': P.v1_member(out_bis, x),
               'v2 Builder': x in v2b, 'v2 union': x in v2u}
        for k, g in got.items():
            if g != want:
                ok = False
                stats['v1_wrong' if k.startswith('v1') else 'v2_wrong'] += 1
        if len(set(got.values())) > 1:
            stats['mismatch'] += 1
            ok = False
    # the state after every insertion: v1 re-sorted list vs v2 Builder.build() at that step
    cur = list(base)
    for item, step in zip(new, steps):
        cur.append(item)
        cur.sort()
        for x in P.test_points(cur):
            if P.v1_member(cur, x) != (x in step):
                stats['step_mismatch'] += 1
                ok = False
    return ok


rng = random.Random(4102026)
bad = 0
for i in range(400):
    base = P.rand_records(rng, rng.randint(0, 8))
    if rng.random() < 0.5:
        base.sort()  # bisect assumes a sorted base; also try unsorted
    new = P.rand_records(rng, rng.randint(0, 5))
    if not check(f'rand#{i}', base, new):
        bad += 1
print(f'random incremental: 400 cases, disagreeing: {bad}')

random.seed(11)
for i in range(5):
    base = v1.generate_sorted_chunk(60, 0)
    new = [((x[0][0] * 0.5, x[0][1]), x[1]) for x in v1.generate_sorted_chunk(10, 0)]  # benchmark2's scatter
    if not check(f'bench2-shape#{i}', base, new):
        bad += 1
print('stats:', stats)

caught = not check('SABOTAGE', [((0, 0), (1, 0))], [((2, 2), (3, -2))], sabotage=True)
print('sabotage caught:', caught)
assert caught
