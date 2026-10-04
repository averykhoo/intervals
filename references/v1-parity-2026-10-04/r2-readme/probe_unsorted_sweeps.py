"""
differential probe: compare.py's run_unsorted_process / run_timsort_no_key / run_incremental_sort
(and run_incremental_bisect, fast_sweep, optimized_sweep) vs v2 MultiInterval.from_pieces / Builder.

run from the repo root:
    timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/r2-readme/probe_unsorted_sweeps.py
"""
import copy
import math
import random
import sys
import warnings
from fractions import Fraction

sys.path[:0] = ['.', 'archive/v1']
import compare as v1  # noqa: E402
import intervals as v2  # noqa: E402
from intervals import kernel  # noqa: E402

MI = v2.MultiInterval
INF = math.inf


# ---- the shared reading of a record --------------------------------------------------------------

def closed_start(eps):
    return eps != 2


def closed_end(eps):
    return eps != -2


def rec_covers(rec, x):
    (s, se), (e, ee) = rec
    lo_ok = s < x or (s == x and closed_start(se))
    hi_ok = x < e or (x == e and closed_end(ee))
    return lo_ok and hi_ok


def oracle(records, x):
    """brute force: x is in the union of the input records"""
    return any(rec_covers(r, x) for r in records)


def v1_member(out, x):
    return any(rec_covers(r, x) for r in out)


def v2_pieces(records):
    return [(s, e, closed_start(se), closed_end(ee)) for (s, se), (e, ee) in records]


def v2_from_pieces(records):
    return MI.from_pieces(v2_pieces(records))


def v2_builder(records):
    b = kernel.Builder()
    for p in v2_pieces(records):
        b.add_piece(*p)
    return MI.from_cuts(b.build())


def test_points(records):
    vals = set()
    for (s, _), (e, _) in records:
        vals.add(s)
        vals.add(e)
    vals = sorted(vals)
    pts = list(vals)
    for a, b in zip(vals, vals[1:]):
        if math.isinf(a) or math.isinf(b):
            pts.append(b - 1 if math.isinf(a) else a + 1)
        else:
            pts.append((Fraction(a) + Fraction(b)) / 2)
    finite = [v for v in vals if not math.isinf(v)]
    if finite:
        pts.append(min(finite) - 1)
        pts.append(max(finite) + 1)
    else:
        pts.append(0)
    return pts


def v1_canonical(out):
    """v1 output pieces as (lo, hi, lo_closed, hi_closed), empty records dropped"""
    canon = []
    for (s, se), (e, ee) in out:
        lc, hc = closed_start(se), closed_end(ee)
        if s < e or (s == e and lc and hc):
            canon.append((s, e, lc, hc))
    return canon


def v2_canonical(mi):
    return [(lo, hi, lc, hc) for lo, lc, hi, hc in kernel.pieces(mi.cuts)]


def same_number(a, b):
    return a == b


# ---- comparison ----------------------------------------------------------------------------------

stats = {'cases': 0, 'set_mismatch': 0, 'struct_mismatch': 0, 'v1_wrong': 0, 'v2_wrong': 0,
         'v2_raised': 0, 'v1_raised': 0}
notes = []


def compare_case(label, records, v1_fn, v2_fn, *, expect_fail=False, note_limit=6):
    """returns True if v1 and v2 agree as sets (and against the oracle)"""
    stats['cases'] += 1
    try:
        out1 = v1_fn(copy.deepcopy(records))
    except Exception as exc:  # noqa: BLE001
        stats['v1_raised'] += 1
        out1 = exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out2 = v2_fn(records)
    except Exception as exc:  # noqa: BLE001
        stats['v2_raised'] += 1
        out2 = exc
    if isinstance(out1, Exception) or isinstance(out2, Exception):
        if len(notes) < 200:
            notes.append(f'{label}: v1={out1!r} v2={out2!r} records={records!r}')
        return False
    ok = True
    for x in test_points(records):
        want = oracle(records, x)
        got1 = v1_member(out1, x)
        got2 = x in out2
        if got1 != want:
            stats['v1_wrong'] += 1
            ok = False
        if got2 != want:
            stats['v2_wrong'] += 1
            ok = False
        if got1 != got2:
            stats['set_mismatch'] += 1
            ok = False
            if len(notes) < 200:
                notes.append(f'{label}: at x={x!r} v1={got1} v2={got2} oracle={want} records={records!r}')
    return ok


def struct_case(label, records, v1_merge_fn, v2_fn):
    """for the merging strategies: v1's non-empty output pieces vs v2's pieces, exactly"""
    try:
        c1 = v1_canonical(v1_merge_fn(copy.deepcopy(records)))
        c2 = v2_canonical(v2_fn(records))
    except Exception:  # noqa: BLE001
        return None
    if c1 != c2:
        stats['struct_mismatch'] += 1
        if len(notes) < 200:
            notes.append(f'{label} STRUCT: v1={c1!r} v2={c2!r} records={records!r}')
        return False
    return True


# v1 strategies, adapted to one record list
V1 = {
    'run_unsorted_process': lambda recs: v1.run_unsorted_process(recs),
    'run_timsort_no_key[1 list]': lambda recs: v1.run_timsort_no_key([recs]),
    'run_timsort_no_key[split 2]': lambda recs: v1.run_timsort_no_key([recs[::2], recs[1::2]]),
    'run_timsort_merge[split 2]': lambda recs: v1.run_timsort_merge([recs[::2], recs[1::2]]),
    'fast_sweep(sorted)': lambda recs: v1.fast_sweep(sorted(recs)),
    'optimized_sweep(sorted)': lambda recs: v1.optimized_sweep(sorted(recs)),
}
V2 = {'from_pieces': v2_from_pieces, 'Builder': v2_builder}


def run_all(label, records):
    agree = True
    for n1, f1 in V1.items():
        for n2, f2 in V2.items():
            agree &= compare_case(f'{label} {n1} vs {n2}', records, f1, f2)
            struct_case(f'{label} {n1} vs {n2}', records, f1, f2)
    return agree


# ---- hand-picked edge cases ----------------------------------------------------------------------

HAND = {
    'empty list': [],
    'one point': [((1, 0), (1, 0))],
    'gap at 1: [0,1) (1,2]': [((1, 2), (2, 0)), ((0, 0), (1, -2))],
    'touch: [0,1) [1,2]': [((1, 0), (2, 0)), ((0, 0), (1, -2))],
    'touch: [0,1] (1,2]': [((1, 2), (2, 0)), ((0, 0), (1, 0))],
    'overlap+contain': [((0, 0), (10, 0)), ((2, 2), (3, -2)), ((5, 0), (12, -2))],
    'NEG_ZERO end then open start: [0]_-0 (0,1]': [((0, 2), (1, 0)), ((0, 0), (0, -1))],
    'NEG_ZERO with -0.0 values: [-1,-0] (0,1]': [((0.0, 2), (1, 0)), ((-1, 0), (-0.0, -1))],
    '(...,-0) & [0]': [((0.0, 0), (0.0, 0)), ((-1, 0), (-0.0, -2))],
    '(...,0) & [-0]': [((-0.0, -1), (-0.0, -1)), ((-1, 0), (0, -2))],
    'NEG_ZERO start after open end: [0,1) [1,2]_-1': [((1, -1), (2, 0)), ((0, 0), (1, -2))],
    'empty (1,1) between': [((1, 2), (2, 0)), ((1, 2), (1, -2)), ((0, 0), (1, -2))],
    'empty [1,1) between': [((1, 2), (2, 0)), ((1, 0), (1, -2)), ((0, 0), (1, -2))],
    'empty (1,1] alone': [((1, 2), (1, 0))],
    'start eps -2 / end eps 2 (out-of-role eps)': [((1, -2), (2, 2)), ((2, 2), (3, 0)), ((5, -2), (5, 2))],
    'rays with gap at 0': [((0, 2), (INF, 0)), ((-INF, 0), (0, -2))],
    'open at infinities': [((-INF, 2), (INF, -2))],
    'degenerate inf': [((INF, 0), (INF, 0)), ((-INF, 0), (-INF, 0))],
    'int/Fraction/float mix touching': [((1.0, 2), (2, 0)), ((Fraction(1, 3), 0), (1, 0)),
                                        ((Fraction(4, 2), 2), (2.5, -2))],
    'Fraction==float tie at 0.5, gap': [((0.5, 2), (1, 0)), ((0, 0), (Fraction(1, 2), -2))],
    'duplicates': [((0, 0), (1, 0))] * 3,
}

print('== hand-picked ==')
for label, recs in HAND.items():
    ok = run_all(label, recs)
    print(f'{label:50s} agree={ok}')

# reversed and nan records: handled separately (v2 raises by design?)
SPECIAL = {
    'reversed record alone': [((3, 0), (1, 0))],
    'reversed inside a cover': [((3, 0), (1, 0)), ((0, 0), (5, 0))],
    'reversed bridging a gap': [((0, 0), (1, 0)), ((4, 0), (2, 0)), ((3, 0), (5, 0))],
    'nan value': [((math.nan, 0), (1, 0)), ((0, 0), (2, 0))],
}
print('\n== reversed / nan ==')
for label, recs in SPECIAL.items():
    r1 = {}
    for n1 in ('run_unsorted_process', 'run_timsort_no_key[1 list]'):
        try:
            r1[n1] = V1[n1](copy.deepcopy(recs))
        except Exception as exc:  # noqa: BLE001
            r1[n1] = exc
    try:
        r2 = v2_from_pieces(recs)
    except Exception as exc:  # noqa: BLE001
        r2 = exc
    try:
        rb = v2_builder(recs)
    except Exception as exc:  # noqa: BLE001
        rb = exc
    print(f'{label}: v1={r1} | v2 from_pieces={r2!r} | Builder={rb!r}')

# ---- seeded random sweep -------------------------------------------------------------------------

POOL = [-3, -2, -1, 0, 1, 2, 3, Fraction(1, 2), Fraction(-5, 3), 0.5, 2.0, -0.0, 0.25, -INF, INF]
EPS = [-2, -1, 0, 2]


def rand_records(rng, n):
    recs = []
    for _ in range(n):
        a, b = rng.choice(POOL), rng.choice(POOL)
        if b < a:
            a, b = b, a
        recs.append(((a, rng.choice(EPS)), (b, rng.choice(EPS))))
    rng.shuffle(recs)
    return recs


rng = random.Random(20261004)
n_random = 0
disagree_random = 0
for i in range(600):
    recs = rand_records(rng, rng.randint(0, 10))
    n_random += 1
    if not run_all(f'rand#{i}', recs):
        disagree_random += 1
print(f'\n== random: {n_random} record lists x {len(V1)} v1 strategies x {len(V2)} v2 ways; '
      f'lists with any disagreement: {disagree_random}')

# generate_sorted_chunk data (float values, the benchmark's own generator), shuffled
random.seed(7)
gs_dis = 0
for i in range(8):
    recs = v1.generate_sorted_chunk(40, 0) + v1.generate_sorted_chunk(40, 0)
    random.shuffle(recs)
    if not run_all(f'genchunk#{i}', recs):
        gs_dis += 1
print(f'== generate_sorted_chunk x2 (80 records, shuffled) x 8: lists with disagreement: {gs_dis}')

print('\nstats:', stats)
for line in notes[:40]:
    print('NOTE', line)

# ---- sabotage: the comparator must catch a wrong v2 mapping --------------------------------------

saved = dict(stats)


def v2_wrong(records):  # treats NEG_ZERO=-1 at an end as open
    return MI.from_pieces([(s, e, se != 2, ee not in (-2, -1)) for (s, se), (e, ee) in records])


caught = not compare_case('SABOTAGE', [((0, 0), (1, -1))], V1['run_unsorted_process'], v2_wrong)
caught2 = struct_case('SABOTAGE2', [((0, 0), (1, 0)), ((3, 0), (4, 0))],
                      V1['run_unsorted_process'], lambda r: MI.from_pieces([(0, 4)])) is False
print(f'\nsabotage caught: membership={caught} structure={caught2}')
assert caught and caught2
