"""
compare.py's full-scale data (50000 x 2 bulk; 50000 unsorted; 100 incremental into 50000) through v1 and v2,
results compared exactly (piece structure), timings recorded. single runs, not compare.py's 50 / 20 repeats.

run from the repo root:
    timeout 110 C:/Users/user/anaconda3/envs/intervals/python.exe -u .scratch/v1-parity/r2-readme/bench_fullscale.py
"""
import itertools
import random
import sys
import time

sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-readme']
import compare as v1  # noqa: E402
from intervals import kernel  # noqa: E402
from intervals import MultiInterval as MI  # noqa: E402
import sweep_helpers as P  # noqa: E402


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t


random.seed(2026)
N, N_NEW = 50000, 100
list_a = v1.generate_sorted_chunk(N, 0)
list_b = v1.generate_sorted_chunk(N, 0)
data_2 = [list_a, list_b]
unsorted = v1.generate_sorted_chunk(N, 0)
random.shuffle(unsorted)
data_large = v1.generate_sorted_chunk(N, 0)
new_items = [((x[0][0] * 0.5, x[0][1]), x[1]) for x in v1.generate_sorted_chunk(N_NEW, 0)]

# 1. bulk A | B
r1, t1 = timed(v1.run_timsort_no_key, data_2)
r1h, t1h = timed(v1.run_heapq_merge, data_2)
flat = list(itertools.chain.from_iterable(data_2))
r2, t2 = timed(lambda: MI.from_pieces(P.v2_pieces(flat)))
r2u, t2u = timed(lambda: MI.from_pieces(P.v2_pieces(list_a)) | MI.from_pieces(P.v2_pieces(list_b)))
same = P.v1_canonical(r1) == P.v2_canonical(r2) == P.v2_canonical(r2u) == P.v1_canonical(r1h)
print(f'1. bulk 2x{N}: v1 timsort_no_key {t1:.3f}s, v1 heapq {t1h:.3f}s, v2 from_pieces {t2:.3f}s, '
      f'v2 from_pieces|from_pieces {t2u:.3f}s; pieces v1={len(P.v1_canonical(r1))} v2={len(r2)}; identical={same}')

# 2. unsorted user input
r1, t1 = timed(v1.run_unsorted_process, list(unsorted))
r2, t2 = timed(lambda: MI.from_pieces(P.v2_pieces(unsorted)))
r3, t3 = timed(lambda: MI.from_cuts(P.v2_builder(unsorted).cuts))
same = P.v1_canonical(r1) == P.v2_canonical(r2) == P.v2_canonical(r3)
print(f'2. unsorted {N}: v1 run_unsorted_process {t1:.3f}s, v2 from_pieces {t2:.3f}s, v2 Builder {t3:.3f}s; '
      f'pieces v1={len(P.v1_canonical(r1))} v2={len(r2)}; identical={same}')

# 3. incremental 100 into 50000
r1, t1 = timed(v1.run_incremental_bisect, data_large, new_items)
r1s, t1s = timed(v1.run_incremental_sort, data_large, new_items)


def v2_builder_inc():
    b = kernel.Builder()
    for p in P.v2_pieces(data_large):
        b.add_piece(*p)
    for p in P.v2_pieces(new_items):
        b.add_piece(*p)
    return MI.from_cuts(b.build())


def v2_union_inc():
    mi = MI.from_pieces(P.v2_pieces(data_large))
    t = time.perf_counter()
    for p in P.v2_pieces(new_items):
        mi = mi | MI.from_pieces([p])
    return mi, time.perf_counter() - t


r2, t2 = timed(v2_builder_inc)
(r3, t3_inner), t3 = timed(v2_union_inc)
merged_v1 = P.v1_canonical(v1.optimized_sweep(r1))  # v1 leaves the list unmerged; sweep it to compare
merged_v1s = P.v1_canonical(v1.optimized_sweep(r1s))
same = merged_v1 == merged_v1s == P.v2_canonical(r2) == P.v2_canonical(r3)
print(f'3. incremental {N_NEW} into {N}: v1 bisect {t1:.3f}s, v1 append+sort {t1s:.3f}s, '
      f'v2 Builder(all)+build {t2:.3f}s, v2 100 unions {t3_inner:.3f}s (+ base {t3 - t3_inner:.3f}s); '
      f'pieces={len(r2)}; identical (after sweeping v1)={same}')
# sabotage: one extra far-away piece on the v2 side, must differ
bad = MI.from_cuts(P.v2_builder(data_large + new_items + [((-10, 0), (-9, 0))]).cuts)
print('sabotage caught:', P.v2_canonical(bad) != merged_v1)
