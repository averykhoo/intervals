"""per-insert queryable state: v1 bisect insert (unmerged) / append+sort vs v2 Builder.build() after each add,
10 inserts into 50000 (compare.py benchmark2 shape). run: timeout 110 <python> -u <this file> from the repo root"""
import random, sys, time
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-readme']
import compare as v1
from intervals import kernel, MultiInterval as MI
import sweep_helpers as P
random.seed(2026)
base = v1.generate_sorted_chunk(50000, 0)
new = [((x[0][0] * 0.5, x[0][1]), x[1]) for x in v1.generate_sorted_chunk(10, 0)]
t = time.perf_counter(); v1.run_incremental_bisect(base, new); tb = time.perf_counter() - t
t = time.perf_counter(); v1.run_incremental_sort(base, new); ts = time.perf_counter() - t
b = kernel.Builder()
for p in P.v2_pieces(base):
    b.add_piece(*p)
t = time.perf_counter()
for p in P.v2_pieces(new):
    b.add_piece(*p)
    mi = MI.from_cuts(b.build())
tv = time.perf_counter() - t
print(f'10 inserts into 50000, state queryable after each: v1 bisect {tb:.4f}s (unmerged), v1 append+sort {ts:.4f}s (unmerged), v2 add_piece+build each {tv:.3f}s (merged)')
