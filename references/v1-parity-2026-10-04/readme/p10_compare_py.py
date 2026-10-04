# archive/v1/compare.py: a benchmark scratch. capabilities in it: (1) merging sorted interval tuples with the
# {-2,-1,0,2} epsilon topology (fast_sweep / optimized_sweep), (2) bulk union of 2 or 10 sorted lists, (3) incremental
# insertion (bisect), (4) Eps.NEG_ZERO. compare against v2: MultiInterval.from_pieces / union(*) / Builder.
from common import *
import time
import compare as cp
import intervals
rng = random.Random(99)
def to_v2(recs):
    return MI.from_pieces((s, e, se == 0, ee == 0) for (s, se), (e, ee) in recs)
def gen(n, grid=True):
    out = []; cur = 0
    for _ in range(n):
        s = cur + (rng.randint(0, 2) if grid else rng.uniform(0.1, 2)); e = s + (rng.randint(0, 2) if grid else rng.uniform(0.1, 2))
        se = rng.choice([0, 2]); ee = rng.choice([0, -2])
        if s == e: se = ee = 0
        out.append(((s, se), (e, ee))); cur = e
    return out
ok = {'fast_sweep': 0, 'optimized_sweep': 0}; bad = []
N = 400
for _ in range(N):
    lists = [gen(rng.randint(0, 8)) for _ in range(rng.randint(1, 4))]
    flat = sorted(r for l in lists for r in l)
    want = MI().union(*(to_v2(l) for l in lists))        # v2 union(*)
    for name, f in (('fast_sweep', cp.fast_sweep), ('optimized_sweep', cp.optimized_sweep)):
        got = to_v2(f(flat))
        # brute membership on a fine grid (ends are integers): every half-integer and integer in range
        pts = [Fraction(k, 2) for k in range(-2, 2 * 60)]
        truth_ok = all((p in want) == any(((s < p) or (s == p and se == 0)) and ((p < e) or (p == e and ee == 0)) for l in lists for (s, se), (e, ee) in l) for p in pts)
        if got == want and truth_ok: ok[name] += 1
        else: bad.append((name, str(got), str(want), truth_ok))
print(f'sweep vs v2 union(*), N={N}:', ok, bad[:3])
# sabotage: a sweep merging open-open touches would be caught
assert to_v2(cp.optimized_sweep([((0, 0), (1, -2)), ((1, 2), (2, 0))])) == MI.parse('[0, 1) | (1, 2]')
assert to_v2([((0, 0), (2, 0))]) != MI.parse('[0, 1) | (1, 2]')
# Builder (incremental) vs bisect-insert-then-sweep
okb = 0
for _ in range(200):
    base = gen(rng.randint(0, 20)); new = gen(rng.randint(0, 10))
    lst = cp.run_incremental_bisect(base, new)
    b = intervals.Builder()
    for (s, se), (e, ee) in base + new: b.add_piece(s, e, se == 0, ee == 0)
    okb += MI.from_cuts(b.build()) == to_v2(cp.optimized_sweep(lst))
print(f'Builder vs bisect insert + sweep: {okb}/200')
# scale: v2 bulk union of two sorted lists of 20000 float pieces (compare.py used 50000; shortened for the time limit)
la, lb = gen(20000, grid=False), gen(20000, grid=False)
t = time.perf_counter(); r_cp = cp.run_timsort_no_key([la, lb]); t_cp = time.perf_counter() - t
t = time.perf_counter(); A, B = to_v2(la), to_v2(lb); t_build = time.perf_counter() - t
t = time.perf_counter(); U = A | B; t_union = time.perf_counter() - t
print(f'scale 2x20000: compare.py sort+sweep {t_cp:.3f}s; v2 from_pieces x2 {t_build:.3f}s, A | B {t_union:.3f}s; same set: {U == to_v2(r_cp)}; pieces {len(U)}')
print('NEG_ZERO:', cp.Eps.NEG_ZERO, '-> v2 has one zero:', MI(-0.0) == MI(0.0), MI(-0.0).cuts)
