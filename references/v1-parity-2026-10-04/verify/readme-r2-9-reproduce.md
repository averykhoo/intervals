# readme-r2-9 (reproduce and find a way): incremental bisect-insert cost model

claim under test: v1 had cheap bisect insert into a sorted state (compare.py::run_incremental_bisect, benchmark2); v2 has
none (Builder appends, build() re-sorts; mi | piece ~85 ms/insert). auditor verdict UNCLEAR.

## result: REFUTED. corrected verdict EQUIVALENT_RENAMED (v1 `mi.add(piece)` / `update` -> v2 `mi | piece`)

1. **compare.py's bisect is a benchmark kernel, not a v1 library capability.** `archive/v1/compare.py::run_incremental_bisect`
   bisect-inserts raw `((v, eps), (v, eps))` tuples into a plain list and never merges them. Its own comment says
   "Local Merge (Simplified for benchmark) ... just benchmark the insertion cost O(N)". The result is an unmerged list,
   not an interval set. Nothing in `archive/v1/multi_interval.py` calls it, and `archive/v1/README.md` never mentions
   bisect, insort, incremental or compare.py (grep came back empty).
2. **v1's real incremental path re-sorts every insert.** `multi_interval.py::MultiInterval.add` calls `update`, which
   extends the endpoints and then calls `merge_adjacent()` with `sort=True`. That runs `sorted()` over all n pairs and
   sweeps them on each call, so v1 never had a bisect-insert path in its library.
3. **On v1's real path, v2 is faster and the results agree** (own probe below).
4. **v2 also has a public composition for a bisect-style insert:** read `mi.cuts`, bisect over `cuts[1::2]` and
   `cuts[0::2]`, splice the merged pair, then call `MultiInterval.from_cuts`. That is an O(log n) locate plus an
   O(n) splice at C speed, plus an O(n) validity check in `from_cuts`.

remaining real nit (documentation only, not a capability gap): two docs still describe a bisect-insert Builder, but the
code sorts once in build():
- v2-plan.md:2619 "incremental building via a small builder (bisect-insert is 82x faster than re-sort per compare.py)"
- v2-implementation-plan.md:122 "`Builder`: collect, sort once, sweep (compare.py: bisect-insert for incremental, timsort for bulk)"
- intervals/kernel.py::Builder: "collect pieces, then sort and sweep once in `build()`"

## evidence

probe: `.scratch/v1-parity/verify/r2-9/probe_incremental.py`
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe -u .scratch/v1-parity/verify/r2-9/probe_incremental.py` (also with `-O`)

correctness:
```
composition vs mi|piece: 2125 inserts, 0 mismatches; vs v1 add (closed, membership): 0 diffs / 300
sabotage check: wrong expectation caught: True
[0, 2] { (0, 1) , (1, 2) } (0, 2)        # [0,1)+[1,2] tiles; (0,1)+(1,2) stays two; point 1 fills the gap
final sets equal: True  v1 vs v2 membership on 2000 pts: True
```
cost: 50000 pieces, 10 inserts, a membership query after each insert (asserts on / `-O`):
```
v1 library MultiInterval.add (its real incremental path)   289.50 / 70.56 ms/insert
v1 compare.run_incremental_bisect (unmerged list, no set)    0.04 /  0.05 ms/insert   <- not a set, no merge
v2 mi | piece                                               37.31 / 26.84 ms/insert
v2 Builder.add_piece + build each                           62.15 / 56.43 ms/insert
v2 composition bisect + from_cuts (public)                  32.29 / 18.98 ms/insert
v2 splice via private _wrap                                 16.91 /  3.60 ms/insert
```
The auditor's figure of 85 ms/insert for `mi | piece` did not reproduce: I measured 27-37 ms. Either way it beats v1's
`add` at 71-290 ms. The only faster row, compare.py, measures a list insert that produces no interval set.

hang note: one first run timed out, but in my probe's final check, not in either library. It ran 8000 v1 `in` queries
on a 50000-piece v1 set with `_consistency_check` enabled. v1 is slow there, not wrong. With fewer points it finished.
