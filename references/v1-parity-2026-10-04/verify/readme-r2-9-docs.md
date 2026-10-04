# verify readme-r2-9 (lens: DOCS): incremental bisect-insert cost model

claim: UNCLEAR, v2 lacks a cheap insert into a sorted state (compare.py::run_incremental_bisect, 82x).

## verdict: refuted -> EQUIVALENT_RENAMED (v1 `mi.add(piece)` -> v2 `mi | piece`; batch: `Builder`)

1. the bisect strategy was never a v1 library capability. `archive/v1/compare.py` is a standalone benchmark
   script (`if __name__ == "__main__": benchmark2()`), imported by no v1 module (grep `import compare` over
   archive/v1: none). its own comment says the local merge was not implemented ("Simplified for benchmark").
2. v1's actual incremental insert, `multi_interval.py::MultiInterval.add` -> `update`, extends endpoints and
   calls `merge_adjacent()` with default `sort=True`, which re-sorts ALL pairs every call: the "append + sort"
   strategy compare.py measured as the slow one.
3. probe `readme_r2_9_probe.py` (timeout 120 .../envs/intervals/python.exe), 50000 pieces, 10 inserts:
   `10 library inserts into 50000: v1 MultiInterval.add 1.318s, v2 mi | piece 0.298s`
   `membership disagreements 0 of 833`; deliberately wrong expectation caught 833 times (probe can fail).
   v2 is ~4x faster per queryable insert than v1's library; results agree.
4. docs: the normative current design (README.md: "its 'current design' section is normative") has
   v2-plan.md:1304 `kernel.py ... size; Builder (collect, sort once, sweep)`; matches kernel.py:88 docstring.
   v2-plan.md:2619 ("bisect-insert is 82x faster") sits in the decision log (§ starts line 1588, historical);
   v2-implementation-plan.md:122 (M2 spec) is ambiguous but also says "collect, sort once, sweep".
   surface map v2-implementation-plan.md:4892: `add/discard/pop/remove/clear` -> "gone (immutable); `Builder`
   for incremental construction". no record names a bisect-insert path for insert-then-query; that is adjacent,
   but moot since v1 never had one. the plan doc wording (2619, 122) is stale/inconsistent, a docs nit only.
