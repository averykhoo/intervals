# verify construct-4 (reproduce + find a way): empty-by-flags pairs `[1,1)`, `(1,1)`, `(1,1]`, no-arg differing flags

claim under test: UNDOCUMENTED_DIFFERENCE ("only kernel.py::piece docstring records it; no design doc").

## reproduction (own probe)
probe: `.scratch/v1-parity/verify/construct-4/probe_empty_pairs.py`
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/construct-4/probe_empty_pairs.py`

key lines:
```
2-arg [1,1)     v1 ValueError: Interval start (1, 0) is after end (1, -1)  | v2 empty=True {}
2-arg (1,1)     v1 ValueError: Interval start (1, 1) is after end (1, -1)  | v2 empty=True {}
2-arg [2,1]     v1 ValueError ...                                          | v2 ValueError: interval start 2 is after end 1
1-arg (1)       v1 empty=True                                              | v2 empty=True {}
0-arg flags (,] v1 ValueError: (empty message)                             | v2 empty=True {}
0-arg flags (,) v1 empty=True                                              | v2 empty=True {}
'(1, 1)'        v1 merge: ValueError                                       | v2 parse: {}
'(1)'           v1 merge: empty                                            | v2 parse: {}
'[1, 1) | [2, 3]' v1 ValueError                                            | v2 [2, 3]
v1 merge((1,1)) ('err', 'AssertionError: ')      # v1's tuple spelling of (1,1) crashes on a bare assert
sweep (600 seeded, a,b in {-2,-1,0,1,2,1/2,0.5,1.0,-0.0,+-inf}): 0 cases where v1 ok and v2 raises;
  0 membership mismatches where both ok; every a==b v1-raise is v2-empty (the rest are the separate,
  documented INFINITY_IS_NOT_FINITE closed-inf cases)
strict() agrees with v1 on raise/ok: 16/16
DELIBERATE WRONG EXPECTATION CAUGHT: v2 [1,1) is empty
```
the auditor's observation reproduces.

## why it is not an undocumented difference
* **design doc records it**: `v2-plan.md:85`: "empty iff `start >= end`: `[1,1)` and `(1,1)` normalize to empty.
  reversed *values* (`[2,1]`) are a ValueError". the auditor's "no design doc" is wrong.
* also `references/gemini-conversation-recap.md:124-126` ("`(1, 1)`: This is **Empty**, not Error. `[1, 1)`:
  This is **Empty**, not Error. `[2, 1]`: This is **Error**.") and pinned by
  `tests/test_multi_interval.py:53` (`(MultiInterval(1, 1, end_closed=False), '{}'),  # [1, 1) is empty`).
* exact arithmetic: `[1,1) = {x : 1 <= x < 1} = {}`, a well-defined set; v2 returns exactly it. v1's refusal is not
  a different answer, it is no answer, and v1 contradicts itself: `MultiInterval(1, start_closed=False,
  end_closed=False)` and `merge('(1)')` give `{}` (its own "null set" branch), `MultiInterval(start_closed=False,
  end_closed=False)` gives `{}`, but `MultiInterval(1, 1, start_closed=False, end_closed=False)` and
  `merge('(1, 1)')` raise; `merge((1, 1))` (the tuple spelling of the same open pair) dies on a bare AssertionError.
* v2 keeps v1's real error: reversed values `[2,1]` still raise ValueError in the constructor and in parse.
* a way to get v1's refusal back in v2 (RAN, 16/16 agree on finite a==b, every flag combo):
  `m = MultiInterval(a, b, start_closed=sc, end_closed=ec); if not m: raise ValueError(...)`.

## residual (minor)
`MultiInterval(start_closed=False)` (no bounds, mismatched flags): v1 ValueError with an empty message, v2 `{}`.
no doc line found covering flags without bounds (the v2 `__init__` docstring says only "`MultiInterval()` is
empty"; not pinned by any test). no capability is lost (the result is the empty set either way; v1 itself accepted
`(,)` and `[,]`). trivially low impact; if anything is filed, it is a one-line doc/pin, not a gap.

verdict: **DIFFERS_DOCUMENTED** (claim refuted: v2-plan.md:85 records it; v1 was self-inconsistent).
