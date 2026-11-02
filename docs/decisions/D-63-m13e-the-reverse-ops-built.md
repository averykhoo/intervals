---
type: Decision
id: D-63
title: M13e, the reverse ops, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:43daa78842232a486817c2ef4efd2a421d93913b9c4d8b042ed82c3d3a30f341
---

### 2026-09-26 revision: M13e, the reverse ops, built

M13e is complete (2026-09-26; details in v2-implementation-plan.md, M13e, parts 1 to 4 and the
close-out). D12 is now in "current design" whole: elementary and step functions (the reverse ops,
their exact pieces and D12's cap for the periodic ones), empties and warnings (`HullWarning` for
the periodic ones past the cap or over an unbounded piece of `x`) and ieee 1788 (the reverse ops,
reverse multiplication, the periodic and the power reverse ops, and all of them together). the
choices the plan left open are the four parts' entries below; the close-out made none. in short:
* **ten functions in `multiinterval/reverse.py`**, exported from `multiinterval`, each the exact set
  `{t ∈ x : f(t) ∈ c}` (for `mul_rev`, `pow_rev1`, `pow_rev2`, `∃` over the other operand) with the
  library's own f, ±inf points like any other, where 1788 answers its hull; an irrational end is its
  tightest float enclosure, open, and the domain is intersected after rounding
* **the itf1788 vectors**: all 1955 of the 19 ops run in both passes; 1879 match, 52 are
  degenerate infinities (the unary `pownRev` with n < 0 and 0 in `c`), 6 `[nai]` rows move with
  M13g, and 18 (11 keys) are under **"tighter than the vector", still PROPOSED and awaiting the
  owner**: 1788's hull is looser than the tightest, which ours is
* **a pin until M13's exit**: `tests/itf1788/test_itf1788.py::test_only_m13g_ops_are_skipped`, since
  dropping a reverse op from `OPS` turned no test red before it (its statements only become skips)
* itf1788 now 9269 vectors of 102 ops; 273 statements of 9 ops skipped, all M13g's (2026-09-26,
  `tools/itf1788_census.py`). the totals in "ieee 1788" above are M13d's until the merge with
  M13g's branch re-measures them
