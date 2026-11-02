---
type: Decision
id: D-66
title: owner answers on M13's proposed categories (D18)
actor: unrecorded
decided: 2026-09-27
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:7bbe272ef15219672ddfa69c8a7630fbdc4fa3dc0b2a8d3d069307d8fc33ad28
---

### 2026-09-27 revision: owner answers on M13's proposed categories (D18)

* "tighter than the vector" (M13e: 11 keys, 18 vectors, the two grossly loose `pow_rev.itl:609`,
  `:642` included) and "exact parsing decides validity" (M13g: 7 keys) are approved residual
  categories; the `(PROPOSED)` markers are gone from `REASONS` and from the rows' reasons
* the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY`
  12) stay under "decoration expectations"; `set_dec` keeps demoting as 1788's `setDec` does
* the adapter's `tests/itf1788/test_itf1788.py::is_decorated` unpacked an `Interval` result (a
  NamedTuple) into its fields, so a decoration on the result alone went unseen; 0 vectors were
  affected in what runs (the 103 it missed are all of the six decorated ops, which take the decorated
  path by op), but the census undercounted: 1624 vectors carry a decoration, not 1521 (121 of the
  decorated ops, not 18; measured 2026-09-27). fixed and pinned by
  `::test_is_decorated_sees_the_result_alone` (the old line: 1 red)
