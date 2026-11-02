---
type: Decision
id: D-52
title: M13c, the interval orders and the interior, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:8fc5f3b3c58d3ba79ad10940d50e6a40f7ba7caa117707ece7d27bb10c7e980a
---

### 2026-09-26 revision: M13c, the interval orders and the interior, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13c). D10 moved into "current
design" (comparisons, set operations and size, testing). the choices D10 and the plan left open,
each the most conservative reading, are now current design too:
* **"equal infinite ends count" exactly as 1788 writes it**: two starts at -inf, two ends at +inf.
  a start at +inf or an end at -inf (`[inf]`, `[-inf]`, points 1788 has no interval for) is not
  strictly less than itself; the looser "any equal infinite ends" would make `[inf]` strictly less
  than `[inf]`
* **`.interior` is the interior in the reals**: the infinite ends are opened too, so a closed end
  at ±inf drops out (`[5, inf]` → `(5, inf)`, `[inf]` → `∅`, `[-inf, inf]` → `(-inf, inf)`), the
  plan's "every end opened". the extended reals' interior would keep `(5, inf]`. degenerate
  pieces drop out; the class is kept (`OutwardMultiInterval`)
* **open or closed ends do not matter to the orders** (they are on the ends as values, the hull's),
  so `[0, 2]` is weakly but not strictly less than `[0, 2)`; the orders coerce a number to a
  point and raise `TypeError` on anything else, as `within` does
* **empty sets follow the vectors**: two empty sets are weakly and strictly less than each other,
  an empty and a non-empty one neither; `∅.within(B.interior)` is True
* the 184 vectors match with no rule and no listed row (12 generated `[nai]` rows); itf1788 now
  5133 vectors of 67 ops, 4409 statements of 44 ops skipped
