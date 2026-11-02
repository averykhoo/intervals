---
type: Decision
id: D-51
title: M13b, the numeric functions, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:02975c62ecaf6e5e187b0daa077fc83bedce88e49dbebe54e0978c56e0ec25c7
---

### 2026-09-26 revision: M13b, the numeric functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13b). D9 moved into "current
design" (set operations and size, ieee 1788, testing). the choices D9 and the plan left open, each
the most conservative reading, are now current design too:
* **methods, not properties**: `mid()`, `rad()`, `wid()`, `mag()`, `mig()`, `mid_rad()` (the plan's
  spelling; `inf` and `sup` stay properties)
* **"float" is the operand's kind**: any finite float end makes every number a float, rounded once;
  otherwise int or Fraction. the one exception is D9's ±max float for a half-bounded `mid`
* **`rad` is 1788's radius around the rounded midpoint**, not half the width rounded up, because the
  vectors require it (`rad [1, 1 + 3ulp]` = `2ulp`); `mag` rounds up and `mig` down (exact on
  double ends); both classes give the same numbers, the direction being the function's
* **edge cases no source covered**: a point, `[inf]` included, is its own midpoint with radius and
  width 0; an exact end past max float keeps the midpoint inside the hull; the empty set raises
  `ValueError` naming the missing quantity, with no warning
* **the adapter gained a float pass for the six ops** (`test_vector_float`, both classes, compared
  as they are): the first pass's exact operands would never exercise the library's own rounding
* the 167 vectors match, no divergence row beyond the 6 generated `[nai]` ones; itf1788 now 4949
  vectors of 64 ops, 4593 statements of 47 ops skipped
