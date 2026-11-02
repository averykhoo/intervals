---
type: Decision
id: D-50
title: M13h, the reductions, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:11f814aaa493d29f3a30dd2435641ececa236dc09fdf0526a00fe0e5a99082ec
---

### 2026-09-26 revision: M13h, the reductions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13h). no D row covered it, so
the choices below were made while building, each the most conservative reading of the plan, and
are now "current design" (arithmetic, ieee 1788, testing):
* **exported from `multiinterval`**: `sum_`, `sum_abs`, `sum_sqr`, `dot`, following M13e's plan for
  the reverse ops (`sum_` keeps its underscore so it never shadows the builtin)
* **the direction is a keyword-only string**, `rounding='nearest'` by default, `'down'`, `'up'`:
  no public API took a direction before, and `rounding.py`'s constants stay internal
* **the result is always a float**, also for int and Fraction operands (the plan's "rounded once",
  an exception to "int and Fraction are never rounded", since a reduction is 1788's float op)
* **no value raises**: a `nan` operand, `inf + -inf`, `0 * inf` raise `ValueError` where 1788
  answers `NaN`, as `nan` does in a constructor and as D9 decided for `mid` of the empty set; the
  adapter reads the error as `NaN`. no warning. open to the owner: returning `nan` instead (1788,
  ieee 754 and python's float `sum` do) would be a looser, compatible change later
* the 15 reduction vectors match with no divergence row; itf1788 now 4782 vectors of 58 ops, 4760
  statements of 53 ops skipped
