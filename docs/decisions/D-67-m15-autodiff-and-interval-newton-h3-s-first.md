---
type: Decision
id: D-67
title: M15, autodiff and interval newton (H3's first part), built
actor: unrecorded
decided: 2026-09-27
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:dca4bde374bc8c6ee9724e879ed24c3f6d88285be47ba6f5fe0bb381ee0fbde4
---

### 2026-09-27 revision: M15, autodiff and interval newton (H3's first part), built

the owner asked for H3 first; its suggested first pick (2026-09-26 revision below) was built. the
choices the build made, each the session's default, open for the owner (`HANDOFF.md` Q11; D19 in
v2-implementation-plan.md):
* **public, in the package**: `multiinterval/autodiff.py` (`Dual`, `derivative`) and
  `multiinterval/solver.py` (`newton`, `Root`), exported from `multiinterval`, as the 2025-12 layout sketch
  named them (`autodiff.py`, `solver.py`), rather than newton "as a test" only
* **C¹ is proved by decorations** (dac or better on the value and on the derivative), the decorated
  type's use in the solver that the "later" list kept back; where it fails the piece is only pruned
  and bisected. the alternative, trusting the caller that `f` is differentiable, is the silent
  failure `tests/test_solver.py::test_not_c1_would_lose_a_zero` shows
* **the step is `mul_rev`**, the reverse op, never `/`: `[0] / [0]` is `∅` (D7)
* **one variable**: a `Dual` carries one derivative; several variables (a gradient, a jacobian and
  a krawczyk or newton step in n dimensions) are not built. **superseded 2026-09-28**: built as M16a (the 2026-09-28
  revision above), `Dual` untouched
* `tol=1e-10` absolute, `max_steps=10_000`; float midpoints; the magnitude split at a factor of 16
