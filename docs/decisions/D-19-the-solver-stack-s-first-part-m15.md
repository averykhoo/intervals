---
type: Decision
id: D-19
title: the solver stack's first part (M15)
actor: unrecorded
decided: 2026-09-27
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-27
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:09740c02650fe179a18e2a43418d82302bc422eec204cc0e77599438665afabb
---

| D19 | **decided in the build 2026-09-27 (the session's defaults); confirmed 2026-10-03 by owner (Q11, as built; `tol` documented as absolute, `max_steps` as boxes).** the solver stack's first part (M15): (a) `multiinterval/autodiff.py` (`Dual`, `derivative`) and `multiinterval/solver.py` (`newton`, `Root`) are public and exported from `multiinterval`, not newton as a test only; (b) newton's step runs only where `f` is proved C¹ by decorations (dac or better on the value and the derivative), else the piece is pruned and bisected; (c) the step is `mul_rev`, never `/` (D7); (d) one variable; (e) `tol=1e-10` absolute, `max_steps=10_000` | as built | M15 |
