---
type: Decision
id: D-20
title: the solver stack's second part (M16a)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-28
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:cce87c17aeabdeb17ae7ef670cfe8dedb8a7ab6b368147761430b5760a6fade5
---

| D20 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q12, as built).** the solver stack's second part (M16a): (a) a gradient or a jacobian is n passes of `F`, `Dual` untouched (not vector mode); (b) names `gradient`, `jacobian`, `solve`, `RootBox`, public and exported from `multiinterval`, `Root` unchanged; (c) `solve` at n == 1 is `newton`; (d) uniqueness by krawczyk only, on the closed hull, with a float preconditioner (identity fallback), narrowing by gauss-seidel with `mul_rev`; (e) before an unproved box is output, its simplest rational point (exactly `[0]`: a unique point) and then krawczyk on the box inflated within its region; (f) wide components bisected first, round robin, then the widest; (g) `tol=1e-10` absolute on the widest component, `max_steps=10_000` boxes; (h) no direction tag | as built | M16a |
