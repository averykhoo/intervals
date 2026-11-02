---
type: Decision
id: D-13
title: '`cancelMinus(A, B)` is the Minkowski difference, the largest `X` with `B + X ⊆ A`'
actor: owner
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:62c9fcb4719e404c094d068dd5240dc9ea4d12d70c49acb55377d282cdaa0679
---

| D13 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13f), now in `v2-plan.md` "arithmetic" and "ieee 1788".** `cancelMinus(A, B)` is the **Minkowski difference**, the largest `X` with `B + X ⊆ A`, which is defined on any multi-intervals; `cancelPlus(A, B)` is `cancelMinus(A, -B)`. for connected `A` and `B` it is exactly 1788's answer whenever 1788 has one (`[a1 - b1, a2 - b2]` when `wid A ≥ wid B`). where 1788 has no answer it returns entire as a "no answer" signal, and ours is a real set: `cancelMinus([-inf,-1], [-1,5])` = `(-inf, -6]`, and `∅` when nothing fits. those vectors are divergence rows under a **new residual category, "cancellation as a Minkowski difference"**, approved with this decision | Minkowski difference; new divergence category | M13f |
