---
type: Decision
id: D-10
title: names for 1788's interval orders, since `<` and `<=` are pointwise and return a `TruthSet`
actor: owner
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:b3aa9f2501e799cad13a636629736e653556a3d30db69a5b11ae60ee73c3415d
---

| D10 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13c), now in `v2-plan.md` "comparisons" and "set operations and size".** names for 1788's interval orders, since `<` and `<=` are pointwise and return a `TruthSet` (`MI(1,3) < MI(2,4)` is `BOTH`): `A.weakly_less(B)` for `less` (both of A's ends ≤ B's; 1788's own wording, "weakly less than"), `A.strictly_less(B)` for `strictLess`. `interior` is not a method: a new property `B.interior` (the set with every end opened, a set operation in its own right) and the existing `A.within(B.interior)` | `weakly_less`, `strictly_less`, `.interior` | M13c |
