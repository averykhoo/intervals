---
type: Decision
id: D-42
title: degenerate indeterminate boxes return ∅
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:2013410e881a3ab62516993e7804d4c530a6a2251b480260d5bbaea7903a5b61
---

### 2026-09-23 revision: degenerate indeterminate boxes return ∅

> **amended same day**: D1 and D2 were decided after this entry; see the entry above.

owner decision (D7 in v2-implementation-plan.md). a box that *is* an indeterminate point — `1/[0]`,
`[0] * [inf]`, `[inf] - [inf]`, `[0] / [0]` — returns `∅` and emits `IndeterminateResultWarning`,
where the 2026-09-22 design gave `[-inf] ∪ [inf]` or the entire line.

**why.** inclusion isotonicity (`A ⊆ B ⇒ f(A) ⊆ f(B)`) forces it: `[0]` is inside both `[-1, 0]` and
`[0, 1]`, so `1/[0] ⊆ [-inf, -1] ∩ [1, inf] = ∅`; `[0] * [inf] ⊆ [0] * [5, inf] ∩ [0, 1] * [inf]`,
empty under the sharp corner rule. solvers (bisection, forward-backward contraction) need isotone
ops; it matches 1788's empty and removes the `1/[0]` row from the itf1788 divergence table.

**cost.** `1/(1/[inf])` = `∅`, and `1/x` round-trips only for sets with no degenerate piece at `0`,
`inf` or `-inf` (`[0] ∪ [1, 2]` → `[1/2, 1]` → `[1, 2]`). the direction tag under "later" stays the
recovery path.

**separately.** `f(A ∪ B) == f(A) ∪ f(B)` is false for reciprocal under the direction-from-the-piece
rule with *any* value of `1/[0]`: `A = [-1, 0)`, `B = [0]` gives `1/(A ∪ B)` = `[-inf, -1]` but
`1/A ∪ 1/B` = `(-inf, -1]` (with D1's flag propagation). that law is only `⊇` for reciprocal/div.
