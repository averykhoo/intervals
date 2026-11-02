---
type: Decision
id: D-59
title: M13e, first part (sqr_rev, abs_rev, pown_rev, cosh_rev), built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c17f871e03a55407d0d8fccb4c7df0cd0e432ee81fbfd03a5a8c4736f16c14e4
---

### 2026-09-26 revision: M13e, first part (sqr_rev, abs_rev, pown_rev, cosh_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, in progress). D12's
reading for these four moved into "current design" (elementary and step functions: reverse ops;
ieee 1788: the reverse ops rule). the choices the plan left open, each the most conservative
reading, are now current design too:
* **`x` defaults to `[-inf, inf]` with f's own values at ±inf** (the plan's `x=REALS`), so the
  infinities are points of a preimage like any other: `pown_rev([0], -2)` is `[-inf] ∪ [inf]`,
  1788's `[empty]`. the 1788 vectors this touches are rows under "degenerate infinities"
* **a point where f has no value is in no preimage**: 0 for n < 0, where the library's `1/[0]` is
  empty, even though the set op `1/[-1, 1]` attains ±inf around it. so the result is pointwise,
  and `T ⊆ rev(f(T))` holds for every `T` without such points
* **an empty operand warns** (`EmptySetPropagationWarning`, off by default), as the functions do;
  an empty answer from non-empty operands does not warn: "no solution here" is a normal answer
* **the result class is `OutwardMultiInterval` if either operand is one**, as for the dunders;
  the functions are module-level (`multiinterval.sqr_rev`), not methods, as the plan's signatures say
* **the adapter compares the closed hulls** (the output rule unchanged): 1788's reverse op is the
  hull of the preimage by definition
* **a proposed new residual category, "tighter than the vector" (not yet approved)**: two
  `pownRev` vectors (`rev.itl:276`, `:277`) expect an end one double outside the tightest enclosure
  of `2 ** (1074/7)`; ours is the tightest (checked against arb). they are rows under it, and the
  owner decides whether the category stands
* the 476 vectors: 420 match in both passes, 52 (26 keys) are degenerate infinities of the unary
  `pownRev`, 4 (2 keys) the proposed category. itf1788 now 7790 vectors of 91 ops; 1752
  statements of 20 ops skipped (the rest of M13e, and M13g)
