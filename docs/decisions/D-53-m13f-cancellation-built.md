---
type: Decision
id: D-53
title: M13f, cancellation, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:e7cf94d2f573672ce7b28a0780a2c474f08a21e1f4fe71313353383b8487c388
---

### 2026-09-26 revision: M13f, cancellation, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13f). D13 moved into "current
design" (arithmetic, ieee 1788, testing), and its new residual category, "cancellation as a
Minkowski difference", is in the divergence table. the choices D13 and the plan left open, each
the most conservative reading, are now current design too:
* **float operands round outward in `OutwardMultiInterval`**, to the tightest enclosure of the exact
  `X`, and to nearest in `MultiInterval`, computed exactly and rounded once like `fma`. the vectors
  require it (`cancel.itl:218`, `:221`: 1788's answer is the hull of the exact difference) and it
  is that class's promise, every `x` that fits is in the result. inward rounding would certify
  `B + X ⊆ A` but break the outward pass of 16 matching vectors and the class's meaning; it is
  **open to the owner** as a separate method or type if a certified inner answer is wanted.
  today exact operands (a float as `Fraction(f)`) give it
* **an empty `B` gives `[-inf, inf]`, for `A = ∅` too**: every `X` has `∅ + X = ∅ ⊆ A`, so the
  largest is the whole line. 1788 answers `∅` for `cancelMinus [empty] [empty]` (and entire for a
  non-empty `A`, which matches); the two `[empty] [empty]` vectors are rows under the new category
  with their own reason. special-casing `∅ ⊖ ∅ = ∅` would contradict "the largest `X`"
* **the infinite points follow the library's `+`**: `inf + -inf` has no value, so `{inf} + [-inf]`
  is empty and `inf` fits any `A` when `B = [-inf]` (`[0, 1].cancel_minus([-inf])` = `[inf]`)
* **methods on the receiver's class**, as `fma`: `OutwardMultiInterval` only when `self` is one; a
  number is coerced to a point; no warning, not even for an empty operand
* the 242 vectors: 148 match in both passes; 94 (47 keys) are rows under the new category, 45 where
  1788 has no answer and 2 for `[empty] [empty]`. itf1788 now 5375 vectors of 69 ops, 4167
  statements of 42 ops skipped
