---
type: Decision
id: D-61
title: M13e, third part (sin_rev, cos_rev, tan_rev), built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:7800f9e6377bf11837a93d07028ec8dae6c6113ab7b2ca0a5472b19c74e0fb52
---

### 2026-09-26 revision: M13e, third part (sin_rev, cos_rev, tan_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 3). D12 for the
periodic ops moved into "current design" (elementary and step functions: reverse ops; ieee 1788:
periodic reverse ops). the choices the plan left open, each the most conservative reading, are now
current design too:
* **a `c` holding `[-1, 1]` is not hulled** for sin and cos: every finite t is a solution, one
  piece, so `sin_rev([-1, 1])` is `(-inf, inf)` with no warning; D12 hulls only answers with too
  many pieces. tan has no such `c` (its poles), so `tan_rev((-inf, inf))` warns
* **the poles and ±inf are in no preimage**, part 1's rule for a point with no value, although the
  set op `tan` attains ±inf around a pole inside a piece: `tan_rev([inf])` is `∅`. a piece around a
  pole keeps it, since it is irrational and the enclosures of the two branches' ends overlap
* **the cap per piece of `x`**, as `steps.step` has it: earlier pieces stay exact, the piece that
  would pass 1000 pieces and every unbounded piece are hulled, one `HullWarning` per call; the
  hull stays inside its piece of `x`
* **the result is the union of every branch's enclosure, then `∩ x`**, so one branch more is listed
  on each side of a piece of `x`, for an `x` that starts inside a neighbour's rounding slack
* a `c` whose gaps are single irrational points (`[-1, 1)` for sin) counts as having gaps: a wide
  `x` is hulled with the warning even though listing would give the same set
* far out, where the doubles are coarser than a period (above about 1.8e16), neighbouring
  enclosures merge into one piece; still an enclosure
* **a proposed category gains rows**: six more `*Bin` vectors (12 with decorated copies) expect an
  end one or two doubles outside the tightest enclosure; ours is the tightest (arb). they are rows
  under "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 136 vectors: 124 match in both passes, 12 are those rows. itf1788 now 8465 vectors of 100
  ops; 1077 statements of 11 ops skipped (`powRev1`, `powRev2`, and M13g's)
