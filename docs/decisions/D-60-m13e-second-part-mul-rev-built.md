---
type: Decision
id: D-60
title: M13e, second part (mul_rev), built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:daedbe4f0ec659548f98fa4f477aeddd6198ba6c524f6ceb367811e0297299b8
---

### 2026-09-26 revision: M13e, second part (mul_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 2). `mul_rev` moved
into "current design" (elementary and step functions: reverse ops; ieee 1788: the pair rule). the
choices the plan left open, each the most conservative reading, are now current design too:
* **the library's own `*` defines it**: `t` is in `mul_rev(b, c, x)` iff `{t} * b` meets `c` and
  `t ∈ x`, so `0 * ±inf` (no value) contributes nothing: `mul_rev([inf], [0])` is `∅` and
  `mul_rev([1, inf], [3], [0, 10])` is `(0, 3]`, 0 excluded. 1788 has no infinite points, so no
  1788 vector sees the difference
* **rounding is the division's**: every finite end other than 0 is a quotient of an end of `c` by one
  of `b`, computed by `ops.div`, so it is exact for int and Fraction ends and rounded (to nearest,
  flags kept, or outward, a moved end open) exactly where the division would round it. the other
  candidate, `fma`'s rule (all ends rounded once if any operand has a finite float end), would
  make `mul_rev([3], (-2, 0.0))` differ from `(-2, 0.0) / 3` in its exact end `-2/3`
* **x defaults to `[-inf, inf]`**, as for the other reverse ops; no 1788 vector is touched, since
  the input rule never gives `c` an infinite point and ±inf join a preimage only through one
* **`mulRevToPair` is `mul_rev` under a pair rule** in the adapter, piece by piece (stricter than
  as a union, which the plan allowed); the one exported function serves all three 1788 ops
* the 539 vectors: all match in both passes but the 6 `[nai]` ones (generated rows under
  "decoration expectations", which M13g moves). no new category
