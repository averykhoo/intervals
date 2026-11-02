---
type: Decision
id: D-68
title: M16a, the solver in several variables (H3's second part), built
actor: owner
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:1e71c81a44ed7427cf57932388147a0a5ce9baf34f8bca9e16ab6e9a8a20260e
---

### 2026-09-28 revision: M16a, the solver in several variables (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done". M16a is its stream nd-solver; the choices are the
build's defaults, open for the owner (Q12; D20):
* **n passes, `Dual` untouched**: a gradient or a jacobian is n evaluations of `F`, each with one
  variable seeded. vector mode (a tangent tuple inside `Dual`, one pass) edits M15's pinned chain
  rules; it stays the alternative if a measured solve is too slow (Q12(a))
* **`solve` and `RootBox`** (names, Q12(b)); `Root` stays one set for `newton`. n == 1 delegates to
  `newton`, so one variable keeps M15's behaviour exactly
* **krawczyk proves, gauss-seidel with `mul_rev` narrows**, on the closed hull, with a float
  preconditioner (identity fallback); uniqueness for n >= 2 by krawczyk only (hansen and sengupta's
  own test is not used: its proof was not checked)
* **the simplest point and the inflation, before an unproved box is output**, each measured to be
  needed: without the first, the cusp's `(0, 0)` and `(1, 1)` end unproved; without the second,
  `(x ** 2 - 2, y - 1/4)` proves none of its two zeros (critique B1). the inflation is clipped to the
  box's region, carried on the stack, without which it can claim a box holding no zero
* **the C¹ gate in n variables**: every value and every partial of the n decorated passes dac or
  better, over the closed hull
* **bisection**: wide components first, round robin; then the widest (kearfott's smear measured on
  the prototype, 148 against 146 calls, 345 against 340, not taken)
* **no direction tag** (the solver stack, above)
