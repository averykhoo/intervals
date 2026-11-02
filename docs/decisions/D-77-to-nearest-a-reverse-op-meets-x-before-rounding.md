---
type: Decision
id: D-77
title: to nearest, a reverse op meets `x` before rounding (D26)
actor: unrecorded
decided: 2026-09-30
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:9b163d0c07999472bf9441c9ebaeaf2a2b3b2d38be6ac4304637111cbdceaade
---

### 2026-09-30 revision: to nearest, a reverse op meets `x` before rounding (D26)

* the fuzz job found `pown_rev(c, -1, (-inf, -2))` empty to nearest for `c = (-2.2e-309, 0)`, whose
  exact answer `(-inf, -4.49e308)` is wholly past -MAX: the preimage rounds to the point `[-inf]` and
  `x`, open at -inf, dropped it (plan §2 "fuzz-rev-inf"). 1788 intersects with `x` first and then
  encloses (`[-inf, -MAX]` there, no infinite point); python's float rounds such a value to -inf
* chosen (owner): 1788's order in the reverse ops, python's rounding kept: `x` meets the preimage
  before the rounding, so a part of it inside `x` that rounds wholly onto one double is that double,
  as a point: `[-inf]` here, an end `x` excludes (the one case where a result leaves `x`); or a point of
  `x` at an end the rounding kept open (`mul_rev(10, (1, 2), [0.1])` is `[0.1]`, was `{}`). the nearest class stays IEEE round-to-nearest everywhere, so forward ops
  are unchanged (`(M(1e308) * 10) & M.parse('(0, inf)')` is `{}`: `[inf]` is the point inf).
  rejected: saturating an overflow to `(MAX, inf)` in the nearest class (1788's enclosure rule; it
  would stop matching python's float), and keeping the loss as a documented rule
