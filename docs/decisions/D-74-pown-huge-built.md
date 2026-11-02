---
type: Decision
id: D-74
title: pown-huge, built
actor: unrecorded
decided: 2026-09-29
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:4eff7dc80ed401c5b6337d680102092f14961c8f01fe451362ee8391f0fd02e4
---

### 2026-09-29 revision: pown-huge, built

* pown (`A ** n` for an integral n, and every path to it: `ieee1788.pown`, `Interval ** n`,
  `DecoratedInterval`, `Dual`, `np.power`) of a float corner with a huge n never finished: the outward
  descriptor built the exact `Fraction(x) ** n`. it now builds it only within `EXACT_POWER_LIMIT` bits
  and otherwise rounds with `elementary.rounded_pow`, the route pow already took, so pown and pow give
  the same doubles; the design, its proof and the tests are plan §2 "pown-huge". "rounding"'s "fn is
  exact" holds for pown up to that limit; past it attainment is against a marker equal to nothing,
  which the proof shows is the exact value's answer
* the nearest class past `|n| = 2 ** 53` is `rounded_pow` to nearest: python's `float ** int` lost the
  parity (a wrong sign) or read an int past the double range as an overflow (`M(0.5) ** 10 ** 400` was
  `[inf]`). up to 2**53 it is python's `float ** int` as before (whether that promises correct
  rounding is open, Q-nearest-libm)
* the descriptor's name is bounded (`pow<a 20001-bit int>`): `f'pow{n}'` raised python's 4300-digit
  ValueError in every class
* exact int/Fraction operands are unchanged and still huge by nature (Q-exact, for the owner)
