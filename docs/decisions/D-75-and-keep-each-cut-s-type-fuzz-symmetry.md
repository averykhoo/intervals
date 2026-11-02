---
type: Decision
id: D-75
title: "`-` and `+` keep each cut's type (fuzz-symmetry)"
actor: unrecorded
decided: 2026-09-29
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:1acd1964f5a02cffdad4b6564de3495df08c19ebc28d4142779fe4d14e0d567e
---

### 2026-09-29 revision: `-` and `+` keep each cut's type (fuzz-symmetry)

* a point can hold one value in two types: `[0, 1/2] & [0.5, 1]` is the point with a float 0.5 below
  and an exact 1/2 above, and the fuzz drew the other way round. the applicator reads a point by its
  low cut alone (`applicator._ends`), so `-` of the fuzz's point gave `[-1/2]` exact, while the reverse ops read each end by its own
  type (`reverse._end`): `pown_rev(-c, -7)` was not `-pown_rev(c, -7)`, found by M14's first GitHub
  fuzz run (plan §2 "fuzz-symmetry")
* `-` is now the cut mirror and `+` the identity, each cut keeping its type, as `-` already did for a
  piece with two ends: exact, an involution, and the `-` `test_trig_rev_symmetry` had to build for
  itself (`reverse.negate`). the other forward ops still read a point at its low cut; the reverse ops
  still read each end by its own type. a mixed point is sound either way; only its rounding differs
