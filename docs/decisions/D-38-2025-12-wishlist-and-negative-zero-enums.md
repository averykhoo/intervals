---
type: Decision
id: D-38
title: 2025-12 wishlist and negative-zero enums
actor: unrecorded
decided: 2025-12-12
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, 2025-12-12, date from git
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:87066a79f0c1a5e0167e0071c9eddbc90a988d00921c5c13f395a9c281584f6d
---

### 2025-12 wishlist and negative-zero enums

* the range will be the affine extended real numbers, meaning support for ±inf along with negative zero, i.e.:
  `[-inf] + (-inf, 0) + [-0, 0] + (0, inf) + [inf]`
* divide by zero is supported, and there will be warnings for indeterminism
* newton's method solver as a test
* generalized function applicator as long as its continuous and differentiable
* forward mode autodiff
* numpy compat via https://data-apis.org/array-api/latest/ or `__array_ufunc__` or https://numpy.org/doc/stable/user/basics.interoperability.html
* consider ieee 1788 compatible decorations, although multi intervals actually support a bit more so not sure if it matters
