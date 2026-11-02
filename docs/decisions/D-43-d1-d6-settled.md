---
type: Decision
id: D-43
title: D1–D6 settled
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:eafbb0f1728dd2c6bdbe7e74063ed5ad057f4e531b9b6d0aa0232cf6a7024cb4
---

### 2026-09-23 revision: D1–D6 settled

owner decisions on the review's open list (v2-implementation-plan.md section 0).

* **D1, closure at infinity**: flags propagate; an infinite endpoint is closed iff attained.
  "closure over attainable values and their limits" is gone: it made `1/(-1, 0)` = `[-inf, -1)`,
  which contradicted both `1/[1, inf)` = `(0, 1]` and the involution (`1/(1/(-1, 0))` came back
  `(-1, 0]`)
* **D2, indeterminate corners**: limit along the box, as reciprocal already did at zero. replaces
  "`[-inf, -1] * [0]` is the entire line". this is what forces D7's `∅` for `[0] * [inf]` and
  `[inf] - [inf]`; under the entire-line rule `∅` was allowed but not forced
* **D3**: exact division as Fraction, integral Fractions normalized to int
* **D4**: time layer deferred; v1's `time_interval.py` (and v1 `multi_interval.py` under it) stay
* **D5**: modulo for every sign combination is required before release, as its own milestone; not
  shipped Q1-only
* **D6**: `[a, inf]` literal — already the design, recorded as settled
* **v1 files** (owner, later the same day): archived to `archive/v1/`, never deleted, until v2
  works. this includes `time_interval.py`, which returns on top of the v2 class; the D4 bullet's
  "stay" means "stay in the archive"
