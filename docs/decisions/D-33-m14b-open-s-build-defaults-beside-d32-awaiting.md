---
type: Decision
id: D-33
title: m14b-open's build defaults beside D32 (awaiting the owner)
actor: unrecorded
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:55391a2bb41713784e0c9a9d6a0b41bf9ba296deb915538a7ac2c93a928d0f88
---

### 2026-10-08: D33, m14b-open's build defaults beside D32 (awaiting the owner)

> **confirmed 2026-10-08 by the owner, as built** (Q27 (a), D35); `from_cuts`'s message to say why (row from-cuts-reason)

the build of D32 (`docs/records.md` "m14b-open") met four choices D32 did not spell out; each is the session's
default, asked as Q27 (a).

* **csc's and sec's ±1** are exact where attained, as sin's and cos's: D32 named sin and cos, and its rule (a), "the
  constants a function reaches: its interior extrema", covers them (`sec(M(-1.0, 1.0))` is `[1, 1.85...]`)
* **`round(A, ndigits)` with `ndigits > 0`** on a float piece stays floats: its grid values (0.12) are no doubles,
  so not "integers the step functions list", and python's `round(2.675, 2)` is a float. `ndigits <= 0` gives ints
* **the library's own clips keep the operand's cut on a tie** (`kernel.restrict`): a domain clip, the splits at 0,
  modulo's finite and nonzero parts. D32's tie rule is for set operations; applied to the clip it made `asin([1.0])`
  exact, against D32's `acos([-1.0])` = `[3.141592653589793]`. a clip point the domain adds stays exact
  (`acos([-1.0000000000000002, -1.0])` open around pi, as D32 says)
* **a point of two types is not a representation**: `normalize` makes it exact (D32's tie, within one piece) and
  `kernel.is_valid` refuses it, so `from_cuts` raises on `(Cut(2.0, BELOW), Cut(2, ABOVE))`, which it took before
