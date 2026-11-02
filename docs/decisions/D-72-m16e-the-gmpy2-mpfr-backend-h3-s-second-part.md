---
type: Decision
id: D-72
title: M16e, the gmpy2/mpfr backend (H3's second part), built
actor: owner
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:b83c9a5873e9c1e61f724edc1e782f8b0f0e676300b0c043a987110c92ef451f
---

### 2026-09-28 revision: M16e, the gmpy2/mpfr backend (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now" (`HANDOFF.md` H3; plan §2 M15: "gmpy2/mpfr stay out"). built as M16e,
one of M16's five streams. the choices, each the build's default, open for the owner (D24, Q16):
* **the pure path is the default**; gmpy2 only with `MULTIINTERVAL_BACKEND=gmpy2` (forced) or `auto`.
  the design had automatic-when-importable; its critique held that the conservative reading wins:
  the pure path is the reference, the local gate and itf1788 then keep checking it, and a user's
  gmpy2 (linked to whatever MPFR their distribution ships) never changes code paths unasked
* **the backend only picks the double** (the contract above), so it is correct by construction
  wherever it declines, and bit-identical where it answers (the differential)
* **declined where one MPFR call is not one rounding**: non-dyadic inputs (but atan2 of two ints and
  the hook's exact mpq), `log` to a base, `acoth`, `rootn` with n < 0, `k pi + ...`; and short of
  where MPFR's limits bite: `rootn` from `n = 2**31` (a margin: gmpy2 raises from `2**32` on
  windows), an operand past `2**20` bits
* **`gmpy2>=2.3,<3` in `[test]`** (the differential never skips, and CI installs the series `auto`
  takes) and a new extra `[fast]` (`gmpy2>=2.3`, unpinned); no CI change:
  every gate job installs `.[test]`, so `tests/test_backend.py` runs everywhere, and the rest of the
  suite runs on the pure path there as here
* **`auto`'s window is `2.3 <= version < 3`** (the series verified); forced takes any gmpy2 at the floor
