---
type: Decision
id: D-24
title: the gmpy2/mpfr backend (M16e)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-28
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:9bd273a18fa44ae2b7879c2288a5899f091983262b07e5e2465fdaf77221c5a9
---

| D24 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q16): (a)-(c), (f) as built; changed: (d) `[fast]` pinned to `auto`'s window as `[test]` is, (e) one CI gate job on the forced gmpy2 backend.** the gmpy2/mpfr backend (M16e): (a) the default is the pure path; `MULTIINTERVAL_BACKEND=gmpy2` forces gmpy2 (ImportError if missing or below 2.3 / MPFR 4.2), `auto` takes it if importable and `2.3 <= version < 3`; (b) public surface: the env var and the `[fast]` extra only; `multiinterval.backend.name()` not exported, no setter; (c) non-dyadic points stay pure (no mpfr ziv loop), but atan, acot, atan2's angles and the hook's mixed operands; (d) `gmpy2>=2.3,<3` in `[test]`; (e) CI unchanged: the whole suite on the pure path, `tests/test_backend.py` compares both in every job; no gmpy2 fuzz job; (f) ships in 2.0 as an opt-in, or waits under "later" | as built | M16e |
