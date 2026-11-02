---
type: Decision
id: D-23
title: numpy interop (M16d)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-28
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:1edbe2d71e5fca6cd6246fcea3219bb67b16da1d9cad6fa677926e9bb4241d15
---

| D23 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q15): (a), (b), (d), (e), (h) as built; changed: (c) `==`/`!=` against an ndarray is elementwise, (f) `fmin`/`fmax` are `minimum`/`maximum`, (g) numpy is in `[test]`; and the methods follow (h)'s rule (a mixed method call returns the class the operators do).** numpy interop (M16d): (a) `__array_ufunc__` on `MultiInterval`, `DecoratedInterval`, `Dual` (`multiinterval/numpy_compat.py`), operator ufuncs as python's operators on our dunders only, the others the method of the same set image, the rest `TypeError`; `__array__` on `MultiInterval` only (a 0-d object array); the array API standard not built (the alternative: an interval-array type); (b) a foreign real is its exact value (a `Rational` by type, else where `float()` would round), alternatives refuse or keep `float()`; (c) an ndarray meeting ours is elementwise into an object array, `==`/`!=` never broadcast; (d) no numpy-named alias methods; (e) `np.invert` the complement; (f) `fmin`/`fmax` not mapped; (g) numpy not in `[test]`; (h) both operands ours in a method ufunc (`hypot minimum maximum arctan2`): the subclass decides, as for the operators | as built | M16d |
