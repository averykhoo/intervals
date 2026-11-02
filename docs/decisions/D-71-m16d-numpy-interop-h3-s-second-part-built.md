---
type: Decision
id: D-71
title: M16d, numpy interop (H3's second part), built
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:4d66383d7e86197775a3386a5b6f89f83346340680d8a7a37c321b03fd44019a
---

### 2026-09-28 revision: M16d, numpy interop (H3's second part), built

the owner asked (2026-09-27) for the rest of H3; numpy was one of its five streams. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q15; D23 in
v2-implementation-plan.md):
* **`__array_ufunc__`, not the array API**: the owner's 2025-12 line names "numpy compat via
  data-apis.org/array-api or `__array_ufunc__` or the interoperability page"; a multi-interval is an
  element, not an array, so the hook; the array API would be a new interval-array type (Q15(a))
* **a foreign real is exact** (Q15(b)): a `numbers.Rational` by type, any other real by value where
  `float()` would round it; fixes the outward class for `np.longdouble` on linux and gmpy2's `mpq`
  and `mpfr` operands; a double stays the float it is
* **an ndarray meeting ours is elementwise into an object array**, `==`/`!=` never broadcast
  (Q15(c))
* **no numpy-named alias methods** (`arcsin`, `rint`, ...) on the classes (Q15(d)), so numpy's
  object loops and `np.round` refuse them
* **`np.invert(A)` is the complement `~A`** (Q15(e)); `fmin`/`fmax` not mapped (Q15(f))
* **numpy not in the `[test]` extra** (Q15(g)); the README's numpy section is prose
* **both operands ours in a method ufunc: the subclass decides** (Q15(h)), as for the operators, so
  `np.hypot(M, O)` is outward (the review found the first operand's class decided)
* found on the way and fixed (not choices): M15's `Dual ** r` computed `r - 1` in r's own
  arithmetic, unsound in the outward class with plain python floats (`Dual.variable(O(1e300)) ** 0.1`
  missed its derivative, bare and decorated; `Dual.variable(O(-1)) ** 2.0 ** 60` had its sign
  flipped); a number exponent is now an int (integral) or a point set of u's kind first, which
  also changes the nearest class (an exact derivative for an integral float exponent; pow, not
  pown, for an r like 1e-20 whose float `r - 1` is -1.0)
