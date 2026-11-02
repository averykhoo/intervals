---
type: Decision
id: D-54
title: M13d, power and the rest of the elementary functions, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:6438ed530a784bfacd657cfd4e7bf105e603a27f758e5f0be2ee50c5aba85c3f
---

### 2026-09-26 revision: M13d, power and the rest of the elementary functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13d). D11 moved into "current
design" (arithmetic: power; elementary and step functions: pow, hypot and the twelve functions;
ieee 1788: the counts). the choices D11 and the plan left open, each the most conservative
reading, are now current design too:
* **a Fraction with an integral value is pown**, as D11 says for a float: `A ** Fraction(4, 2)` is
  `A ** 2`. any `MultiInterval` exponent is pow, even a degenerate integral one, as D11 says
* **`0 ** y` for y <= 0 is outside the domain**, dropped with the `DomainClippedWarning` negative
  bases get, not an `IndeterminateResultWarning`: D11 names the domain as x > 0 plus x = 0 where
  y > 0. the one-sided limit `0 ** -1` = inf, which the M12 rule for a domain's end would allow, is
  not taken, since 1788's domain excludes those points outright; so `[0, 1] ** [-1]` is `[1, inf)`
  while pown's `[0, 1] ** -1` is `[1, inf]`
* **±inf in pow are points where the power has a limit** (`inf ** 2` = inf, `(1/2) ** inf` = 0),
  and `1 ** ±inf`, `inf ** 0` are indeterminate points with atan2's treatment: a box that is one of
  them contributes nothing and warns, a larger box keeps the values of its other points
* **the poles at 0 of cot, csc, coth, csch and an odd negative root follow `reciprocal`** (a
  one-sided limit at a piece's end, closed iff the piece holds 0; the point 0 alone empty with an
  `IndeterminateResultWarning`), the plan's "split a piece as tan's do" for the poles inside
* **acot is continuous**, `pi/2 - atan x` with values in (0, pi), fi_lib's reading (its 30 vectors
  are all on positive operands, where the two readings agree); `atan(1/x)`, with a jump at 0 and
  values in (-pi/2, pi/2], is the other convention
* **rootn(n) for every int n other than 0**, 1788's: n < 0 is the root of `1/x`; an even root's
  domain is x >= 0 with `rootn(0, -2n)` = inf at the domain's end; an odd negative root has the
  two-sided pole at 0; `rootn(x, 0)` is a `ValueError`, a non-int degree a `TypeError`
* **hypot is rounded once** from the exact sums of squares, like `fma`; its class is the receiver's
* the 1939 vectors of the 14 ops match in both passes with no divergence row. 41 of them end at or
  are the pole 0 of cot, csc, coth or csch: `[0]` is empty in both, and a piece ending at 0 has the
  same closed hull whether the infinity is attained or not. `0 ** y` for y <= 0 is outside both
  domains (`pow [0.0,0.0] [0.0,0.0] = [empty]`); no vector has `acoth` at ±1, and the input rule
  never gives a closed infinity. itf1788 now 7314 vectors of 83 ops; 2228 statements of 28 ops
  skipped (M13e, M13g)
