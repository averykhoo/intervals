---
type: Decision
id: D-35
title: an extremum known exactly is an exact end; Q27 answered
actor: owner
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:12af889100e4309ada0175c44f5ffbda96beac0ebb462bcf1d42e3645f0e5d01
---

### 2026-10-08: D35, an extremum known exactly is an exact end; Q27 answered

the owner, 2026-10-08, answering Q27: "trig functions can return exact ±1 when they include it yes"; "okay to b
and c and record the principle too"; then, on the rest of (a) as recommended (the last bullet), "yes okay with
all 3".

* **the principle**, the owner's words: "when we take the range of a function and the input domain selects a region
  that is certain to have a min or max at an exact value, we can set that exact value as the endpoint". D32's rule
  (a), "the constants a function reaches", said as a range rule. it holds for the library's own functions, whose
  extrema are known in closed form: sin's, cos's, csc's and sec's ±1, abs's and `x ** 2`'s 0, cosh's 1. it does not
  generalize to an expression, which is evaluated one operation at a time (the owner: "i don't think we can fully
  generalize this across arbitrary functions"). probed 2026-10-08:
  * `x ** 2` on `[-1, 1]` is `[0, 1]`: the 0 is the function's minimum, attained at x = 0 inside the operand.
    `x * x` on the same set is `[-1, 1]`: mul sees two independent operands, never the minimum
  * `x * (1 - x)` on `[0, 1]` is `[0, 1]`, though its range is `[0, 1/4]`, the maximum exact at x = 1/2. reaching
    it would take a range solver and critical points that happen to be rational; not planned
* **"certain" is D32's no ulp contamination**: the extremum's point is proved inside the operand on exact values,
  and the value there is exact by definition, so no rounded step reaches it. this answers the owner's worry ("there
  could have been a missing/additional ulp of error in the unfortunate direction"): a double just short of the
  point gives the float, as D32's pins (`sin(O(0.0, h))` is `[0.0, 1.0)` for `h` the double below pi/2; past it
  `[0.0, 1]`). a float operand is taken as exactly its double: `1.5707963267948966` is not pi/2
* **csc's and sec's ±1** (D33's first bullet) confirmed, as built
* **(c) an operand end at the extremum is certain too**: `x ** 2` and cosh at an operand end `0.0` give the exact 0
  and 1, as `abs(x) ** 2` and `cosh(abs(x))` do (the session's recommendation). `M(0.0, 1.0) ** 2` becomes
  `[0, 1.0]` (today `[0.0, 1.0]`), `M(0.0, 1.0).cosh()` `[1, 1.5430806348152437]` (today `[1.0, ...]`); row
  x2-cosh-zero builds it. not against D32's "an end that is a float operand's own value is a float": the 0 is the
  function's constant, reached, not the operand's value carried
* **(b) D34 confirmed, as built**: the fuzz and gate:gmpy2 phases on `GATE_WORKERS` (default 8), the gate's own
  phases serial, CI unchanged
* **the rest of (a), D33's other three bullets, confirmed as built**, after the owner's questions:
  * `round(A, ndigits > 0)` on a float piece stays floats, as python's `round(2.675, 2)` is `2.67`; Fractions only
    from an exact piece (`round(M(Fraction(1, 9), Fraction(1, 7)), 2)` is `{[11/100], [3/25], [13/100], [7/50]}`).
    the owner asked whether Fractions from a float would be unexpected: they would; the float `0.12` is not
    12/100, the trade python makes
  * the library's own clips (`kernel.restrict`: asin's and acos's domain `[-1, 1]`, the splits at 0, modulo's
    parts) keep the type of an end the operand has; only an end the clip adds is exact: `[-1.0, 2.0]` clipped is
    `[-1.0, 1]`, and `asin([1.0])` is `[1.5707963267948966]`, from the float given
  * a point whose two ends are one exact and one float of the same number is the exact point: `M(1.0, 1)` is
    `[1]`, `M(0.0, 0)` `[0]`, `M(1, 2) & M(2.0, 3.0)` `[2]` (probed 2026-10-08). the owner asked whether to take
    the float instead: no, the set lies within `{2}`, so its value is known exactly. `MultiInterval.from_cuts`
    (cuts already normalized) still refuses the mixed form, and its message is to say why (the owner: "the reason
    should probably also go in the error message"; row from-cuts-reason): today `not a normalized cut tuple:
    (...)`
