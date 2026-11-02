---
type: Decision
id: D-32
title: 'precision first: what is known exactly is exact (revises D31)'
actor: owner
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c5c812458889075c88defc052bffe4c6bee0503749f9d64292afd25bd98e49a3
---

### 2026-10-08: D32, precision first: what is known exactly is exact (revises D31)

the owner, 2026-10-08, the same day as D31: "abs can produce an exact end from abs [-1.0, 1.0] since 0 is
included, let's prioritize precision where it's possible"; then, asked, the step functions give ints and the acos
clip stays exact (both the session's recommendations). D31's first bullet (types from the operands, python's
tower, `/` as D3), its tie rule and its reasons stand; three of its bullets are superseded (marked there).

* **exact where the value is known exactly, not only carried**: two kinds of result value are exact (an `int` or
  a `Fraction`) whatever the operands' types. (a) the **constants a function reaches**: its interior extrema
  and its domain ends. abs's and `x ** 2`'s 0 and cosh's 1 already are (`abs(M(-1.0, 1.0))` is `[0, 1.0]`,
  `cosh(M(-1.0, 1.0))` `[1, 1.5430806348152437]`, unchanged); sin's and cos's ±1 become so
  (`cos(M(-1.0, 1.0))` was `[0.5403023058681398, 1.0]`, `sin(M(0.0, 2.0))` `[0.0, 1.0]`: their 1 becomes `1`).
  (b) the **integers the step functions list**: `floor`, `ceil`, `trunc`, `round` and the rest of
  `multiinterval.steps` give ints on a float piece, as python's `math.floor` does (`floor(M(-2.5, 3.0))` was
  `{[-3.0], [-2.0], ..., [3.0]}`, becomes `{[-3], [-2], ..., [3]}`)
* **everything else computed from a float operand stays a float** (D29, D31), even when its value happens to be
  exact: `M(1.0) + 1` is `[2.0]`, an end that is a float operand's own value (`abs`'s `1.0` from `-1.0`) is a
  float. the line is what the value depends on: a float operand's value carried through arithmetic or a
  continuous function is float; an integer by definition or a constant of the function is exact
* **a domain clip is exact**: the clip point of `M(-1.0000000000000002, -1.0)` against acos's domain is both the
  operand's `-1.0` and the domain's `-1`, a tie, so exact `-1` (D31's tie rule), and
  `M(-1.0000000000000002, -1.0).acos()` stays `(3.141592653589793, 3.1415926535897936)`, which contains pi,
  as today. it differs from `M(-1.0).acos()`, `[3.141592653589793]` to nearest, where the user gave the float
  point
* **ties as D31**: `abs(M(-1, 1.0))` and `abs(M(-1.0, 1))` are both `[0, 1]` (were `[0, 1]` and `[0, 1.0]`)
* **abs is `(X ∪ -X) ∩ [0, inf]`** (the owner, 2026-10-08: "reframe abs as x + -x then intersected with [0, inf]
  which preserves 0 without any fancy rules"; "+" read as the union, which the owner accepted: the arithmetic sum
  is `[0, 2]` for `[-1, 1]`). an exact identity of sets, and its exact 0 is the cut of the exact constant
  `[0, inf]`: set operations select cuts and never round. it is abs's **definition and a pin, not its
  implementation**: `ops.py::ABS` stays (the shared machinery carries 1788's decorations and `Dual`'s derivative),
  and a property test asserts `repr(abs(X)) == repr((X | -X) & C(0, inf))` in both classes, types included.
  `x ** 2` (as `abs(x) ** 2`) and cosh (as `cosh(abs(x))`) get their exact 0 and 1 the same way. sin's and cos's
  ±1 have no such identity (reached at irrational points): they need the no-ulp-contamination test above.
  probed 2026-10-08: the identity equals today's abs in both classes on 8 shapes but the two ties
* **the tie rule lives in the kernel, not in each function**: `∪` decides a tie by position today
  (`M(-1, 1.0) | M(-1.0, 1)` is `[-1, 1.0]`, swapped `[-1.0, 1]`, 2026-10-08). where `∪`, `∩` or normalization
  meets two cuts of equal value and different type, the exact one is kept; abs, every set operation and
  whatever is built on them follow
* **what m14b-open builds**: the kernel's tie rule; abs's identity pin; sin's and cos's extrema exact; the step
  functions' ints. abs's own code and the acos clip are unchanged
* **no ulp contamination: the test for exact**, the owner's rule and phrase (2026-10-08: "only choose the exact type if we know for sure it
  has no ulp contamination, and the endpoint is the exact value. we could end up at 1.0 by many paths and not all
  guarantee that 1 is at the endpoint"): an end is exact only when it has **no ulp contamination**: its value is
  proved to be that exact number and attained there, decided on exact values as attainment already is. a value
  that merely rounds to it, or that any rounded step reached, is contaminated and stays a float. pins for the build, probed 2026-10-08 (`h = 1.5707963267948966`, the double nearest pi/2, below it):
  `sin(O(0.0, h))` is `[0.0, 1.0)` (pi/2 outside the piece, the max below 1) and stays a float, `[0.0, 1.0]` to
  nearest; `sin(O(0.0, nextafter(h, 2)))` and `sin(O(0.0, 2.0))` hold pi/2, today `[0.0, 1.0]`, and become
  `[0.0, 1]` in both classes
