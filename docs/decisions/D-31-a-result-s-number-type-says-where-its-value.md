---
type: Decision
id: D-31
title: a result's number type says where its value came from (Q26)
actor: owner
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:89ef5b9d8e621361201c8c3b6cacfee1aeb4e935347f9e823c828afddb4c7b2e
---

### 2026-10-08: D31, a result's number type says where its value came from (Q26)

the owner, 2026-10-08, answering Q26 and the question under it: prefer one type per result ("all float, all
int, all frac") or the most precise type ("int over frac over float")? neither: the session's recommendation,
accepted ("okay with everything, record it"). it states the rule D3 and D29 already follow and fixes the cases
m14b-open found (`HANDOFF.md` row m14b-open; the fix is built before 2.0.0).

* **types come only from the operands.** `int` and `Fraction` mean exact, never rounded; `float` means on the
  double grid, so the next operation on it rounds. per end (D29), python's numeric tower: exact op exact stays
  exact, a float operand makes a float; `/` of exact operands is an `int` when integral, else a `Fraction`,
  never a float (D3, unchanged). a float is never upgraded to exact (`0.1` is not
  `3602879701896397/36028797018963968`) and an exact value is never silently made a float
* **a value inside a piece takes the piece's type**: a result end that is no operand's end (abs's 0, `x ** 2`'s
  0) is a float if the piece has a finite float end, as the step functions already decide per piece. so
  `abs(M(-1.0, 1.0))` is `[0.0, 1.0]` (was `[0, 1.0]`). `0.0` has no rounding error; the type records the grid,
  so `0.0 + 1/3` rounds where `0 + 1/3` does not (the owner: "whether a float can ever produce an int ... 0.0
  is right")
  > **superseded 2026-10-08 by D32**: a constant the function reaches (abs's 0) is exact
* **the step functions are per piece, all of them**: `trunc(M(-2.5, 3))` is all floats, as `floor` is (was
  `{[-2.0], [-1.0], [0.0], [1], [2], [3]}`)
  > **superseded 2026-10-08 by D32**: the step functions give ints
* **a domain constant contributes no type**: a clip at a domain end takes the operand's value there.
  `M(-1.0000000000000002, -1.0).acos()` clips to the point `[-1.0]` and is `[3.141592653589793]` in the nearest
  class, as `M(-1.0).acos()` (was the open `(3.141592653589793, 3.1415926535897936)`: the clip left `[-1, -1.0]`
  and acos ran on its exact cut). the outward class is unchanged (open around pi either way)
  > **superseded 2026-10-08 by D32**: the clip is exact, the result stays open around pi
* **a tie goes to the exact type**: an end reached by an exact and a float value of the same number is exact,
  so `abs(M(-1, 1.0))` and `abs(M(-1.0, 1))` are both `[0.0, 1]` (`[0, 1]` under D32) (were `[0, 1]` and `[0, 1.0]`: the order of the
  operand's ends decided). ends from different values keep their own types: `abs(M(-1, 2.0))` and
  `abs(M(-2.0, 1))` are `[0.0, 2.0]`, `abs(M(-1.0, 2))` is `[0.0, 2]` (the owner: "this is fine and I guess it
  makes sense")
* **why not one type per result**: it rounds free exact information away (`O([1.0, 2]) + Fraction(1, 3)` is
  `(1.3333333333333333, 7/3]`; all-float would open and widen the exact end, and every later step inherits it),
  and it is not inclusion-isotone either (Q19's report). **why not the most precise type**: only among exact
  types (an integral `Fraction` is an `int`, D3); a float made exact claims a precision nobody had
* **Fractions in results**: correct and visible rather than silently rounded; a user who wants floats gives one
  float operand (`M(1.0) / 3`). the README should say so where `/` is introduced, and the to-nearest
  `MultiInterval.rounded()` (row "later") would make the way back one call. long exact iterations grow their
  denominators: the docs should point heavy numerical work at floats
