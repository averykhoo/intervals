---
type: Decision
id: D-44
title: implementability review
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:f6204d3e91172a68ad9cbb4dd3039c3b73d83ed866f30573195f1d4c2ab63b06
---

### 2026-09-23 revision: implementability review

a review of whether the plan could be built as written. the design stands; these holes would have
been built in, two of them past the planned tests. all are written into "current design".

* **corner-flag rule at infinity** (unsound): add/sub/mul are flat at ±inf, so "closed iff both
  corners closed" excluded attained infinities — `[inf] + (1, 2)` came out `∅`, `[1, inf] + [0, 1)`
  came out `[1, inf)`. infinite result endpoints now always go through the attainment predicate.
  the planned attainment tests ran "on int/Fraction only", and `inf` is a float, so they would not
  have seen it
* **mul splits at zero** (not sharp): `[-1, 1] * [inf]` hulled to `[-inf, inf]` although only
  `[-inf] ∪ [inf]` is attained. sound, so the fuzz passed, and endpoint checks cannot see an
  interior point. an interior-sharpness test is added
* **±inf is exact** whatever its python type, so the exact/float split and the rounding hook
  have a rule for it
* **`-` is arithmetic subtraction** (as v1, `multi_interval.py:1188`); set difference is a named
  method. the 2026-08 dunder list had given `-` to both
* **`==` does not coerce** scalars (hash contract)
* **relations return bool**; only pointwise comparisons return a `TruthSet`
* **warnings**: import-time filters use `append=True`; property tests that hit indeterminate cases
  opt out of the suite's warnings-as-errors
* **modulo with infinite operands** (D8): clip an infinite dividend, follow python for an infinite
  divisor
* **itf1788 output rule** hulls the expected value too
* **division** computed directly, not as `a * (1/b)` (double rounding on floats)
* implementation detail, in the implementation plan: `typing.NamedTuple` refuses a `__new__`
  override (`AttributeError: Cannot overwrite NamedTuple attribute __new__`, executed 2026-09-23 on
  python 3.13.15), so `Cut` normalizes in a subclass of the named tuple
