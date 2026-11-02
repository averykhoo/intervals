---
type: Decision
id: D-64
title: owner answers to the open questions
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:f2bd4faed06ecc5eec847aa8a90816767537ec5e8bbb80b57542e8ef201cde65
---

### 2026-09-26 revision: owner answers to the open questions

the owner answered `HANDOFF.md`'s questions and items on 2026-09-26:
* **Q3 and Q7 confirmed**: the readings of M13c (equal infinite ends as 1788 writes them;
  `.interior` in the reals) and M13d (`0 ** y` for y <= 0 outside the domain, `1 ** ±inf` and
  `inf ** 0` indeterminate, an integral Fraction exponent is pown, acot continuous) stand as
  recorded in their entries below
* **Q4 closed, no inward variant.** 1788 has none either: `cancelMinus` at level 2 is the hull of
  the exact level-1 answer, an outer enclosure, which `OutwardMultiInterval` already gives.
  reopen if a solver needs a certified inner answer; exact operands give one today
* **Q5 on hold** (the time layer, M8): no rush
  > **superseded 2026-10-04**: answered by D4 (a) and D30, the time layer built (M8)
* **Q6: `<<` and `>>` will be ported**; `random_multi_interval` and a public `apply()` are kept as
  to-dos, undecided. v1 applied python's int shifts endpoint-wise (`archive/v1/multi_interval.py::__lshift__`),
  so floats raised; the meaning on real sets (`A << n` as `A * 2**n`, and `>>` as exact division
  or as python's floor) is chosen when built
  > **superseded 2026-10-04**: the shifts were built, then dropped ("v1's leftovers settled; shifts dropped" above);
  > `random_multi_interval` a test helper, `apply()` not public
* **H1**: release 2.0.0 when everything is fully done, which narrows D17's "whenever"
* **H4**: delete `archive/v1/` after v2 is stable
* **H5**: the v1 README's reading list and illustration to-do are kept, moved to
  `references/todo-from-v1-readme.md` so they outlive `archive/v1/`
* **H2**: push `v2` approved
* **Q1, after 1788's context** (in 1788 every signal is a flag, like IEEE 754's: the result is
  returned, empty for bare and NaI for decorated, and the program goes on, since the result itself
  carries the news: empty and NaI propagate). the owner's choice for python, where errors raise:
    * **`UndefinedOperation` raises**, a `ValueError` subclass: a 1788 constructor given invalid
      input (`numsToInterval(2, 1)`, `"[ foo ]"`) stops, as `MultiInterval(2, 1)` already does.
      so NaI, which 1788 makes only from such input, arises only when made on purpose
    * **`PossiblyUndefinedOperation` warns** (an `IntervalWarning` subclass) and returns 1788's
      result. an exact parser can always decide validity, so it may never fire; the one vector
      that expects it (`ieee1788-exceptions.itl:18`) is settled when M13g is built
    * **`IntvlPartOfNaI` raises** (owner, 2026-09-26, after first leaning to a warning): NaI
      propagates like NaN through decorated ops, but `intervalPart` is the one place it stops,
      turning "invalid" into `∅`, which then reads as a real "no values"
    * **no NaI at all** (owner, 2026-09-26, Q8): with both raising, nothing in the library makes
      NaI. its only other uses, a per-element "invalid" in batch work and reading another 1788
      library's `[nai]`, come with numpy or data import, if ever, and it can be added back then.
      the decorated type has com/dac/def/trv, no `ill`, so `IntvlPartOfNaI` cannot arise; the
      statements needing a NaI become rows under a new divergence category, "no NaI: invalid
      input raises", approved with the decision (details: v2-implementation-plan.md M13g)
* **Q2 settled: the reductions keep raising `ValueError`** where 1788 answers the float `nan`
  (they are operations on floats, not intervals, so an empty interval is no answer either);
  `math.fsum` raises on `inf + -inf` too
* **H3**: numpy interop and a gmpy2/mpfr backend are recorded, not built now ("later (not in
  v2.0)" above). the session's suggested first pick when the solver stack starts: Newton's
  method with forward-mode autodiff, the demonstration of what multi-intervals are for. **superseded
  2026-09-27** by the owner's "get the rest of h3 done": numpy interop and the gmpy2/mpfr backend
  were built as M16d and M16e (the 2026-09-28 revisions above)
