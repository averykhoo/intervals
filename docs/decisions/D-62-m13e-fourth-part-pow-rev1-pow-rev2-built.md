---
type: Decision
id: D-62
title: M13e, fourth part (pow_rev1, pow_rev2), built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:258c49e7e1908e4f2e68c1e783bba6eb251cbd38cf3c3095cb857ce2943f63cd
---

### 2026-09-26 revision: M13e, fourth part (pow_rev1, pow_rev2), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 4). `pow_rev1` and
`pow_rev2` moved into "current design" (elementary and step functions: reverse ops; ieee 1788: power
reverse ops). the choices the plan left open, each the most conservative reading, are now current
design too:
* **the library's own pow defines them**, ±inf as points: `pow_rev1([-2], [0, 1])` is `[1, inf]`
  (`inf ** -2` = 0), `pow_rev1([inf], [0])` is `[0, 1)` and `pow_rev2([0], [0])` is `(0, inf]` (`0 **
  inf` = 0); `0 ** y` for y <= 0, `1 ** ±inf` and `inf ** 0` have no value, so they solve nothing.
  1788 has no infinite points and every 1788 vector gives the domain through the input rule, so no
  vector sees the difference
* **the operand order is 1788's** (`b` or `a` first, then `c`, then the domain), and the domain is
  named `x` for the bases and `y` for the exponents, defaulting to `[-inf, inf]`
* **the float rule is pow's, per operand**: a finite float end in either operand makes every end
  float (to nearest, flags kept; outward, a moved end open), as `**` does; part 1's ops take each
  end of `c` on its own (the functions' rule). so `pow_rev1([n], c)` equals `pown_rev(c, n)` on `[0,
  inf]` when `c`'s ends are all exact or all float, not when they mix
* to nearest, an end past the largest double is inf, closed, the library's nearest rule (`M(1e200)
  ** M(1000.0)` is `[inf]` too); `OutwardMultiInterval` keeps it open
* the class, the coercion of a number, the warnings (an empty operand only; a negative base or a
  point with no value warns nothing) and `∩` the domain after rounding are part 1's
* **a bug found while building**: `MultiInterval(2).log(4)` hung, since `elementary._exact_log`
  looked only for an int k with `4 ** k == 2` and ziv's loop never settles on the rational 1/2.
  `log_b x` is now exact wherever rational, through each operand's perfect-power decomposition
  (`elementary._perfect_power`; since M13e's review, 2026-09-27, euclid on the exponents,
  `elementary._log_ratio`, as the root search took minutes on `3 ** 20000 + 1`); `pow_rev2` needs it,
  as every `powRev2` vector's ends are rational
* **a proposed category gains rows**: two `powRev2` vectors expect a hull far looser than the
  tightest (`[entire]` and `[-infinity, 0.0]` where it is `[-inf, -0.5]`); they are rows under
  "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 804 vectors: 802 match in both passes, 2 are those rows. itf1788 now 9269 vectors of 102 ops;
  273 statements of 9 ops skipped, all M13g's (2026-09-26, `tools/itf1788_census.py`)
