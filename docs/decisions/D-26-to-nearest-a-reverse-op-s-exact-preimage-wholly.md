---
type: Decision
id: D-26
title: to nearest, a reverse op's exact preimage wholly past MAX squeezes to the point `[±inf]`
actor: owner
decided: 2026-09-30
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c95d9eafc917769b114e55fae36c7a8c5430717bdb6e38b34fe64f01a2693c91
---

| D26 | **decided 2026-09-30 by owner** (fuzz-rev-inf): to nearest, a reverse op's exact preimage wholly past MAX squeezes to the point `[±inf]` (IEEE 754 rounds such a value to ±inf; `_widened` reads it as `[MAX, inf]`), and intersecting with `x` after that rounding lost it whenever `x` is open at that infinity: `pown_rev(c, -1, (-inf, -2))` for `c = (-2.2e-309, 0)` was `{}` though its exact answer `(-inf, -4.49e308)` is not empty; the same at a finite double (`sqr_rev([2, 2.0000000000000004], (1.4142135623730951, 2])` was `{}`). options weighed with the owner: (a) `x` meets the preimage before the rounding, 1788's order; (b) keep it and document it; (c) the nearest class saturates an overflow to `(MAX, inf)`, 1788's enclosure rule, no longer python's float | **(a)**, in the reverse ops only (`reverse._keep_squeezed`): a part of the exact answer inside `x` that rounds wholly onto one double is that double, as a point: an end `x` excludes (the case above; the one way a result leaves `x`), or a point of `x` where the rounding kept an end open (`mul_rev(10, (1, 2), [0.1])`, whose exact answer holds the double 0.1, was `{}`, now `[0.1]`; a known loss M13e's tests had worked around by leaving `x` out of their checks). the nearest class keeps IEEE 754 round-to-nearest (as python's float) everywhere; forward ops unchanged: `[inf] & (0, inf)` is still `{}` (documented, README "rounding"). not a 1788 divergence row: the vectors test the outward class, which was already right | `multiinterval/reverse.py`; `tests/test_reverse.py::test_float_operands` (two `@example`s), `::test_exactly_the_points_with_f_in_c` |
