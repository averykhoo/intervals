---
type: Task
id: IV-7
title: "Not in v2.0: the 'consider / optional / if asked' list"
brief: Each item is 'consider', optional or 'if asked'; not scheduled; the whole list is in the body
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [later]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 7 of 8 (id `later`). The cells below are verbatim.

What: not in v2.0, each "consider", optional or "if asked". from the owner's answers (2026-10-03): `Root`/`RootBox` as frozen dataclasses if a third state appears; an `rtol` beside `tol`; outward fma, `%`, hypot, `cancel_minus` typed per corner (tighter); a to-nearest `MultiInterval.rounded()`; rootn on cbrt's worst-case inputs; a numpy hook for the 1788 layer; an `AllenMatrix` class or a public `allen_pairs`; a strategies module after 2.0; Q20's optional pin. from the design's "later": a direction tag on a degenerate zero piece (if a solver needs `1/(1/[inf]) == [inf]`); an interval array type (the array API); the backend's non-dyadic part (an mpfr ziv loop for the points it declines, until a workload measures them); allen's composition table for the cut reading; vector-mode autodiff (if a measured solve is too slow); 1788's recommended ops not in the layer (`exp2m1`, `exp10m1`, `log2p1`, `log10p1`, `compoundm1`, `rsqrt`, the `*Pi` functions, text and interchange conversions, inf-sup types but binary64). from M8: pandas past `to_pandas()` (an `IntervalIndex` of several pieces, `IntervalArray`)

Status / blocker: not scheduled

Spec: `references/owner-questions-2026-10-03/`; `docs/archive/v2/v2-plan.md` "later (not in v2.0)"; `docs/archive/v2/v2-implementation-plan.md` §2 M8

The list above is one catch-all row, not separate tasks with specs; it is kept whole as one item, to be split only when an owner call or a measured need picks one entry.
