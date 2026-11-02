---
type: Task
id: IV-10
title: 'Fuzz gaps a census left: the unbuilt rest of the 2026-10-03 shortlist'
brief: No @given test for DecoratedInterval.log(base), decorated reflected ops and divmod, slow decorated functions, and more
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 2 of 14. Source tag: fuzz gaps a census left (2026-10-03; the rest of its shortlist is built). The bullet below is verbatim.

* fuzz gaps a census left (2026-10-03; the rest of its shortlist is built): no @given test for `DecoratedInterval.log(base)` (random base), the decorated reflected ops and divmod (examples only), the slow decorated functions (pow, hypot, trig) on float operands, exact-operand equality of the outward and nearest classes for about 25 more functions, `Builder` (low value)
