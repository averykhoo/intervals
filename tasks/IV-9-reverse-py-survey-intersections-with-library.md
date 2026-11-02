---
type: Task
id: IV-9
title: 'reverse.py: survey intersections with library constants (kernel.intersection vs restrict)'
brief: Not surveyed case by case; the tests are green either way; a tie there goes to the exact cut
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 1 of 14. Source tag: m14b-open (2026-10-08). The bullet below is verbatim.

* m14b-open (2026-10-08): `reverse.py`'s intersections with the library's constants (`_NONZERO`, `_REAL_LINE`, the
  branch images) stay `kernel.intersection`, not `restrict`; their tests are green either way, and their `x` is
  the user's set, so a tie there goes to the exact cut. not surveyed case by case
