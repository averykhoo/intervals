---
type: Task
id: IV-14
title: M16 long double tests discriminate only where nmant > 52 (CI's linux)
brief: The local env is python 3.13, so a 3.12-only hole would reach CI unseen (as D25's 3.11 one did)
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 6 of 14. Source tag: M16's tests. The bullet below is verbatim.

* M16's tests: the long double cases (`tests/test_numpy_compat.py::test_longdouble_is_exact`, the `longdouble`
  rows of test 2) discriminate only where `np.finfo(np.longdouble).nmant > 52` (CI's linux, never this laptop).
  the local env is python 3.13, so a 3.12-only hole would reach CI unseen (as the 3.11 `Fraction ** interval`
  one did, D25)
