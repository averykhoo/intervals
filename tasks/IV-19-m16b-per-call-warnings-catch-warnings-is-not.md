---
type: Task
id: IV-19
title: 'M16b: per-call warnings.catch_warnings is not thread-safe; the pass imports the adapter twice'
brief: Recorded, not addressed; the double import (vectors parsed twice, 7.5 s cold, 2026-09-27) accepted
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 11 of 14. Source tag: M16b. The bullet below is verbatim.

* M16b: the layer's per-call `warnings.catch_warnings` is not thread-safe on python 3.11-3.13 (as
  `decorated.py::_quietly`); recorded, not addressed. the pass imports the adapter a second time as
  `tests.itf1788.test_itf1788` (the vectors parsed twice, 7.5 s cold, 2026-09-27); accepted
