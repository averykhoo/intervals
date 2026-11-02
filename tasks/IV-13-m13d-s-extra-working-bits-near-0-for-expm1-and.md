---
type: Task
id: IV-13
title: M13d's extra working bits near 0 for expm1 and log1p are unpinned (speed only)
brief: Removing them keeps every test green, since ziv then doubles the precision itself; recorded, not scheduled
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 5 of 14. Source tag: none (an untagged bullet). The bullet below is verbatim.

* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (archive plan §2 M13d sabotage). recorded, not scheduled
