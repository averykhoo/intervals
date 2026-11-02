---
type: Task
id: IV-17
title: Re-run sabotage rows that were red in their first run; recheck M15's table for stale bytecode
brief: tools/sabotage.py could re-run either now (M16a closing tests; the stale-bytecode hazard M16c found)
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 9 of 14. Source tag: none (an untagged bullet). The bullet below is verbatim.

* sabotage rows red in their first run were not re-run after the closing tests were added (M16a:
  the four closing tests and the review's, which only add red paths); M15's table was not re-checked
  for the stale-bytecode hazard M16c found (`tools/sabotage.py` could re-run either now)
