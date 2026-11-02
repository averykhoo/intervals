---
type: Task
id: IV-22
title: 'pown-huge: an EXACT_POWER_LIMIT lowered at run time, after import, still stalls'
brief: ops._check_marker_premises guards import only; O(0.5, 1.0) ** 1074 passes a 60 s timeout (review SAB-3); not guarded
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 14 of 14. Source tag: pown-huge. The bullet below is verbatim.

* pown-huge: `ops._check_marker_premises` refuses at import an `EXACT_POWER_LIMIT` below the
  marker proof's floor (36550 bits), but a limit lowered at run time, after import, still stalls
  (`O(0.5, 1.0) ** 1074` past a 60 s timeout; review SAB-3). recorded, not guarded
