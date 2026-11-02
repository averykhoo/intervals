---
type: Task
id: IV-18
title: 'T1 (tools/sabotage.py): the finally in Run.pytest and the linux kill paths are untested'
brief: The linux kill paths ran but were not sabotaged; on macOS stop trusts the PID alone
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [T1]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 10 of 14. Source tag: T1 (`tools/sabotage.py`). The bullet below is verbatim.

* T1 (`tools/sabotage.py`): the `finally` in `Run.pytest` that kills the child when the wait itself is
  interrupted is untested; the linux kill paths ran but were not sabotaged; on macOS `stop` trusts the PID
  alone (no creation time)
