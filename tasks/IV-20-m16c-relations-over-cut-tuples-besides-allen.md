---
type: Task
id: IV-20
title: 'M16c: relations over cut tuples besides allen_relations do not assert normalized operands'
brief: Only allen_relations asserts it, since its wrong answer would be silent and partial; the methods are unaffected
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 12 of 14. Source tag: M16c. The bullet below is verbatim.

* M16c: the other relations over cut tuples in `relations.py` (`before`, `adjoins`, ...) also read
  normalized operands and do not assert it; only `allen_relations`, whose wrong answer would be
  silent and partial, does. the methods are unaffected
