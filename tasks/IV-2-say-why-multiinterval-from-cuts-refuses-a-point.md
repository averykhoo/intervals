---
type: Task
id: IV-2
title: Say why MultiInterval.from_cuts refuses a point of an exact and a float cut
brief: 'Ready (D35): the message is to say why; a sabotage check that the message reaches the user'
pri: NEXT
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [from-cuts-reason]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 2 of 8 (id `from-cuts-reason`). The cells below are verbatim.

What: `MultiInterval.from_cuts` refuses a point of an exact and a float cut of the same number (`(below(2.0), above(2))`) with `not a normalized cut tuple: (...)`; the message is to say why, e.g. "a point is one value: its ends are 2.0 and 2; write [2] or [2.0]" (the owner, Q27; a sabotage check that the message reaches the user)

Status / blocker: ready (D35); not scheduled

Spec: `docs/decisions.md` D35, D33; `kernel.is_valid`
