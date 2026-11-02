---
type: Task
id: IV-1
title: Make x ** 2 and cosh exact at an operand end 0.0
brief: 'Ready (Q27 (c), D35): exact 0 and 1 at an operand end 0.0, as abs(x) ** 2 and cosh(abs(x)) already give'
pri: NOW
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [x2-cosh-zero]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 1 of 8 (id `x2-cosh-zero`). The cells below are verbatim.

What: `x ** 2` and cosh at an operand end `0.0` give the exact 0 and 1, as `abs(x) ** 2` and `cosh(abs(x))` do since m14b-open (`M(0.0, 1.0) ** 2` is `[0.0, 1.0]`, `abs(M(0.0, 1.0)) ** 2` `[0, 1.0]`; cosh `[1.0, ...]` against `[1, ...]`; probed 2026-10-08)

Status / blocker: ready (Q27 (c) yes, D35); not scheduled

Spec: `docs/records.md` "m14b-open", "left"
