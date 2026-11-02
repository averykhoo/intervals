---
type: Task
id: IV-6
title: "evaluate_box: evaluate a float corner's exact value once, not three times"
brief: 'Idea, not scheduled (M16e, 2026-09-28): a pure speed change, no dependency'
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [evaluate-box]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 6 of 8 (id `evaluate-box`). The cells below are verbatim.

What: a pure speed change: `applicator.evaluate_box` evaluates a float corner's exact value three times under `OUTWARD`; passing `fn`'s value into the hook would cut it to one, maybe worth as much for arithmetic as the backend, with no dependency

Status / blocker: idea, not scheduled (M16e, 2026-09-28)

Spec: `docs/archive/v2/v2-implementation-plan.md` §2 M16e
