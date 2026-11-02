---
type: Task
id: IV-5
title: Fix ziv running past 120 s on an exact corner of about 2M bits
brief: 'Ready, extreme sizes only: a corner within about 2 ** -(its size) of a rounding breakpoint, past EXACT_RESULT_LIMIT'
pri: NEXT
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [pown-ziv]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 5 of 8 (id `pown-ziv`). The cells below are verbatim.

What: an exact corner of about 2M bits within about 2 ** -(its size) of a rounding breakpoint, past `EXACT_RESULT_LIMIT`, runs ziv past 120 s where the old code built the power in milliseconds (`O(3 + 2 ** -2100000) ** 2`, 2026-10-04); pow the same at 2M bits. follow-ups: a near-1 shortcut in `rounded_pow`, or the exact build when ziv passes a precision cap and the build is affordable

Status / blocker: ready, not scheduled; extreme sizes only

Spec: `docs/archive/v2/v2-implementation-plan.md` §2 "owner-answers"; `references/owner-questions-2026-10-03/streams/pown.md` step 6
