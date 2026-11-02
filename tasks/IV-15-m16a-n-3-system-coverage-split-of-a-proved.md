---
type: Task
id: IV-15
title: 'M16a: n = 3 system coverage, split of a proved-unique box, continuum output'
brief: n = 3 with two zeros costs over 120 s; a continuum costs up to 2n + 1 output boxes per box; output form never asked
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 7 of 14. Source tag: M16a. The bullet below is verbatim.

* M16a: a constructed n = 3 system with two zeros costs more than 120 s, so n = 3 is covered by one
  constructed zero and the sphere only, with no random n = 3 test; whether a faster jacobian
  (Q12(a)) or a tighter form of `F` would change that is not measured. the natural path to a split
  of a box already proved unique was not found (pinned by a monkeypatched `_krawczyk` only). a
  continuum costs up to 2n + 1 output boxes per box of width `tol` (review F3); whether to output it
  differently (one box per connected unproved region) was never asked. `newton`'s own `width <=
  piece.wid() / 2` is left as M15 wrote it (archive plan §2 M16a)
