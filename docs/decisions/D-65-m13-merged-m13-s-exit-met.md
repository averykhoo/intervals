---
type: Decision
id: D-65
title: M13 merged, M13's exit met
actor: unrecorded
decided: 2026-09-27
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:3cff69594dc3e7954a2ff794b93d491b48e9c867cbbd43ac0b5a1ec7c00e338e
---

### 2026-09-27 revision: M13 merged, M13's exit met

M13e (the reverse ops) and M13g (decorations, constructors and signals), built in parallel on
branches `m13e` and `m13g`, are merged on `m13-merge` (details in v2-implementation-plan.md, "exit
for M13"). M13 is done: every statement of the 19 files is a vector of an op in `OPS`, and
`tests/itf1788/test_itf1788.py::test_nothing_is_skipped` asserts `SKIPPED` empty. what the merge
itself decided, each the conservative reading:
* **the reverse ops take decorated operands** ("ieee 1788" above, decorated reverse ops): the
  core's set, decorated trv, as 1788 decorates a reverse op's result
  (`multiinterval/reverse.py::_decorated`); a bare `MultiInterval` beside a `DecoratedInterval` is a
  `TypeError`, as for the wrapper's other ops
* **mulRevToPair's better decoration is not built**: 1788 decorates the pair's first interval as
  the decorated division `c / b` where `0 ∉ b`, but its mulRev, the same set's hull, trv. ours is
  one op, `mul_rev`, which cannot be both; it is trv (sound: trv claims nothing). 52 vectors are
  rows under "decoration expectations", on the decoration alone (the set must match). whether to
  add a pair op with 1788's decoration is an owner question
* the counts at M13e and at M13g are replaced by one re-measurement ("counts at M13's merge");
  the two categories proposed at M13e and M13g were approved the same day (next revision up)
