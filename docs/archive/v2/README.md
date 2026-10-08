# the v2 plans, archived (2026-10-08)

v2's build is done: M1-M16 and every named item after them are built and recorded (a read-only audit,
2026-10-08, checked each record and spot-checked a code symbol for each). what is left before 2.0.0 is in
`HANDOFF.md`, not here. the owner, 2026-10-08: archive the v2 plans; one decisions log, not in HANDOFF.

* `v2-plan.md`: the design. its "current design" is the fullest statement of the library's semantics as built
  for v2, and code comments and tests cite it ("v2-plan.md \"ieee 1788\"", "v2-plan \"arithmetic\""). it is
  frozen: a later change of behaviour is a decision in `docs/decisions.md` and a record in `docs/records.md`,
  and this file is not edited to match.
* `v2-implementation-plan.md`: the milestones, each one's spec and, once built, its record (§2), the order
  (§3) and the v1 -> v2 surface map (§4, the source for 2.0's release notes). code comments citing "plan §2
  M13e" or "plan §4" mean this file.

## what moved out

* the decisions: the D table (D1-D30, this plan's §0) and `v2-plan.md`'s decision log, verbatim, to
  `docs/decisions.md`, the repo's one decisions log. each file keeps a one-line pointer where they were.
* what was still live here (left open, recorded not scheduled, "later"): copied into `HANDOFF.md`'s open items
  and "still owed" on 2026-10-08, each with its line here.
* rules for future work: the testing skill (`.claude/skills/testing/SKILL.md`) and `CLAUDE.md`.

## lines known to be stale (the audit, 2026-10-08; left as they were)

* `v2-implementation-plan.md` "M14 fuzzing (open, ...)" and "what is still open in M14": M14's exit was met
  2026-09-30 (its "the exit's green GitHub run"), and M14-breadth is done.
* M1 and M2 have no done line of their own; §3's "M1 → M2 → ... → M10, all done by 2026-09-25" is their record.
* M11's backlog rows still list M8, H3's rest, H4 and H5 as open: all done (M8, M15, M16, q22-h4).
* pown hangs listed as still owed under M16 (F4) and in M16d/M16e: D28 (2026-10-03) removed them;
  `MultiInterval(2) ** 2 ** 60` and `O(0.5, 1) ** (2 ** 31 - 1)` answer at once (re-run 2026-10-08).
* "Q-exact" and "Q-nearest-libm" under pown-huge became Q17 and Q18, answered by D28.
* §4's "v2 reads flags by truthiness": flags are strict since Q22 (2026-10-04).
* `v2-plan.md`'s preamble and "current design" point at `HANDOFF.md` Qnn items that are all answered
  (`docs/decisions.md`).
