---
type: Decision
id: D-83
title: strict flags; v1 deleted (Q22)
actor: unrecorded
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c0fd76e3f4c1a6cf86cc46aee1f369c466efeae6ba21a6a79769d229c5235c8e
---

### 2026-10-04 revision: strict flags; v1 deleted (Q22)

the owner, on the v1 parity audit's two questions (`v2-implementation-plan.md` §4):
* **(a) a flag is a bool**: `start_closed`/`end_closed` (and `from_pieces`' and `Builder.add_piece`'s flags) take
  python's bool or numpy's; anything else is a TypeError (`cuts.flag`). they were read by truthiness, so
  `MultiInterval(0, 1, start_closed='no')` was `[0, 1]`, silently; `0`, `1` and `None` are refused too. the same in
  the time classes, for the empty set's flags as well. internal callers pass computed bools and are not checked
* **(b)** the v1 README is kept whole as `references/v1-readme.md` (its "notes:" and "Geminis feedback" were kept
  nowhere else); then H4: `archive/v1/` deleted, with its two differential tests and the `pythonpath` entry
