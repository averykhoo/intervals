---
type: Decision
id: D-87
title: separators exactly between items (Q24)
actor: owner
decided: 2026-10-06
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:267fb444a9bf647fdf9aab35ba04a015af458f54ae563c9f963bbb5e2b18279b
---

### 2026-10-06 revision: separators exactly between items (Q24)

the owner, 2026-10-06, answering Q24: a trailing or a doubled separator is refused ("okay yes refuse both"):
`[1,]`, `{1,}`, `{1,,2}` were read as if it were not there; a leading one was already refused. and items with
nothing between them (`[1,2)[3,4)`, v1's form, refused by Q23's build) stay refused, on the session's
recommendation: brackets cannot be part of a number, so the form is unambiguous, but allowing it makes a second
rule ("a separator between numbers, but brackets may touch") and lets white space alone part two pieces again
(`[1,2) [3,4)`), while nothing writes it and v1 is deleted. the rule is one sentence: two items always have
exactly one of `,` `;` `|` `∪` between them (`,` or `;` between a piece's two numbers)
