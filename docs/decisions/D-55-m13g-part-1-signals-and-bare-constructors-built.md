---
type: Decision
id: D-55
title: M13g part 1, signals and bare constructors, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:9597dc4912599da16d3aa4a531fb2eb80a958d3f7c619ffb59bb27422df13fa9
---

### 2026-09-26 revision: M13g part 1, signals and bare constructors, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 1").
the signals as the owner chose them (Q1): `UndefinedOperationError(ValueError)` raises,
`PossiblyUndefinedOperationWarning(IntervalWarning)` would warn and return. `multiinterval/literals.py`
reads 1788's interval literals, a syntax separate from `MultiInterval.parse`, **exactly**: a decimal
is the rational it spells, so validity is decided exactly and the warning is never emitted today.
`text_to_interval` and `nums_to_interval` give a bare `MultiInterval` with an infinite end open.
"no NaI: invalid input raises" is in `REASONS` (D16) with the 52 former `[nai]` rows and every
`isNaI`; **a new category, "exact parsing decides validity", is PROPOSED for the owner** (the 4
vectors expecting `PossiblyUndefinedOperation`, "ieee 1788" above). the decorated type is still to
come
