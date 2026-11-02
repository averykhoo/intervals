---
type: Decision
id: D-16
title: decorations (com/dac/def/trv/ill)
actor: owner
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:990afdbe43a1b4246d0532645220c76f4937d6ee6c03bae6b06d0e2b16bc1c64
---

| D16 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13g), now in `v2-plan.md` "ieee 1788" (`DecoratedInterval`, `UndefinedOperationError`, `PossiblyUndefinedOperationWarning`).** decorations (com/dac/def/trv/ill), NaI and 1788's constructors go in a **separate decorated wrapper type**: the solver stack's (M11), brought forward. the core `MultiInterval` stays undecorated, so `v2-plan.md` "ieee 1788" ("decorations are not in the core") holds. 1788's signals, owner 2026-09-26 (`v2-plan.md` "2026-09-26 revision: owner answers"): `UndefinedOperation` **raises** (a `ValueError` subclass, so it reads like `MultiInterval(2, 1)`'s `ValueError`); `PossiblyUndefinedOperation` is an **`IntervalWarning` subclass** (the result is returned); names chosen when built. **no NaI** and no `ill` (owner 2026-09-26): its statements are rows under a new category, M13g | wrapper type; signals as the owner chose | M13g |
