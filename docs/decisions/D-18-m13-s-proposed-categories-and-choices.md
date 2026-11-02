---
type: Decision
id: D-18
title: M13's proposed categories and choices
actor: owner
decided: 2026-09-27
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:23f57eae009e369da8c7f53c8da65be60b7fcd42a16ddfd533b473c970f91f3a
---

| D18 | **decided 2026-09-27 by owner**, on M13's proposed categories and choices: (a) **"tighter than the vector"** is an approved residual category (M13e): 11 keys, 18 vectors where 1788's expected hull is looser than the tightest double enclosure and ours is the tightest, checked with arb or exactly; the two grossly loose `pow_rev.itl:609`, `:642` stay in it. (b) **"exact parsing decides validity"** is approved (M13g): the 1788 text constructors read bounds exactly, so no `PossiblyUndefinedOperation` for a near-tie literal; 7 keys. (c) the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY` 12) stay under **decoration expectations**. (d) `set_dec` **demotes** as 1788's `setDec` does; only the `DecoratedInterval` constructor raises | both categories approved; rows stay; set_dec demotes | M13e, M13g |
