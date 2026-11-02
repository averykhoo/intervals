---
type: Decision
id: D-22
title: the per-piece allen matrix (M16c)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-28
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:8329e9db98dab55dffd474c6f30d1f631446b9d44a96d200bef954c150b5ae4f
---

| D22 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q14, as built; every cut-tuple relation asserts normalized operands).** the per-piece allen matrix (M16c): (a) `A.allen_matrix(B)`, a tuple of tuples of `Allen` (rows the pieces of `A`, columns those of `B`), and `relations.allen_matrix` over cut tuples; (b) `A.allen_relations(B)`, the `frozenset` of the relations holding between some pair of pieces, a second public name the H3 row did not list; (c) an empty operand gives `()` / one empty row per piece / `frozenset()`, no raise and no warning; (d) the matrix is the plain `n x m` loop over `allen()` (no dependence on normalized input; the design's ~2-3x faster fill + sweep not taken), the set view an `O(n + m)` sweep that never builds the matrix; (e) methods on `MultiInterval`, functions in `relations.py`, nothing at the top level, not on `DecoratedInterval`, the sparse `(i, j, relation)` view private | as built | nothing (additive: `allen()` and every existing name unchanged); M16c's record |
