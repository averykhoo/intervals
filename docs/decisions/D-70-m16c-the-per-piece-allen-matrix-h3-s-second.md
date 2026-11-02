---
type: Decision
id: D-70
title: M16c, the per-piece allen matrix (H3's second part), built
actor: owner
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:dc45a1aa9d070826d2d838286a6a687118422ec5473e1847ec17868e10c91476
---

### 2026-09-28 revision: M16c, the per-piece allen matrix (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done" (`HANDOFF.md` H3). the per-piece allen matrix
was built as M16c, one of five streams. the choices the build made, each the session's default,
open for the owner (`HANDOFF.md` Q14; D22 in `v2-implementation-plan.md`):
* **two views**, the 2026-08-16 note's "per-piece relation matrix, or the set of relations
  holding between any piece pair": `A.allen_matrix(B)` (nested tuples of `Allen`, not a matrix
  class) and `A.allen_relations(B)` (a `frozenset`). the set view is a second public name the H3
  row did not list; built because the note motivates it and it is the `O(n + m)` form. functions
  over cut tuples in `relations.py`, methods on `MultiInterval`
* **an empty operand gives no rows or empty rows and `frozenset()`**, not `allen()`'s
  `ValueError`: a matrix owes one entry per pair and there are none. matches `before`, `after`,
  `adjoins` of an empty operand (False) and the pointwise comparisons (`NEITHER`)
* **the matrix is the plain `n x m` loop over `allen()`**, the conservative choice: it does not
  depend on the operands being normalized. the design's alternative filled every entry BEFORE or
  AFTER with one cut comparison and overwrote the pairs the sweep visits: faster (~3x at 1000
  pieces the designer measured 2026-09-27; ~2x measured 2026-09-28 on a loaded laptop, M16c's
  record), but wrong on a cut tuple whose pieces are out of order. not taken; the trade is recorded
* **the set view is the merge sweep** (`relations.py::_allen_pairs`) plus two corners, and never
  builds the matrix; pinned by counts (calls to `allen()` and cut comparisons), not time. it needs
  normalized operands and asserts them under `__debug__` (the review, 2026-09-28); the matrix
  takes any cut pairs
* **not on `DecoratedInterval`** (through `.interval`, as `allen`); the sparse `(i, j, relation)`
  view private; nothing at the top level
* not built: allen's composition table over the cut reading ("later")
