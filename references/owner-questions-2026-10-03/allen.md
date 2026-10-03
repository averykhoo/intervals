# Q14: M16c's choices (D22), the per-piece Allen matrix

report for the owner, written 2026-10-03 by a read-only advisory agent. sections are appended as
written; a section that is missing was not reached.

sources read: `HANDOFF.md` Q14 and the "still owed" M16c bullet; `v2-implementation-plan.md` §0 D22,
§2 M16c, §2 M8; `v2-plan.md` "comparisons", "2026-09-28 revision: M16c";
`intervals/relations.py` (`Allen`, `allen`, `allen_matrix`, `allen_relations`, `_allen_pairs`);
`intervals/multi_interval.py` (`MultiInterval.allen_matrix`, `.allen_relations`);
`tests/test_relations.py`; `archive/v1/time_interval.py`. probes under
`.scratch/owner-questions/probes-allen/`.

## 0. what is built today

verified by reading the code (`intervals/relations.py` lines 224-333, `intervals/multi_interval.py`
lines 639-668, `tests/test_relations.py` lines 259-508, `tests/test_propagation.py::NOT_ON_THE_WRAPPER`):

* `relations.Allen`: a 13-member `Enum` with `.inverse`; exported at the top level
  (`intervals/__init__.py` line 19). `relations.allen(a, b)` on two contiguous cut tuples, else
  `ValueError`; decided on cuts, so `[1, 2]` and `[2, 3]` OVERLAP and `[1, 2)` MEETS `[2, 3]`.
* `relations.allen_matrix(a, b)`: `tuple(tuple(allen(p, q) for q in pb) for p in pa)`, the plain
  loop; any cut pairs, in any order; an empty operand gives `()` or one `()` per row.
* `relations.allen_relations(a, b)`: `assert kernel.is_valid(a) and kernel.is_valid(b)`, then the
  merge sweep `_allen_pairs` (at most `n + m - 1` calls to `allen`, advancing the piece that ends
  first, both on a tie) plus two corner comparisons for BEFORE and AFTER; returns a `frozenset`;
  `frozenset()` if either is empty.
* `relations._allen_pairs(pa, pb)`: the sweep as an iterator of `(i, j, Allen)`; private.
* `MultiInterval.allen_matrix` / `.allen_relations`: one-line delegations through
  `_coerce_or_raise` (numbers coerce to points, anything else `TypeError`). `MultiInterval.pieces`
  is `tuple(self)`, so `A.allen_matrix(B)[i][j] is A.pieces[i].allen(B.pieces[j])` holds.
* not on `DecoratedInterval`: both names are in `tests/test_propagation.py::NOT_ON_THE_WRAPPER`
  (line 640), beside `allen`, `before`, `after`, ...; a guard test fails if a core name is neither
  on the wrapper nor in that set.
* documented in `README.md` (lines 25-28 doctest, lines 105-107 prose), `v2-plan.md` "comparisons"
  (lines 185-216), the module docstring.
* pins (`tests/test_relations.py`): the matrix against `::allen_loop` on hypothesis pairs; the
  stacking identity `allen_matrix(a1 + a2, b) == top + bottom` on UNnormalized operands
  (`::test_allen_matrix_does_not_need_normalized_operands`, which the record says goes red under
  the fill + sweep); converse = transpose + `.inverse`; self-matrix EQUALS/BEFORE/AFTER; the set
  relations as identities over the matrix; a 13-row table; the empty shapes with warnings as
  errors; the set view's cost by counts (`::test_allen_relations_is_a_linear_sweep`, with a
  deliberate LOWER bound that pins the sweep calling the module-global `allen`;
  `::test_allen_relations_compares_cuts_linearly`, `<= 10 (n + m)` cut comparisons);
  `::test_allen_relations_refuses_unnormalized_operands` (skipped under `-O`).
* HANDOFF "still owed" (line 307): the other relations over cut tuples (`before`, `adjoins`, ...)
  also read normalized operands and do not assert it; only `allen_relations` does. a consistency
  gap, not a bug.
* not built: Allen's composition table over the cut reading (`v2-plan.md` line 1399, "later").

## 1. real uses of Allen relations between multi-intervals

inferred from the literature and from the repo's own stated motivation (`v2-plan.md` line 2311,
the 2026-08-16 note: "allen's algebra natively reasons over relation sets"); no user of
`allen_matrix` exists in the tree outside tests and docs (grep 2026-10-03: `tests/test_relations.py`,
`tests/test_propagation.py`, `README.md`, the two modules).

what Allen's algebra is used for, and what shape each use wants:

1. **qualitative temporal reasoning / constraint networks** (Allen 1983; the IA constraint
   satisfaction literature). variables are convex intervals; a constraint between two variables is
   a SET of the 13 basic relations (a disjunction); inference is composition (`r1 ∘ r2`, a set) and
   intersection of sets; consistency is path consistency. the shape wanted is a `frozenset[Allen]`
   per pair of variables, exactly `allen_relations`'s type, and the composition table. BUT the
   algebra's "set of relations" is an epistemic disjunction ("I do not know which"), while
   `A.allen_relations(B)` is an extensional fact ("these relations hold between some pieces").
   they share a type, not a meaning; feeding the latter into a composition table yields the set of
   relations possible between some piece of A and some piece of C given B, which is a weaker
   statement than people usually want. the composition-table item the plan lists as "later" would
   need to say which reading it computes. so the set view's natural consumer is real but the fit is
   partial.
2. **scheduling / calendars / availability** (the M8 time layer's domain: `DateTimeInterval` as a
   set of busy or free periods; v1's `archive/v1/time_interval.py` exposed only `overlaps(other,
   or_adjacent)` and `overlapping(...)`, lines 314-320). the questions asked are set-level: do these
   two availabilities intersect (`overlaps`), is one inside the other (`within`), what is free
   (`-`, `&`), is a meeting before the deadline (`before`). a per-piece Allen relation is asked
   when one wants to know HOW two particular periods relate ("does shift 3 abut shift 4 or overlap
   it by a minute"), which is `pieces[i].allen(pieces[j])` for a specific pair, or the whole matrix
   for a report of n x m small. sizes: tens of pieces, rarely hundreds (a year of daily busy slots
   is ~365 x k). a 1000 x 1000 matrix is not a calendar question.
3. **interval-based event detection / signal segmentation** (annotation agreement, overlap of
   detected vs ground-truth segments): wants, for each piece of A, the pieces of B it intersects
   and how. that is the SPARSE view `(i, j, relation)` for the non-BEFORE/AFTER pairs, exactly what
   `_allen_pairs` yields (it visits the pairs whose cells intersect, plus at most one neighbour on
   each side), in `O(n + m)`. sizes here can be thousands (audio frames, log events). this is the
   use where the dense matrix is the wrong shape and the set view throws away the indices.
4. **Allen as a finer `overlaps`/`adjoins` for two single pieces**: `allen()` alone.

the honest summary: the set view fits the algebra's type; the sparse view fits the one use case
with large n; the dense matrix fits reporting at small n. none of the three is wrong; the question
is which deserve public names at 2.0.

## (a) the matrix: nested tuples, rename, or an `AllenMatrix` class

**the question.** `A.allen_matrix(B)` is a `tuple[tuple[Allen, ...], ...]`, rows the pieces of
`A`. keep, rename, or wrap in a small class with `.transpose()` / `.converse()`?

**A1. keep nested tuples (built).**
* meaning: a plain Python value; `M[i][j]`; `len(M)` rows; `zip(*M)` transposes; the converse is
  `tuple(tuple(r.inverse for r in row) for row in zip(*M))`; `np.array(M)` is an `(n, m)` object
  array (probe `shape.py`, 2026-10-03: verified, including `(2, 0)` for `A.allen_matrix(EMPTY)`);
  hashable; `==` against the test oracle's tuples.
* pros: zero new type at 2.0; the result of a relation is a value, as `TruthSet`, `Allen` and
  `bool` are; converts to numpy / pandas in one call for the reporting use; doctests print it.
* cons: no method discoverability (`.converse()` must be written out); `EMPTY.allen_matrix(B)` is
  `()`, so the column count is lost (verified: `np.array(()).shape == (0,)`, not `(0, m)`); at 1000
  x 1000 it is 8.0 MB of tuple shells and 1.56 s (probe `trade.py`, 2026-10-03), but that is the
  shape's cost, not the container's.
* when better: when the matrix is a reporting / debugging answer at small `n`, consumed by
  printing, numpy or pandas; when the owner wants no new types before a user exists.

**A2. a small `AllenMatrix` class** (`.rows`, `.shape`, `__getitem__`, `.transpose()`,
`.converse()`, perhaps `.relations` and `.pairs()`).
* meaning: a result type. two variants: (i) eager, wrapping the tuple of tuples; (ii) lazy,
  holding the two piece tuples and computing `[i, j]` on demand with `allen()`, `.relations` by the
  sweep, `.pairs()` as the sparse view. (ii) unifies (a), (b), (d) and (e): `O(1)` to build,
  `O(n + m)` for the set and the sparse pairs, `O(nm)` only if someone asks for every entry.
* pros: keeps the shape on empty operands; discoverable converse; the lazy variant makes the
  1000-piece case a non-issue without choosing between the dense and the set forms; the natural
  home for a future composition table's inputs.
* cons: a new public type with repr, `__eq__`, `__hash__`, pickling, its own tests and docs, at
  2.0, with no user asking; equality semantics to decide (equal to a tuple of tuples or not);
  invites numpy-like expectations (slicing, boolean masks) the library will not meet; the lazy
  variant hides a `Θ(nm)` cost behind an attribute.
* when better: if the composition table, the time layer (M8) or an actual user needs `.converse()`
  and shape-preservation before 2.0; or if the owner wants ONE Allen entry point rather than two
  or three names.

**A3. rename** (`allen_table`, `allen_grid`, ...). no gain: "matrix" says rows x columns, which it
is; the only confusion "matrix" could cause (numpy) is answered by `np.array(M)` working.

**A4 (not in the plan). drop the dense matrix, keep only `allen()` per pair and the set / sparse
views.** the smallest surface; the H3 row named the matrix, the README documents it, and the
reporting use at small `n` is real. not recommended, but if the owner wants one name only, the set
view is the one with the better cost profile (see (b)).

**recommendation: A1, keep nested tuples, no rename. confidence ~75%.** the dense matrix is a
small-`n` reporting shape; a type for it is premature without a user, and the two conveniences a
class would add are one-liners on a tuple. a note worth adding to the docstring: the converse
identity and `np.array(M)`.
* what would change my mind: a composition table or M8 landing before 2.0 and needing the shape or
  the converse; then build A2 lazy, and make `allen_relations` its `.relations`.
* cost of changing later: before 2.0, free. after 2.0, changing the return type to a class is
  breaking UNLESS the class subclasses `tuple` (then `M[i][j]`, `len`, `zip(*M)`, `==` to a tuple
  all still hold, and `.converse()` is additive). so even after 2.0 the A1 -> A2 move stays cheap
  if A2 is `class AllenMatrix(tuple)`. this is the main reason not to decide A2 now.

## (b) the set view `allen_relations`: keep or drop; the name

**the question.** `A.allen_relations(B)` returns the `frozenset` of the 13 relations holding
between some piece of `A` and some piece of `B`, in `O(n + m)`. the H3 row named only the matrix;
the 2026-08-16 note (`v2-plan.md` line 2311) said "matrix, or the set of relations". keep or drop;
if kept, `allen_relations` or `allen_set`?

**B1. keep as built.**
* pros: the only Allen operation that scales (13.5 ms at 1000 x 1000 vs 1.56 s for the matrix,
  probe 2026-10-03); its type is what Allen's algebra manipulates (relation sets), so it is the
  input a composition table would take; already pinned (counts, not times; the matrix identity;
  the corners), documented in the README and the module docstring.
* cons: a second public name for one idea; the type matches the algebra but the meaning differs
  (extensional "these hold between some pieces" vs the algebra's epistemic "one of these holds",
  see §1), which a user from the constraint-reasoning world may trip on; it alone among the
  relations asserts normalized operands (HANDOFF "still owed", a consistency wart); the cost pin's
  lower bound pins an implementation detail (its own docstring says so).
* when better: whenever a user has many pieces, or wants a yes/no "does any piece of A MEET any
  piece of B" without the matrix; whenever the composition table is a live plan.

**B2. drop it.**
* meaning: users write `frozenset(r for row in M for r in row)`, `Θ(nm)`.
* pros: one name; two tests with deliberate implementation pins go away; no assert inconsistency.
* cons: loses the only `O(n + m)` Allen answer; the corner logic (BEFORE / AFTER hold iff the
  first piece of one is before the last of the other) is the non-obvious part and would be lost
  with it; the README and plan already document it.
* when better: if the owner wants the smallest 2.0 surface and expects to never build the
  composition table. adding it back later is free (additive), so B2 is the "if in doubt" choice.

**B3. keep, rename `allen_set`.** shorter; but "set" is the library's core noun (set operations,
"a set of reals"), so `allen_set` reads as an operation producing a set of reals. `allen_relations`
has its own mild ambiguity ("Allen's relations" = the enum), but in context (a method with an
operand) it reads right. no rename.

**B4 (not in the plan). replace the set view by a public sparse view** (`A.allen_pairs(B)`, the
`(i, j, relation)` triples; see (e)). the set is then a comprehension over it PLUS the two corners,
which users would get wrong. if the sparse view is made public it should be beside the set view,
not instead of it.

**recommendation: B1, keep, keep the name. confidence ~65%.** it is cheap, pinned, documented and
the one scalable form. two small fixes to make while 2.0 is open, both non-breaking: (i) one
sentence in the docstring saying the set is extensional (which relations hold between some pair),
not a disjunction; (ii) settle the assert inconsistency (HANDOFF line 307) one way: either every
cut-tuple relation asserts `kernel.is_valid` under `__debug__`, or none does and `allen_relations`
documents "normalized operands" like the others. my preference: assert in all of them (the
methods never reach it; the cost is a few comparisons under `__debug__`), because
`allen_relations` is the one whose wrong answer is silent and partial, and the others' are at
least wrong-not-partial; but consistency matters more than which.
* what would change my mind: the owner saying the composition table will never be built AND
  wanting the smallest surface; then drop it (B2), and re-add if asked.
* cost of changing later: dropping after 2.0 is a deprecation cycle; adding is free. so the
  asymmetry argues for B2 if genuinely in doubt; I lean B1 because it is already built and useful.

## (c) an empty operand

**the question.** `EMPTY.allen_matrix(B)` is `()`, `A.allen_matrix(EMPTY)` is one `()` per piece
of `A`, `allen_relations` is `frozenset()`; no raise, no warning. `allen()` on an empty operand
raises `ValueError`.

**C1. keep (built).**
* pros: a matrix owes one entry per pair and there are none, so the shapes `len(M) == len(A)` and
  `len(M[i]) == len(B)` hold uniformly (verified `np.array(A.allen_matrix(EMPTY)).shape == (2, 0)`);
  consistent with every other relation on the class (`before`, `after`, `adjoins` of EMPTY are
  False, `<` is NEITHER) and with the library treating EMPTY as an ordinary value (D7); callers
  iterating pieces need no special case.
* cons: `EMPTY.allen_matrix(B)` is `()`, so `m` is lost (a `(0, m)` shape is not recoverable);
  `allen()` raising while `allen_matrix` of the same operands returns `()` is a difference a user
  may notice, though a defensible one (`allen` owes exactly one relation).
* when better: always, unless the owner wants EMPTY to be an error in relations generally.

**C2. raise `ValueError`, as `allen()`.**
* pros: consistent with `allen()`; makes an empty operand loud in a pipeline.
* cons: inconsistent with the other six relation methods and the pointwise comparisons; forces an
  `if A.is_empty` before every call; nothing in the library raises on an empty set in a relation.
* when better: only if the owner changes the relations wholesale to raise on EMPTY (not on the
  table).

**C3. warn.** the plan rejected it (`v2-plan.md` line 196-197: "not an op on a set"); the
library's warnings are for operations whose result loses something, not for relations. no.

**C4 (not in the plan). a 14th `Allen` member for "no relation".** breaks JEPD-of-13, the
`inverse` table and 1788's reading. no.

**recommendation: C1, keep. confidence ~85%.** it is the one consistent with the rest of the
relations and with the matrix's shape contract. optional: document that `()` loses `m`.
* what would change my mind: a wholesale decision that relations raise on EMPTY.
* cost of changing later: value -> raise after 2.0 is breaking; raise -> value is lenient. the
  built choice is the one that is cheap to live with, so decide it now and do not revisit.

## (d) the plain `n x m` loop vs the fill + sweep

**the question.** `relations.allen_matrix` is the plain loop, `n m` calls to `allen()`, on any cut
pairs in any order. the design's alternative fills every entry BEFORE or AFTER by one cut
comparison and overwrites the pairs the sweep visits; ~2-3x faster, but wrong on out-of-order cut
tuples, so it needs normalized operands (the methods always have them; the function would assert,
as `allen_relations` does).

**measured 2026-10-03** (probe `.scratch/owner-questions/probes-allen/trade.py`, the record's
`fill_sweep` verbatim, best of 5 (of 2 at 1000), shared laptop, `n = m` pieces `[2k, 2k+1]` vs
`[2k+1/2, 2k+3/2]`; the record's 2026-09-28 table had 3788 / 1942 / 22.6 ms at 1000):

| n = m | matrix, plain loop (built) | matrix, fill + sweep | relations, sweep (built) | sparse `list(_allen_pairs)` | `MI.allen_matrix` method |
|---|---|---|---|---|---|
| 2 | 0.009 ms | 0.013 ms | 0.017 ms | 0.009 ms | 0.009 ms |
| 5 | 0.036 ms | 0.042 ms | 0.039 ms | 0.025 ms | 0.034 ms |
| 10 | 0.112 ms | 0.107 ms | 0.074 ms | 0.050 ms | 0.110 ms |
| 20 | 0.400 ms | 0.322 ms | 0.156 ms | 0.108 ms | 0.415 ms |
| 50 | 2.29 ms | 1.53 ms | 0.356 ms | 0.250 ms | 1.95 ms |
| 100 | 7.86 ms | 4.85 ms | 0.635 ms | 0.464 ms | 9.19 ms |
| 300 | 75.0 ms | 46.6 ms | 2.29 ms | 1.66 ms | 74.6 ms |
| 1000 | 1560 ms | 874 ms | 13.5 ms | 9.03 ms | 1606 ms |

* the fill + sweep is 1.8x at 1000, 1.6x at 100, nothing below 20 pieces (it is slower at 2-5).
* at n = 100, the `n m` pair loop with no `allen()` call is 0.57 ms against 7.9-13.8 ms with it
  (two runs, noisy laptop): ~95% of the plain loop IS the `allen()` call (a Python call plus up to
  6 `Cut` comparisons). the fill replaces that with one comparison per skipped pair; the floor is
  the `n m` list build itself, so no fill variant gets below ~0.5 ms at 100 or ~50 ms at 1000.
* a 1000 x 1000 matrix is 8.0 MB of tuple shells (entries are the 13 shared Enum members).
* the fill + sweep passes the 13 pinned table rows both ways on normalized operands (probe
  asserted it), as the record said.

**D1. keep the plain loop (built).**
* pros: simplest; any cut pairs in any order (pinned by `::test_allen_matrix_does_not_need_normalized_operands`
  and `::test_allen_matrix_of_unordered_overlapping_pieces`); every entry is literally `allen()`,
  so the matrix can never disagree with it; `Θ(nm)` is the size of the answer anyway; at realistic
  sizes (<= 50 pieces) under 2.3 ms, the fill would save under 0.8 ms.
* cons: 1.6 s at 1000 x 1000.
* when better: now, with no user at hundreds of pieces; whenever "every entry is `allen()`" is
  worth more than 1.8x.

**D2. the fill + sweep in `relations.allen_matrix`**, asserting normalized operands.
* pros: 1.8x at 1000; the "any order" freedom has no caller outside the tests (grep 2026-10-03:
  only `multi_interval.py` calls `relations.allen_matrix`, with normalized cuts), and every other
  cut-tuple function in the library assumes normalized input, so dropping it is consistent, not a
  loss.
* cons: two of the pinned tests go (or flip to "refuses unnormalized"); the matrix's correctness
  now depends on the sweep's invariant (the skipped pairs are BEFORE or AFTER), a second place to
  get wrong; 1.8x on an answer nobody can consume at that size (a million entries to read) is not
  the bottleneck: a user at 1000 pieces wants `allen_relations` (13.5 ms) or the sparse view.
* when better: a measured user who needs the DENSE matrix at 300-1000 pieces (a heatmap, an
  agreement matrix) and finds 75 ms-1.6 s too slow.

**D3 (not in the plan). hybrid**: the method uses the fill + sweep (its operands are always
normalized), the function keeps the plain loop. two code paths for one answer; not worth it.

**D4 (not in the plan). the lazy matrix** (A2 variant (ii)): nothing is computed until indexed;
the set and the sparse view are `O(n + m)`; the dense form `Θ(nm)` only if asked for. the
asymptotically right answer, but it is a type decision ((a)), not a loop decision.

**D5 (not in the plan). micro-optimise the plain loop** (inline `allen`'s comparisons, cache
`Allen` members). maybe 1.3x; the matrix has no "calls the module-global `allen`" pin (only
`_allen_pairs` does), so it is allowed; not worth the readability.

**recommendation: D1, keep the plain loop. confidence ~80%.** the 2-3x does not matter at
realistic sizes (sub-millisecond either way under 50 pieces; 3 ms apart at 100), and at the sizes
where it would matter the dense matrix is the wrong tool and the set view is 100x faster. this is
also the sub-item with the lowest cost of being wrong.
* what would change my mind: a real user of the dense matrix at hundreds of pieces; then D2 with
  the assert (the "any order" property has no user to protect).
* cost of changing later: zero API cost before or after 2.0 — the result is identical on
  normalized operands, and `relations.allen_matrix`'s acceptance of unnormalized cuts is a
  behaviour of a non-exported module function, not of the public method. decide D1 now and let a
  user's measurement reopen it.

## (e) the surface: methods, top level, `DecoratedInterval`, the sparse view

**the question.** methods on `MultiInterval`, functions in `relations.py`, nothing at the top
level, not on `DecoratedInterval`, the sparse `(i, j, relation)` view private. keep, or widen?

four independent decisions:

**(e1) top level.** `intervals/__init__.py` exports `Allen` and `TruthSet` (the result types) and
no relation function at all: `before`, `allen`, `overlaps`, ... all live on the class and in
`relations.py` (verified, `__all__` lines 59-112). adding `allen_matrix` alone to the top level
would be the one relation there. **keep nothing at the top level, confidence ~90%.** additive
later; no cost either way.

**(e2) `DecoratedInterval`.** every relation is in `tests/test_propagation.py::NOT_ON_THE_WRAPPER`
(line 640): relations go through `.interval`, by design (the decoration says nothing about how two
sets relate). `allen_matrix` and `allen_relations` on the wrapper alone would break that line.
**keep off the wrapper, confidence ~90%.** additive later. (if the owner ever puts the relations
on the wrapper, do them all at once.)

**(e3) functions in `relations.py` beside the methods.** consistent with every relation; the
functions over cut tuples are the kernel-level API the tests and the ieee1788 layer use. keep.

**(e4) the sparse view: make `_allen_pairs` public?**
* what it is today: `(i, j, allen(pa[i], pb[j]))` for the pairs of cells whose common refinement
  intersects, at most `n + m - 1`. that is a SUPERSET of the non-BEFORE/AFTER pairs, not the set
  itself: probe `shape.py` 2026-10-03 on `[0, 1] | [3, 5]` vs `[1, 4] | [6, 7] | [8, 9]` swept
  `(0,0,OVERLAPS), (1,0,OVERLAPPED_BY), (1,1,BEFORE)`, while the non-BEFORE/AFTER pairs are only
  the first two. its cost is pinned down to this exact visiting rule
  (`::test_allen_relations_is_a_linear_sweep` asserts `visited == cells_meet` pairs).
* **E4a. keep private (built).** pros: no contract to publish for a visiting rule that is an
  implementation detail (the test pins it as such); adding a public name later is free; no user
  today. cons: the one `O(n + m)` form that keeps indices is unreachable, so the segmentation /
  agreement use (§1 item 3) must build the matrix or re-implement the sweep.
* **E4b. public `A.allen_pairs(B)` with a clean contract: the pairs `(i, j, relation)` of pieces
  that are not BEFORE or AFTER each other (the pieces that intersect or meet), in order, `O(n +
  m)`.** that is `_allen_pairs` filtered of BEFORE/AFTER, so the sweep's visiting rule stays
  private and the public contract is a property of the operands, not of the algorithm. the set
  view becomes `{r for _, _, r in pairs} | corners`. pros: the natural answer for large `n`
  (9 ms at 1000, probe) with the indices kept; what a composition table and the M8 time layer's
  "which busy slots touch which" would call; small (a filter and a method). cons: a third public
  name for one idea at 2.0, with no user; the indices index `A.pieces`, which materialises `n`
  `MultiInterval`s if the caller then uses them (fine, `O(n)`).
* **E4c. public as a dict `{(i, j): relation}`** of the same pairs. hashable lookups; loses order;
  no real gain over tuples.
* **recommendation: E4a, keep private for 2.0, confidence ~60%.** adding `allen_pairs` later is
  purely additive, so the asymmetry says wait for a user. if the owner has the segmentation /
  agreement use or the composition table in sight before 2.0, build E4b (not E4a's raw sweep), and
  then `allen_relations` should be documented as derived from it.
* what would change my mind: an actual caller with thousands of pieces who needs the indices; the
  composition table being scheduled.
* cost of changing later: private -> public is free at any time; so is renaming a private name.
  the only cost of deferring is that a user re-implements the sweep and gets the tie rule wrong.

## summary table

| sub-item | built | recommendation | confidence | cost to change after 2.0 |
|---|---|---|---|---|
| (a) matrix shape | nested tuples | keep; no class, no rename; note `zip(*M)` / `np.array(M)` in the docstring | ~75% | low if a future class subclasses `tuple`; otherwise breaking |
| (b) set view | `allen_relations`, frozenset | keep, keep the name; add one docstring sentence (extensional, not a disjunction); settle the assert consistency (HANDOFF line 307), preferably assert in every cut-tuple relation | ~65% | dropping = deprecation cycle; adding back = free |
| (c) empty operand | `()` / `((),)*n` / `frozenset()` | keep | ~85% | value -> raise is breaking; do not revisit |
| (d) plain loop vs fill + sweep | plain loop | keep; the 1.8x (2026-10-03) is sub-millisecond at realistic sizes and the set view is 100x faster where it would matter | ~80% | zero (internal; same result on normalized operands) |
| (e1) top level | nothing | keep | ~90% | additive |
| (e2) `DecoratedInterval` | not on it | keep (all relations or none) | ~90% | additive |
| (e4) sparse view | private `_allen_pairs` | keep private; if a user or the composition table appears, publish it FILTERED (non-BEFORE/AFTER pairs), not the raw sweep | ~60% | additive |

the one thing I would do before 2.0 regardless: fix the assert inconsistency named in HANDOFF
"still owed" (line 307), since whichever way it is settled is a behaviour under `__debug__` that
is awkward to change after a release only if someone depends on the assert firing.

probes: `.scratch/owner-questions/probes-allen/trade.py` (+ `trade.out.txt`), `shape.py` (+
`shape.out.txt`); both run 2026-10-03 with the env's interpreter, `PYTHONPATH=.` from the repo root.
no tracked file was edited; no git command was run.
