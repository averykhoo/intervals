# M16c, H3's second part: the per-piece allen matrix (stream allen-matrix)

the stream's record, for the orchestrator to merge into `v2-plan.md`, `v2-implementation-plan.md`,
`HANDOFF.md` and `README.md` (not edited on this branch). built 2026-09-28 on `h3-allen-matrix`,
from `v2` at `04946af`. design: `.scratch/h3b/design/allen-matrix.md` and its critique (gitignored;
what they decided is below).

## design

text for `v2-plan.md` "current design" / "comparisons". after the bullet "`allen(a, b)` on
contiguous pieces only (raise otherwise) ...":

* **allen of any operands, per piece** (M16c, H3's second part, built 2026-09-28;
  `intervals/relations.py::allen_matrix`, `::allen_relations`, `::_allen_pairs`):
  `A.allen_matrix(B)` is `allen()` of every pair of pieces, a tuple of tuples of `Allen` with a row
  per piece of `A` and a column per piece of `B`, both in order (`A.allen_matrix(B)[i][j] is
  A.pieces[i].allen(B.pieces[j])`). `A.allen_relations(B)` is the `frozenset` of the relations
  holding between some piece of `A` and some piece of `B`: allen's algebra reasons over relation
  sets (the 2026-08-16 note). every entry is `allen()` of a cut pair, so exactly one of the 13 per
  pair (JEPD per entry, which is why it goes per piece) and no new divergence against 1788: the
  cut-based relations' 5 keys stand, a point never OVERLAPS, `[1, inf)` MEETS `[inf]`
* **an empty operand has no pairs**: `EMPTY.allen_matrix(B)` is `()`, `A.allen_matrix(EMPTY)` is
  one empty row per piece of `A` (the shape is kept: `len(M) == len(A)` always), and
  `allen_relations` is `frozenset()`. not a raise (`allen()` still raises: it owes one relation)
  and not a warning (not an op on a set)
* **cost**: the matrix is the plain `n x m` loop over `allen()`, `Θ(nm)` like its size, and it does
  not use the order of the pieces (any cut pairs do). the set view is an `O(n + m)` merge sweep
  over the pieces (`_allen_pairs`: at most `n + m - 1` calls to `allen()`, every pair that is not
  BEFORE or AFTER among them) plus two corner comparisons (some piece of `A` is BEFORE some piece
  of `B` iff the first of `A` is BEFORE the last of `B`), and never builds the matrix. for many
  pieces the set view is the answer; a 1000 x 1000 matrix is a million references
* within one set the pieces never meet (`kernel.normalize` merges pieces that touch), so
  `A.allen_matrix(A)` is EQUALS on the diagonal, BEFORE above it and AFTER below it. `adjoins`
  stays a fact about the sets' ends: `[0, 1) | [3, 5]` has a piece that MEETS `[1, 2]`, and does
  not adjoin it
* numbers coerce to points and anything else is a `TypeError`, as every relation; the two classes
  mix. not on `DecoratedInterval`: relations go through `.interval`, as `allen` does
  (`tests/test_propagation.py::NOT_ON_THE_WRAPPER` lists both names). nothing new at the top level
  (`Allen` already is); the sparse view `(i, j, relation)` stays private (`_allen_pairs`)

in the same section, "the class exposes before, after, adjoins, overlaps, contains, within and
allen" becomes "... within, allen, allen_matrix and allen_relations".

in "later (not in v2.0)", the list from `HANDOFF.md` H3 drops "a per-piece Allen matrix" and
gains: "allen's composition table (from the relations of `(a, b)` and `(b, c)`, those possible for
`(a, c)`), re-derived for the cut reading with points, where some classical compositions shrink
(a point cannot OVERLAP); `allen_relations` is its input. a design of its own".

## decision-log revision

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
  builds the matrix; pinned by counts, not time
* **not on `DecoratedInterval`** (through `.interval`, as `allen`); the sparse `(i, j, relation)`
  view private; nothing at the top level
* not built: allen's composition table over the cut reading ("later")

## D22

the §0 table row for `v2-implementation-plan.md`:

| D22 | **decided in the build 2026-09-28 (the session's defaults), open for the owner: `HANDOFF.md` Q14.** the per-piece allen matrix (M16c): (a) `A.allen_matrix(B)`, a tuple of tuples of `Allen` (rows the pieces of `A`, columns those of `B`), and `relations.allen_matrix` over cut tuples; (b) `A.allen_relations(B)`, the `frozenset` of the relations holding between some pair of pieces, a second public name the H3 row did not list; (c) an empty operand gives `()` / one empty row per piece / `frozenset()`, no raise and no warning; (d) the matrix is the plain `n x m` loop over `allen()` (no dependence on normalized input; the design's ~2-3x faster fill + sweep not taken), the set view an `O(n + m)` sweep that never builds the matrix; (e) methods on `MultiInterval`, functions in `relations.py`, nothing at the top level, not on `DecoratedInterval`, the sparse `(i, j, relation)` view private | as built | nothing (additive: `allen()` and every existing name unchanged); M16c's record |

## M16c

### M16c the per-piece allen matrix (H3's second part; done 2026-09-28)

the owner, 2026-09-27: "get the rest of h3 done". H3's row lists "per-piece Allen matrix"; its
spec pointer is `v2-plan.md` "v2 consolidated decisions (2026-08-16)" / "comparisons". the choices
the build made are D22, open for the owner as `HANDOFF.md` Q14; the design is `v2-plan.md`
"comparisons" (the text of this file's "design").

* **`intervals/relations.py`**: `allen_matrix(a, b)` (the plain loop), `allen_relations(a, b)`
  (the sweep and two corners), `_allen_pairs(pa, pb)` (the sweep, private; it calls `allen` as the
  module global, which the cost test counts); one sentence in the module docstring
* **`intervals/multi_interval.py`**: `MultiInterval.allen_matrix`, `MultiInterval.allen_relations`,
  each coercing through `_coerce_or_raise`, each with doctests (the worked example and an empty
  operand)
* exit: every entry is `allen()` of its pair of pieces and the set view is the matrix's entries,
  both against the `n x m` loop over the pinned `allen()`; the converse, a set against itself and
  the set relations (overlaps, disjoint, before, after, within, contains, equals, adjoins) as
  identities over the matrix; the empty shapes; the set view `O(n + m)` in calls and never the
  matrix; the gate green; every new property sabotaged once and seen red

record (2026-09-28):
* **what the build found on its way**:
  * **the gate found a name the design missed.** `tests/test_propagation.py::test_every_public_name_of_the_core_is_on_the_wrapper_or_asked_of_the_interval`
    went red on `{'allen_matrix', 'allen_relations'}`: a name the core gains must be on
    `DecoratedInterval` or in `::NOT_ON_THE_WRAPPER`, on purpose. the design's "not on
    `DecoratedInterval`" is now written there, beside `allen` (the design listed the file as
    untouched). the guard worked as meant
  * **the sweep's tie rule was held by chance.** "a tie advances `i` only" is output-equivalent (one
    extra `allen()` per tie, an AFTER pair; still within `n + m - 1`), and the design accepted it
    unpinned. the repo's rule makes a green break a gap: the cost test now asserts the sweep visits
    exactly the pairs of cells that intersect (piece `i` owns the cuts after the end of piece
    `i - 1`, up to its own end; the sweep walks the two partitions' common refinement). that went
    red on a re-run, but green on the first final run: measured 2026-09-28, 19 of 20 seeded
    100-example runs of `operand_pairs` hit a tie followed by more pieces (0 to 15 examples a
    run). so the test carries an `@example` of a two-piece set against itself, where every end
    ties; red since then
  * **the sabotage harness ran stale bytecode.** the template harness (copy the file to `.orig`,
    write the break, run, move `.orig` back) restores a file whose mtime lies in the same second
    as the broken write; for a break of the same byte length ("allen_relations operands swapped")
    the restored source matched the broken `.pyc` (mtime in seconds, size), and python kept
    running the broken bytecode. the next two runs were red for that reason, not their own; both
    were re-run. the harness now clears `__pycache__` before and after each break, runs with
    `PYTHONDONTWRITEBYTECODE=1`, restores with `copy2`, and starts with a control row on the
    intact code. H3's template harness has the same hazard
  * **the conservative matrix is pinned**: the plain loop is a choice, so
    `::test_allen_matrix_does_not_need_normalized_operands` (two cut tuples laid end to end give
    the two matrices stacked) holds it; the design's fill + sweep goes red there
* **tests** (`tests/test_relations.py`, and the two methods' doctests in
  `intervals/multi_interval.py`), all at hypothesis's default settings (no `@settings`, so the
  fuzz profile multiplies them like the rest):
  * operands: `::operand_pairs`, a mixture of two independent `::operands`
    (`exact_cut_tuples` or `::dense`, a grid with ±inf) and pairs where one is derived from the
    other (`::_derived`: itself, its complement, hull, gaps, interior, and itself with the
    complement of its hull), so shared cuts, MEETS and MET_BY are common; oracle `::allen_loop`,
    the `n x m` loop over `relations.allen`; `::converse` takes the column count explicitly
  * `::test_allen_matrix_is_allen_of_each_pair_of_pieces` (and the method equals the function),
    `::test_allen_relations_are_the_matrix_entries`, `::test_allen_matrix_converse`,
    `::test_allen_matrix_of_a_set_with_itself`, `::test_allen_matrix_and_the_set_relations`
    (columns counted with `range(m)`), `::test_allen_matrix_on_single_pieces`, and two lines in
    `::test_allen_table` (on each of its 17 rows the matrix is `((relation,),)` and the set view
    `{relation}`)
  * `::test_allen_matrix_does_not_need_normalized_operands`,
    `::test_allen_matrix_of_unordered_overlapping_pieces` (the plain loop's contract)
  * `::test_allen_matrix_of_an_empty_operand` (no raise, no warning of any kind; `allen()` still
    raises), `::test_allen_matrix_table` (13 worked rows, both directions: a shared closed end,
    a piece that meets where the sets do not adjoin, a point filling a one-point gap, points
    against pieces, `[1, inf)` and `[1, inf]` against `[inf]`, `[-inf]`, both corners at once),
    `::test_a_piece_meets_where_the_sets_do_not_adjoin`,
    `::test_allen_matrix_coerces_as_every_relation` (a number, a string, nan, the two classes in
    both orders), `::test_allen_matrix_mixes_numeric_types_at_a_shared_cut` (`[0, 0.5)` MEETS
    `[1/2, 3]`, `[0, 1.0)` MEETS `[1, 2]`)
  * `::test_allen_relations_is_a_linear_sweep`, the cost pin: `_allen_pairs` directly (no pair
    twice, at most `n + m - 1`, every pair that is not BEFORE or AFTER, exactly the intersecting
    cells, each relation `allen()`'s), then `allen_relations` with `relations.allen_matrix` patched
    to raise (critique B1: flattening the matrix would keep the call count) and `relations.allen`
    spied: at most `n + m - 1` calls and at least one per pair that is neither BEFORE nor AFTER.
    its docstring says the lower bound pins an implementation detail on purpose (a spy that
    cannot pass at 0 calls): relax it knowingly, do not delete it
* **measured 2026-09-28** (shared laptop, four other streams running):
  * the stream: `python -m pytest -q -p no:cacheprovider tests/test_relations.py
    intervals/relations.py intervals/multi_interval.py`: 103 passed in 36.4 s
  * the module three times each, `.hypothesis` cleared before every run, `python -m pytest -q -p
    no:cacheprovider <module>`: at `04946af` 47 tests in 21.1, 25.2, 26.6 s; with M16c 73 tests in
    34.0, 38.8, 38.1 s: about 13 s added to the gate. `too_slow` never fired (these six runs and
    every sabotage run), so no health check is suppressed
  * the trade (a throwaway probe; `n = m` pieces `[2k, 2k+1]` against `[2k+1/2, 2k+3/2]`, best of 3,
    `time.perf_counter`): laptop numbers, the tests pin counts, not times

    | n = m | matrix, plain loop (built) | matrix, fill + sweep (not taken) | relations, sweep (built) | relations, matrix flattened |
    |---|---|---|---|---|
    | 10 | 0.18 ms | 0.18 ms | 0.09 ms | 0.20 ms |
    | 100 | 16.0 ms | 9.9 ms | 0.77 ms | 18.5 ms |
    | 300 | 194 ms | 102 ms | 4.7 ms | 168 ms |
    | 1000 | 4040 ms | 2060 ms | 27.5 ms | 4640 ms |

  * the gate, from the worktree root: `python -m pytest -q tests/itf1788`: 18246 passed in 81.3 s.
    `python -m pytest -q --ignore=tests/itf1788` in three calls (five streams at once would
    overrun one call): `tests/test_reverse.py tests/test_functions.py tests/test_propagation.py
    tests/test_oracle_flint.py tests/test_pow_rev.py` 1011 passed and the one red above in
    398.8 s, `tests/test_propagation.py` re-run after the fix 193 passed in 51.0 s; nine files
    (`test_elementary`, `test_modulo`, `test_ops_examples`, `test_ops_properties`,
    `test_relations`, `test_applicator`, `test_literals`, `test_oracles`, `test_autodiff`) 2467
    passed in 277.5 s; the rest (`--ignore` of those 14 files, with the doctests and `README.md`)
    637 passed in 296.8 s. 4116 items outside itf1788 (22362 collected in all, one process),
    973 s summed (1024 s with the re-run)
* **sabotage** (a throwaway harness: each break alone, `.hypothesis` and `__pycache__` cleared, the
  stream's three files with `-x` and a 600 s timeout, the file restored and compared; 2026-09-28).
  the last column is the first test to fail under `-x`:

| break | first run | final run: red by |
|---|---|---|
| none (control: the intact code) | green | green: 103 passed |
| sweep: advance the other pointer | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| sweep: compare starts, not ends | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| sweep: a tie advances `i` only | green | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| sweep: stops after two pairs | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| sweep: `allen` bound locally (the spy reads 0) | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: BEFORE corner dropped | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: BEFORE corner `<=` (MEETS counted) | red | red: `tests/test_relations.py::test_allen_table[[1, 2)-[2, 3]-Allen.MEETS]` |
| relations: AFTER corner on the wrong pieces | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: flatten `allen_matrix` (critique B1) | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: `n m` calls to `allen()` | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: an empty operand raises | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: the empty guard removed | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| matrix: each entry inverted | red | red: `tests/test_relations.py::test_allen_table[[1, 2]-[3, 4]-Allen.BEFORE]` |
| matrix: transposed | red | red: `tests/test_relations.py::test_allen_matrix_is_allen_of_each_pair_of_pieces` |
| matrix: an empty other gives `()` (shape lost) | red | red: `tests/test_relations.py::test_allen_matrix_is_allen_of_each_pair_of_pieces` |
| matrix: the design's fill + sweep (the choice not taken) | red | red: `tests/test_relations.py::test_allen_matrix_does_not_need_normalized_operands` |
| method: `allen_matrix` without `_coerce_or_raise` | red | red: `tests/test_relations.py::test_allen_matrix_coerces_as_every_relation` |
| method: `allen_relations` operands swapped | red | red: `tests/test_relations.py::test_allen_table[[1, 2]-[3, 4]-Allen.BEFORE]` |

the one green in the first run was a gap, closed the same session and re-run red: the tie rule
(`::test_allen_relations_is_a_linear_sweep`, the intersecting cells and the `@example` above). the
control row was added with the harness fix, so its first run is the final one's. two runs between
the first and the final one were red for a stale `.pyc` (above) and are not in the table.

## readme

`README.md` line 83, the relations bullet: "`allen()` gives the Allen relation" becomes
"`allen()` gives the Allen relation of two contiguous sets, and `allen_matrix()` and
`allen_relations()` that of every pair of pieces, as a matrix or as the set of relations
holding". the example block, after the `strictly_less` line (checked as a doctest 2026-09-28, in a
scratch copy run with the repo's `pyproject.toml`; deterministic):

```python
>>> A = MI(0, 1) | MI(3, 5)
>>> [[r.name for r in row] for row in A.allen_matrix(MI(1, 4))]   # allen() of each pair of pieces
[['OVERLAPS'], ['OVERLAPPED_BY']]
>>> sorted(r.name for r in A.allen_relations(MI(2, 6) | MI(8, 9)))   # the relations holding
['BEFORE', 'DURING']
```

## Q14

for `HANDOFF.md` "questions for the owner":

* **Q14 M16c's choices (D22)**, built as the session's defaults when the owner said "get the rest
  of h3 done":
  (a) the matrix: `A.allen_matrix(B)`, nested tuples of `Allen` (rows the pieces of `A`), and
  `relations.allen_matrix` over cut tuples. keep, rename, or a small `AllenMatrix` class
  (`.transpose()`, `.converse()`)?
  (b) the set view: `A.allen_relations(B)`, a `frozenset` of `Allen`. the H3 row named only the
  matrix; the 2026-08-16 note says "matrix, or the set of relations". keep or drop; if kept, the
  name (`allen_set` was the alternative)?
  (c) an empty operand: no rows or empty rows and `frozenset()`, not `allen()`'s `ValueError`
  (d) the matrix as the plain `n x m` loop over `allen()`, not dependent on normalized input; the
  design's fill + sweep, ~2-3x faster, not taken
  (e) the surface: methods on `MultiInterval` and functions in `relations.py`, nothing at the top
  level, not on `DecoratedInterval` (`.interval` first, as `allen`), the sparse `(i, j, relation)`
  view private (`relations._allen_pairs`). keep, or widen?
  (plan §0 D22, §2 M16c; `v2-plan.md` "2026-09-28 revision: M16c")

## still owed

* the merge into `v2-plan.md` ("comparisons", "later", the decision log), `v2-implementation-plan.md`
  (§0 D22, §2 M16c), `HANDOFF.md` (H3's row loses "per-piece Allen matrix"; Q14; banner and session
  log with the gate numbers above) and `README.md` (the text under "readme", doctested there), from
  this file: the orchestrator's
* `tests/test_propagation.py::NOT_ON_THE_WRAPPER` gained two names on this branch; another stream
  that adds a core name (or wraps one) edits the same set, so the merge may conflict there
* H3's sabotage template (`.scratch/h3/sabotage.py`) has the stale-bytecode hazard found here; a
  same-size break followed by a run that does not rewrite that file executes the broken code.
  M15's table was not re-checked for it
* not built, for "later": allen's composition table over the cut reading
