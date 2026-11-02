# Decision id map: docs/evidence/decisions-normalised-2026-11-02.md into docs/decisions

FROZEN. Printed by `context-engine decision import --dry-run` on 2026-11-02, session 2026-11-02a; the import gives each record below its id, and the old id, where there was one, stays in its `aliases` (FRAMEWORK §11.4). Line numbers are the log's.

| id in the log | record | status | title |
|---|---|---|---|
| D-1 | D-1 | BUILT | closure at infinity |
| D-2 | D-2 | BUILT | indeterminate corners |
| D-3 | D-3 | BUILT | `int / int` that is not integral |
| D-4 | D-4 | BUILT | infinities for the time layer |
| D-5 | D-5 | BUILT | modulo scope for v2.0 |
| D-6 | D-6 | BUILT | constructor default for an infinite bound |
| D-7 | D-7 | BUILT | a box that *is* an indeterminate point |
| D-8 | D-8 | BUILT | modulo with infinite operands |
| D-9 | D-9 | BUILT | 1788's numeric ops on a multi-interval |
| D-10 | D-10 | BUILT | names for 1788's interval orders, since `<` and `<=` are pointwise and return a `TruthSet` |
| D-11 | D-11 | BUILT | power |
| D-12 | D-12 | BUILT | a reverse op whose answer has infinitely many or very many pieces |
| D-13 | D-13 | BUILT | `cancelMinus(A, B)` is the Minkowski difference, the largest `X` with `B + X ⊆ A` |
| D-14 | D-14 | BUILT | the independent oracle for the elementary functions is `python-flint` |
| D-15 | D-15 | BUILT | licence |
| D-16 | D-16 | BUILT | decorations (com/dac/def/trv/ill) |
| D-17 | D-17 | BUILT | M13 does not block the 2.0.0 release, and there is no hurry to release either |
| D-18 | D-18 | BUILT | M13's proposed categories and choices |
| D-19 | D-19 | BUILT | the solver stack's first part (M15) |
| D-20 | D-20 | BUILT | the solver stack's second part (M16a) |
| D-21 | D-21 | BUILT | the 1788 layer (M16b) |
| D-22 | D-22 | BUILT | the per-piece allen matrix (M16c) |
| D-23 | D-23 | BUILT | numpy interop (M16d) |
| D-24 | D-24 | BUILT | the gmpy2/mpfr backend (M16e) |
| D-25 | D-25 | BUILT | CPython 3.11's `Fraction.__pow__` rounds a Fraction base to a float before... |
| D-26 | D-26 | BUILT | to nearest, a reverse op's exact preimage wholly past MAX squeezes to the point `[±inf]` |
| D-27 | D-27 | BUILT | four departures from 1788 that were build choices are deliberate |
| D-28 | D-28 | BUILT | pown of exact operands past one exact-result limit of about 2 ** 22 bits, shared by pown |
| D-29 | D-29 | BUILT | each end keeps its own number type |
| D-30 | D-30 | BUILT | M8's three choices and the smaller ones with them |
| D-46 | D-46 | BUILT | implementability review |
| D-45 | D-45 | BUILT | smaller gaps, accepted and written into "current design" 2026-09-23 |
| D-36 | D-36 | BUILT | MIT licence, a publish workflow and a changelog for 2.0.0; Q25 answered |
| D-35 | D-35 | BUILT | an extremum known exactly is an exact end; Q27 answered |
| D-34 | D-34 | BUILT | the fuzz and gate:gmpy2 phases run on pytest-xdist (fuzz-xdist; a build's defaults, awaiting the owner) |
| D-33 | D-33 | BUILT | m14b-open's build defaults beside D32 (awaiting the owner) |
| D-32 | D-32 | BUILT | precision first: what is known exactly is exact (revises D31) |
| D-31 | D-31 | BUILT | a result's number type says where its value came from (Q26) |
| D-87 | D-87 | BUILT | separators exactly between items (Q24) |
| D-86 | D-86 | BUILT | a number is what python reads (Q23) |
| D-85 | D-85 | BUILT | the package is `multiinterval` |
| D-84 | D-84 | BUILT | M8's choices confirmed (Q21) |
| D-83 | D-83 | BUILT | strict flags; v1 deleted (Q22) |
| D-82 | D-82 | BUILT | M8's review round (three reviews, one fixer) |
| D-81 | D-81 | BUILT | M8, the time layer, built |
| D-80 | D-80 | BUILT | the time layer's choices (M8, D30) |
| D-79 | D-79 | BUILT | v1's leftovers settled; shifts dropped |
| D-78 | D-78 | BUILT | owner answers to Q9-Q20 and the owner's-call items |
| D-77 | D-77 | BUILT | to nearest, a reverse op meets `x` before rounding (D26) |
| D-76 | D-76 | BUILT | fuzz on push, not on a schedule |
| D-75 | D-75 | BUILT | `-` and `+` keep each cut's type (fuzz-symmetry) |
| D-74 | D-74 | BUILT | pown-huge, built |
| D-73 | D-73 | BUILT | python 3.12 minimum (owner, D25) |
| D-72 | D-72 | BUILT | M16e, the gmpy2/mpfr backend (H3's second part), built |
| D-71 | D-71 | BUILT | M16d, numpy interop (H3's second part), built |
| D-70 | D-70 | BUILT | M16c, the per-piece allen matrix (H3's second part), built |
| D-69 | D-69 | BUILT | M16b, the 1788 layer (H3's second part), built |
| D-68 | D-68 | BUILT | M16a, the solver in several variables (H3's second part), built |
| D-67 | D-67 | BUILT | M15, autodiff and interval newton (H3's first part), built |
| D-66 | D-66 | BUILT | owner answers on M13's proposed categories (D18) |
| D-65 | D-65 | BUILT | M13 merged, M13's exit met |
| D-64 | D-64 | BUILT | owner answers to the open questions |
| D-63 | D-63 | BUILT | M13e, the reverse ops, built |
| D-62 | D-62 | BUILT | M13e, fourth part (pow_rev1, pow_rev2), built |
| D-61 | D-61 | BUILT | M13e, third part (sin_rev, cos_rev, tan_rev), built |
| D-60 | D-60 | BUILT | M13e, second part (mul_rev), built |
| D-59 | D-59 | BUILT | M13e, first part (sqr_rev, abs_rev, pown_rev, cosh_rev), built |
| D-58 | D-58 | BUILT | M13g, decorations, constructors and signals, done |
| D-57 | D-57 | BUILT | M13g part 3, decoration propagation, built |
| D-56 | D-56 | BUILT | M13g part 2, the decorated type, built |
| D-55 | D-55 | BUILT | M13g part 1, signals and bare constructors, built |
| D-54 | D-54 | BUILT | M13d, power and the rest of the elementary functions, built |
| D-53 | D-53 | BUILT | M13f, cancellation, built |
| D-52 | D-52 | BUILT | M13c, the interval orders and the interior, built |
| D-51 | D-51 | BUILT | M13b, the numeric functions, built |
| D-50 | D-50 | BUILT | M13h, the reductions, built |
| D-49 | D-49 | BUILT | M13a and M14's fuzz job and oracle built |
| D-48 | D-48 | BUILT | M13 and M14 planned (not built) |
| D-47 | D-47 | BUILT | M12, the unblocked backlog built |
| D-44 | D-44 | BUILT | implementability review |
| D-43 | D-43 | BUILT | D1–D6 settled |
| D-42 | D-42 | BUILT | degenerate indeterminate boxes return ∅ |
| D-41 | D-41 | BUILT | signed zero dropped |
| D-38 | D-38 | BUILT | 2025-12 wishlist and negative-zero enums |
| D-37 | D-37 | BUILT | negative zero |
| D-39 | D-39 | BUILT | update |
| D-40 | D-40 | BUILT | v2 consolidated decisions (2026-08-16) |

## Not imported

- lines 1-25: before the first entry: '# decisions'
- line 102: a heading with no text of its own, outside every entry: "## the log (newest first; D31 on written here, the rest from `v2-plan.md`'s deci"

## Headings kept inside entries

- none

