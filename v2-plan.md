# `MultiInterval` v2 plan

two parts. **current design** is normative: if the code and that section disagree, one of them is a
bug. the **decision log** below it is history, kept verbatim, with a marker wherever a later decision
superseded it.
open work and open questions for the owner (including the ones raised in the decision log's
2026-09-25/26 entries) live in `HANDOFF.md`; the milestones are in `v2-implementation-plan.md`.

## current design (2026-09-23; brought up to date with the build at M12, 2026-09-25, and M13a,
M13b, M13c, M13d, M13e, M13f, M13g, M13h and M14's fuzz job and oracle, 2026-09-26; M13's merge, 2026-09-27;
M15, H3's first part: autodiff and interval newton, 2026-09-27; M16, H3's second part: several
variables, the 1788 layer, the allen matrix, numpy, the gmpy2 backend, 2026-09-28)

### domain and semantics

* values: the affine extended reals `[-inf, inf]`, both infinities as points. **one zero.** `-0.0` is
  normalized to `0.0` at construction and the sign bit is never consulted (as v1 already does)
* `[inf]` and `[-inf]` are legal degenerate intervals; `[a, inf]` and `[a, inf)` are different sets
* number types: int and Fraction (exact, never rounded), float (endpoint arithmetic goes through the
  rounding hook: to nearest in `MultiInterval`, outward in `OutwardMultiInterval` — see
  arithmetic). datetime/timedelta: the time layer is deferred; v1's `time_interval.py` is archived
  with the rest of v1 and returns once it runs on the v2 class (D4)
    * `int / int` that is not integral gives a Fraction; float only if an operand is float. a
      Fraction with denominator 1 is normalized to int in the Cut constructor, next to the `-0.0`
      normalization (they already compare and hash equal; this keeps types and printing clean)
    * python leaks floats at infinity (`Fraction(1) / inf` is `0.0`), so the applicator evaluates
      infinite corners itself and returns exact values (`1/[inf]` = `[0]`, not `[0.0]`)
    * **±inf is exact whatever its python type.** `math.inf` is a float, but it is never rounded,
      it does not make an interval "float" (an interval is float iff a *finite* endpoint is), and
      attainment at ±inf is decided symbolically, never by float arithmetic
    * `nan` in a constructor is a `ValueError`
* meaning of an arithmetic result: the set of values attained, with ±inf as ordinary points. an
  infinite endpoint is closed iff it is attained; open/closed flags propagate through infinity like
  anywhere else. `1/(-1, 0)` = `(-inf, -1)`: -inf is approached, not attained. a pole at a *closed*
  zero endpoint attains the infinity of its piece's sign (next bullet)
* **direction comes from the set, never from a sign bit.** reciprocal splits at zero into sign-pure
  pieces (needed for monotonicity anyway); a zero endpoint of a negative piece maps to `-inf`, of a
  positive piece to `+inf`. a degenerate `[0]` has no direction: `1/[0]` = `∅` and emits
  `IndeterminateResultWarning` (isotonicity forces it, see "indeterminate forms")
    * `1/[-1, 0]` = `[-inf, -1]` — same as Hickey and as ieee 1788, no signed zero needed
    * `1/[-1, 1]` = `[-inf, -1] ∪ [1, inf]`
    * `1/[1, inf]` = `[0, 1]`, `1/[1, inf)` = `(0, 1]`: closedness at infinity and at zero correspond
* consequence: `1/x` is an involution only on sets with no degenerate piece at `0`, `inf` or `-inf`,
  and not unbounded at both ends while holding exactly one of ±inf (the second condition added
  2026-09-24, found by the M6 tests: `1/(-inf, inf]` = `[-inf, inf]`, which maps to itself).
  such a piece is lost: `1/[0]` = `∅`, so `1/(1/[inf])` = `∅`, and `[0] ∪ [1, 2]` → `[1/2, 1]` →
  `[1, 2]`. the sharp answer needs one bit of memory on a degenerate zero (Kahan's argument for the
  sign bit) and is deferred — see "later"
* indeterminate corners (`±inf · 0`, `inf - inf`, `±inf / ±inf`) of a box that is not itself the
  indeterminate point take the limit along the box — the rule reciprocal already uses at zero. the
  corner contributes the limit along each non-degenerate edge that meets it:
    * mul at `(±inf, 0)`: `0` if the infinite factor's interval is non-degenerate, the signed
      infinity if the zero factor's interval is
    * sub at `(inf, inf)`: `-inf` if the minuend is non-degenerate, `+inf` if the subtrahend is; add
      at `(inf, -inf)` likewise; division is multiplication by the reciprocal
    * `[-inf, -1] * [0]` = `[0]`, `[-inf] * [0, 1]` = `[-inf]`, `[-inf, -1] * [0, 1]` = `[-inf, 0]`,
      `[1, inf] / [1, inf]` = `[0, inf]`, `[1, inf] - [1, inf]` = entire: 1788's answers up to
      closure at infinity
    * a user-typed `[1, inf]` is taken literally
* indeterminate forms: a box that *is* the indeterminate point (`1/[0]`, `[0] * [inf]`,
  `[inf] - [inf]`, `[0] / [0]`) returns `∅` and emits `IndeterminateResultWarning`. inclusion
  isotonicity (`A ⊆ B ⇒ f(A) ⊆ f(B)`) forces it: `[0]` is inside both `[-1, 0]` and `[0, 1]`, so
  `1/[0] ⊆ [-inf, -1] ∩ [1, inf] = ∅`; likewise `[0] * [inf] ⊆ [0] * [5, inf] ∩ [0, 1] * [inf]` and
  `[inf] - [inf] ⊆ [inf] - [1, inf] ∩ [1, inf] - [inf]`, both empty under the corner rule. solvers
  (bisection, forward-backward contraction) need isotone ops, and `1/[0]` = `∅` is also 1788's
  answer

### representation: cuts

* a bound is a **cut**: a boundary *between* points, `Cut(value, side)` with
  `side ∈ Side(IntEnum): BELOW = -1, ABOVE = +1`. never call it epsilon — that word names the v1
  point-plus-offset model this replaces
* the same token reads differently by position:

  | cut           | as start | as end |
  |---------------|----------|--------|
  | `(v, BELOW)`  | `[v`     | `v)`   |
  | `(v, ABOVE)`  | `(v`     | `v]`   |

  `[a,b]` = `(a,BELOW),(b,ABOVE)` · `(a,b)` = `(a,ABOVE),(b,BELOW)` · `[a,b)` = `(a,BELOW),(b,BELOW)`
  · `[x]` = `(x,BELOW),(x,ABOVE)`
* a MultiInterval is an immutable, even-length, **strictly increasing** flat tuple of cuts. ordering is
  plain tuple comparison (IntEnum members are ints), so hash and sort come for free
* empty iff `start >= end`: `[1,1)` and `(1,1)` normalize to empty. reversed *values* (`[2,1]`) are a
  ValueError
* merge: pieces merge iff `next.start <= current.end`. `[1,2) | [2,3]` → equal cuts → `[1,3]`;
  `[1,2) | (2,3]` → `(2,BELOW) < (2,ABOVE)` → the point 2 is missing → gap. no distance rule, no table
* complement: prepend `(-inf, BELOW)`, append `(inf, ABOVE)`, re-pair with roles shifted; empty pairs
  drop out by the rule above (the ends of `[-inf, inf]` become `((-inf,BELOW), (-inf,BELOW))` → empty)
* mirror: `(-v, other side)` via a two-entry lookup — negating an IntEnum yields a plain int
* **why no third side `EXACT = 0`**: a closed start at x and a closed end at x are *different*
  boundaries (just below x vs just above x). one shared "exact" token for both is v1's redundancy:
  `[1,2) | [2,3]` would compare end `(2,BELOW)` against start `(2,EXACT)`, unequal, and need a distance
  rule again; complement would need a translation table instead of a role swap. it feels natural
  because it names the point rather than the boundary; two sides is the whole trick
* `-0.0` normalizes to `0.0` inside the Cut constructor (they compare equal but print differently)
* arithmetic never operates on cuts directly. the applicator converts each piece to
  `(lo, lo_closed, hi, hi_closed)`, computes, and converts back. cuts are the storage and set-algebra
  form; nobody adds two cuts

### set operations and size

* union / intersection / difference / complement / membership / subset are kernel functions over
  cut tuples. `in` = scalar membership, with a documented subset alias for interval arguments;
  `__getitem__` slicing = restriction to the closed `[a, b]` (v1 behaviour); `__bool__` = non-empty
  (set precedent)
* `size` — was `cardinality` in v1 and `measure` in the 2026-08 plan; neither fits (Lebesgue measure
  ignores endpoints and rays). a lex-ordered named tuple `Size(rays, length, points)`, compared and
  added componentwise (the tiling invariants need `+`), nothing else. the ω·rays + length +
  ε·points gloss is the right intuition
    * **half-open baseline**: a half-open piece has exactly its length; a closed endpoint adds half a
      point, an open one removes half. so `[1]` = 1 point, `[1,2)` = `1`, `[1,2]` = `1 + 1pt`,
      `(1,2)` = `1 - 1pt`, and `[1] + (1,2) + [2] == [1,2]`. tiling forces this convention:
      `[0,1) + [1,2)` must equal `[0,2)`
    * `points = (closed_endpoints - open_endpoints) // 2`, always an integer (the sum is even). v1's
      integer was this in half-points, which is why an isolated point read as 2 there
    * `rays ∈ {0, 1, 2}`; `length` is the finite remainder after removing the rays from the origin and
      may be negative: `(1, inf]` = `Size(1, -1, 0)`
    * v1 bug, do not port: `multi_interval.py::cardinality` adds `inf - start` for a ray piece and
      then negates it, so `(1, inf]` comes out `(1, -inf, 0)` against its own docstring. the ray branch
      must contribute the finite endpoint only. (executed 2026-09-23 in the `intervals` conda env:
      with the default `INFINITY_IS_NOT_FINITE = True`, constructing `(1, inf]` raises `ValueError`;
      with the flag off `(1, inf]` and `[1, inf)` give `(1, -inf, 0)` and `[-inf, 1]` gives
      `(1, -inf, 2)`)
* **numeric functions** (D9, built at M13b 2026-09-26; `intervals/numeric.py`): 1788's `mid`,
  `rad`, `wid`, `mag`, `mig`, `midRad` as the methods `mid()`, `rad()`, `wid()`, `mag()`, `mig()`,
  `mid_rad()` (the pair `(mid(), rad())`). `mid`, `rad`, `wid` are **of the hull**: a midpoint
  outside the set (`mid([0,1] ∪ [9,10])` = 5) is still a valid bisection point, a per-piece form
  would return a tuple, and `size.length` already gives the width without the gaps. `mag` and
  `mig` are **of the set**, the `sup` and `inf` of `{abs(x) : x ∈ A}`: `mig([-3,-2] ∪ [2,3])` = 2,
  where the hull would give 0; on a connected set the two agree. open and closed ends do not
  matter (these are infima and suprema)
    * an operand with no finite float end gives exact values (int or Fraction); one with a finite
      float end anywhere (`rounding.has_finite_float`) gives floats, the exact value rounded once
      as 1788 specifies: `mid` to nearest, ties to even; `rad` the smallest double `r` with
      `[mid - r, mid + r]` holding the hull, measured from the rounded midpoint (so
      `rad([1, 1 + 3ulp])` is `2ulp`, not the double `1.5ulp`); `wid` and `mag` up; `mig` down.
      the direction is the function's, so `MultiInterval` and `OutwardMultiInterval` give the
      same numbers
    * unbounded operands follow 1788: `mid` of `(-inf, inf)` is 0, of a half-bounded hull ±max float
      (a float even for an exact operand); `rad` and `wid` are inf. a single point, `[inf]` too, is
      its own midpoint with radius and width 0. an exact end past max float keeps the midpoint in
      the hull (a float operand's rounds to ±max float rather than ±inf; an exact half-bounded hull
      starting past max float has its start as midpoint)
    * the empty set raises `ValueError` (`the empty set has no midpoint`, and so on), as `inf` and
      `sup` do; 1788 answers `NaN`, and the itf1788 adapter reads the error as that `NaN`
* **interior** (D10, built at M13c 2026-09-26; `intervals/kernel.py::interior`): the property
  `A.interior`, a set operation in its own right, is every end opened, the infinite ones too: the
  interior in the topology of the reals. a degenerate piece drops out (`[2]` → `∅`), and so does a
  closed end at ±inf, a point with no neighbourhood of reals (`[5, inf]` → `(5, inf)`, `[inf]` →
  `∅`, `[-inf, inf]` → `(-inf, inf)`); pieces stay apart (`[0, 1) | (1, 2]` → `(0, 1) | (1, 2)`).
  1788's `interior(A, B)` is `A.within(B.interior)`, so `∅` is inside every interior and the
  itf1788 input rule, which opens an unbounded end, is what makes `interior [1, infinity]
  [0, infinity]` true

### comparisons

* `< <= > >=` are pointwise. the result is a `TruthSet`: literally the set of truth values attained by
  `a op b` over all `a ∈ A, b ∈ B`, so one of `{}`, `{T}`, `{F}`, `{T, F}`. `__bool__` is True/False on
  the singletons and **raises** on `{T, F}` (ambiguous) and on `{}` (an empty operand attains no truth
  value — set theory, not a convention; `∅ < B` is not vacuously TRUE). `.certainly` and `.possibly`
  are plain bools for callers who do not want try/except. on `{}` they answer "every attained value
  is T" and "some attained value is T": `.certainly` is vacuously True, `.possibly` False; only
  `__bool__` refuses
* `==` and `__hash__` are structural set equality; `!=` is its complement. pointwise equality is a
  named method and is `{T, F}` for any non-degenerate `a == a` (document it; it surprises everyone)
* `==` does **not** coerce: against anything that is not a MultiInterval it returns
  `NotImplemented`, so `MI(5) == 5` is False. coercing would make `MI(5) == 5` True while
  `hash(MI(5)) != hash(5)`, which breaks dicts and sets
* two consequences to document, both correct: **no trichotomy** (`a < b` FALSE and `a == b` False does
  not make `a > b` TRUE), and **`a <= b` is not `a < b or a == b`** (pointwise vs structural)
* `sort_key` = the cut tuple, for structural ordering; `sorted()` raising on ambiguous intervals is a
  feature
* relations are defined **on cuts, not on values**: `before` = `A.end <= B.start`, `adjoins` =
  `A.end == B.start or B.end == A.start` (symmetric), plus disjoint / overlaps / contains / within /
  equals, and certainly_/possibly_ variants of before, after and equal. the class exposes before,
  after, adjoins, overlaps, contains, within, allen, allen_matrix and allen_relations; disjoint, equals and the modal variants
  are functions in `relations.py`. (`sup A < inf B` is wrong for `[1,2)` before `[2,3]`).
  relations return plain `bool` — they are set-level facts; only the pointwise `< <= > >=` return a `TruthSet`. so
  `before([1,2), [2,3])` is True and `before([1,2], [2,3])` is False, while `[1,2) < [2,3]` is
  `{T}` and `[1,2] < [2,3]` is `{T, F}`. for non-empty operands `before(A, B)` is exactly
  `(A < B).certainly`
* `allen(a, b)` on contiguous pieces only (raise otherwise). cuts make it finer than classical Allen:
  tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing one point (`[1,2] ∩ [2,3] = {2}`)
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
  of `B` iff the first of `A` is BEFORE the last of `B`), and never builds the matrix: `O(n + m)`
  in calls to `allen()` and in cut comparisons. for many pieces the set view is the answer; a
  1000 x 1000 matrix is a million references
* **the set view needs normalized operands; the matrix does not.** on out-of-order pieces the sweep
  would miss entries, so `relations.allen_relations` asserts `kernel.is_valid` of both cut tuples
  under `__debug__`, as `MultiInterval._wrap` does (the methods cannot reach it: every
  `MultiInterval` is valid). `relations.allen_matrix` takes any cut pairs in any order
* within one set the pieces never meet (`kernel.normalize` merges pieces that touch), so
  `A.allen_matrix(A)` is EQUALS on the diagonal, BEFORE above it and AFTER below it. `adjoins`
  stays a fact about the sets' ends: `[0, 1) | [3, 5]` has a piece that MEETS `[1, 2]`, and does
  not adjoin it
* numbers coerce to points and anything else is a `TypeError`, as every relation; the two classes
  mix. not on `DecoratedInterval`: relations go through `.interval`, as `allen` does
  (`tests/test_propagation.py::NOT_ON_THE_WRAPPER` lists both names). nothing new at the top level
  (`Allen` already is); the sparse view `(i, j, relation)` stays private (`_allen_pairs`)
* **interval orders** (D10, built at M13c 2026-09-26; `intervals/relations.py::weakly_less`,
  `::strictly_less`): 1788's `less` and `strictLess` are the methods `A.weakly_less(B)` (`inf A ≤
  inf B` and `sup A ≤ sup B`) and `A.strictly_less(B)` (both strict, except that two starts at
  -inf and two ends at +inf count, as 1788 writes it, so `(-inf, inf)` is strictly less than
  itself). they return bool and are **on the ends** as values, so a multi-interval's are its
  hull's and open or closed does not matter; `<` and `<=` stay pointwise (`MI(1,3) < MI(2,4)` is
  `BOTH`, while `MI(1,3).strictly_less(MI(2,4))` is True). a start at +inf or an end at -inf is a
  point there (`[inf]`, `[-inf]`) and is not strictly less than itself. two empty sets are weakly
  and strictly less than each other, an empty and a non-empty set neither, as in 1788

### arithmetic

* one generic applicator, **shape-then-attainment** — the modulo v3 lesson. v1's epsilon propagation
  is unsound, not just inelegant: `[0,1] * (2,3)` gives `(0,3)` but 0 is attained by `0 * 2.5`; true
  result `[0,3)`
    1. split at the domain points the op descriptor names (zero for reciprocal, division, abs,
       even and negative powers **and mul**). a piece with 0 strictly inside splits into `[lo, 0]`
       and `[0, hi]` with the zero in both halves; a piece that only touches 0, and a degenerate
       `[0]`, stay whole. modulo is not a descriptor: `modulo.py` is its own pipeline over
       sign-pure pieces and reuses only `applicator.split_pieces`
    2. locations first, all endpoints treated closed: corner min/max, correct for
       coordinatewise-monotone and bilinear ops **on a box whose discontinuities are corners**.
       on the extended reals mul is discontinuous at `(0, ±inf)` (`0·inf` is indeterminate while
       its neighbours are `±inf` and `0`), which is why mul splits at zero: after the split, ±inf
       and 0 are always piece endpoints, so every discontinuity is a corner and D2's
       limit-along-the-box rule applies. without it `[-1, 1] * [inf]` hulls to `[-inf, inf]`, but
       the attained set is `[-inf] ∪ [inf]`. add/sub need no split: their discontinuity
       `(inf, -inf)` is always a corner already
    3. closure per endpoint separately. the corner-flag rule (closed iff both operand endpoints
       closed — zero cost) is valid only for a **finite** result endpoint of an op that is
       injective in each argument there. every other endpoint is decided per split box by the
       **face rule**: attained at a corner whose operand endpoints are all closed, or along a flat
       edge whose fixed coordinates are closed ends (`applicator.py::_attained`,
       `modulo.py::_attained`). the union of the per-box attained sets is the same as deciding
       against the full operands:
        * **every infinite result endpoint.** add/sub/mul are flat at ±inf (`inf + y = inf` for
          every finite `y`), so the corner-flag rule is wrong there: `[inf] + (1, 2)` would give
          `(inf, inf)` = `∅` instead of `[inf]`, and `[1, inf] + [0, 1)` would give `[1, inf)`
          instead of `[1, inf]`. the rule: an infinite result endpoint is attained iff a closed
          infinite operand endpoint combines with some non-indeterminate partner, or a pole at a
          closed zero produces it (reciprocal, div, negative powers). an instance of the face rule
        * flat spots at finite values: mul at 0, pow, min/max-like
    4. union the piece results, normalize
* op descriptor (`applicator.py::OpDescriptor`): name, the pointwise `fn` (None where the op has
  no value), monotonicity directions (2-corner fast path for add/sub for free), split points,
  `pole(args, dirs)`, an optional `attained` override of the face rule, the `rounded` pair.
  infinite corners are evaluated by the ops' pointwise functions, which return exact values
* **division is computed as its own op** (`a / b` at the corners), not as `a * (1/b)`: the
  reciprocal identity defines the semantics (splits, poles, D2 corners), but evaluating through it
  would round twice on floats
* division: split the denominator at zero, direction rule as above: for x ≠ 0, `x / 0` is the
  infinity with the sign of x times the side of zero the divisor's piece lies on
  (`ops.py::_div_pole`). `A / ∅ = ∅`. exact operands divide as Fraction (see number types); a mixed
  exact/float pair is computed exactly and rounded once
* floordiv: enumerate integer points below a size cap, else hull + warning — never silently drop
  openness. the enumerator is `modulo.py::floor` (public as `MultiInterval.floor()`), the cap
  `FLOOR_ENUMERATION_CAP = 1000`, the warning `HullWarning`, also for any unbounded piece. at a
  zero divisor `//` follows div's poles (`[1] // [0, 1]` holds inf), no clipping. (v1's
  `[1,2) // 1` = `[1,2]` is wrong; should be `[1]`)
    * `floor(A / B)` over the finite divisors, with the quotient taken exactly and only the integers
      made float: a rounded quotient can cross an integer and the floor turns that ulp into a whole
      unit (`1 // 0.001` is 999; the float `1 / 0.001` is 1000.0)
    * **at an infinite divisor `//` is the limit, not `floor(div)`** (owner decision 2026-09-24, as
      python): `x // inf` is -1 for x < 0 and 0 for x >= 0, mirrored for -inf. the mathematical
      oddity: `-5 / inf` is 0, a point with no side, so `floor([-5] / [inf])` = `[0]` while
      `[-5] // [inf]` = `[-1]` = the limit of `floor(-5 / y)`. the limit keeps the infinite point
      continuous with its neighbours (`[-5] // [1, inf]` = `[-5] // [1, inf)`; floor(div) would add a
      stray 0) and makes `divmod(-5, inf)` = `(-1, inf)`, both parts limits of the same finite pairs.
      `x = q * y + r` cannot hold at y = inf either way (`-1 * inf + inf` has no value). it is the
      `1/[0]` story again: the direction of approach is lost at a degenerate point
    * `inf // 3` is `inf` (the limit; python gives nan); `[±inf] // [±inf]` is indeterminate
* **infinite operands in `%` and `//` follow one rule**: a pair's value is the limit of its finite
  neighbours' values, and a pair with no limit has none. D8 (below) and the `//` bullet above are
  instances of it; python agrees except where its float arithmetic gives nan for a limit that exists
* modulo: the v3 far-edge algorithm, `references/modulo-derivations/claude-fable/`, for every sign
  combination (built 2026-09-24, M7; proof in `proof-all-quadrants.md`). Q2's left edge is
  `modulo.py::_scalar_mod_interval_negative`; a negative divisor goes through the antipodal identity
  (`modulo.py::_box`); each located piece's ends are decided before the union, with exact O(1)
  attainment (`_attained`, `_holds_multiple`); operands crossing zero split into sign-pure
  pieces; a divisor touching zero drops 0 with
  `DomainClippedWarning`
    * infinite operands (D8, owner-confirmed 2026-09-24): a dividend of ±inf has no value (python gives `nan`) and is dropped
      with `DomainClippedWarning`; a finite dividend mod a divisor of ±inf follows python's scalar
      result, which is also the limit along the box (`3 % [inf]` = `[3]`, `-3 % [inf]` = `[inf]`,
      `0 % [inf]` = `[0]`)
* **rounding** is a property of the type, never a flag or context manager (wrapper types add
  meanings, flags change what existing objects mean). `MultiInterval` rounds a float result to
  nearest (the descriptor hook's identity default: python's float arithmetic).
  **`OutwardMultiInterval`** (M12) is the subclass that rounds outward: its class attribute
  `_outward` reaches every op as `outward=`, and the `ops.OUTWARD` descriptors evaluate a float corner
  exactly (each float as the Fraction it denotes) and round it down for a low end and up for a high
  end (`rounding.round_rational`). that is the tightest float enclosure (pown's float corners past
  `elementary.EXACT_POWER_LIMIT` bits reach the same doubles through `elementary.rounded_pow`:
  "power" below). the same doubles can come
  from gmpy2/mpfr, faster (the backend, "elementary and step functions" below); it never makes one
  tighter or looser. mixed with a `MultiInterval` on either side, the result is outward: the
  subclass overrides every reflected dunder, which python requires before it tries the right
  operand first. int and Fraction are exact and never rounded. `mod`, `floordiv`, `fma` and the step
  functions have no hook: they compute exactly and round once (`rounding.round_piece`), to nearest
  or outward by type. poles never go through the hook, and a float piece that rounding squeezes to
  one point keeps that point, closed
* **flags at rounded ends**: outward, attainment is decided on exact values (an `OUTWARD`
  descriptor's `fn` is exact too; pown's, past that limit, is a marker equal to nothing, which is
  the exact value's answer, `ops._NOT_A_DOUBLE`), so an end that directed rounding moved is **open** — nothing
  attains it (`OutwardMultiInterval(0.1) + 0.2` = `(0.3, 0.30000000000000004)`), and an end that is a
  double already keeps its flag. to nearest, flags are conservative, not a promise (the suite checks
  the nearest mode against the closure). an irrational value of an exact operand is its tightest
  float enclosure, open at both ends, in both classes (`sqrt([2])`): an exact operand never loses
  its true value
* **power** (D11, built at M13d 2026-09-26; `intervals/multi_interval.py::MultiInterval.__pow__`,
  `::__rpow__`): a number exponent with an integral value (int, or a float or Fraction equal to one;
  never bool) is 1788's **pown**, over every base, as python's numbers do (`[-3, 1] ** 2.0` =
  `[0, 9]`). every other real exponent and **every `MultiInterval` exponent**, `[2]` included, is
  1788's **pow** (`functions.pow_`, "elementary and step functions" below), so `[-3, 1] ** [2]` =
  `[0, 1]`. `b ** A` for a real `b` is `MultiInterval(b) ** A`, in `A`'s class, and a subclass's
  `__rpow__` keeps `MultiInterval(2) ** OutwardMultiInterval(...)` outward. 3-argument `pow` is a
  `TypeError`: dropped (not 1788; v1 had it on integers only). pown's float corners never build a
  power longer than `elementary.EXACT_POWER_LIMIT` bits (pown-huge, 2026-09-29;
  `intervals/ops.py::_power_descriptor`): outward, a float corner's power is exact while
  `elementary.exact_pow` builds it and otherwise `elementary.rounded_pow` in each direction (1788's
  pow route, the same doubles), with attainment against `ops._NOT_A_DOUBLE`, a marker equal to
  nothing (such a power is neither a double nor a midpoint); to nearest, python's `float ** int`
  while `|n| <= 2 ** 53`, past it `rounded_pow` to nearest with the int n's parity (python rounds n
  to a double there: `M(-1.0) ** (2 ** 60 + 1)` was `[1.0]`). int and Fraction operands are still
  exact, so `M(2) ** 2 ** 60` does not finish (open question Q-exact, plan §2 "pown-huge")
* **reductions** (M13h, 2026-09-26; `intervals/reductions.py`): 1788's `sum`, `sumAbs`,
  `sumSquare`, `dot` as `sum_(xs)`, `sum_abs(xs)`, `sum_sqr(xs)`, `dot(xs, ys)`, exported from
  `intervals`. point ops over any iterable of real numbers, not interval ops: each operand is held
  exactly, the value is computed as a Fraction and rounded once to a float, to nearest (ties to
  even) by default or by the keyword-only `rounding='down'` / `'up'`, so the operands' order never
  matters. the result is always a float, never `-0.0`; the empty sum is `0.0`. ±inf are points: a
  sum reaching one infinity is that infinity in every direction. a `nan` operand, `inf + -inf` and
  `0 * inf` raise `ValueError` (1788 answers `NaN`; ours follows the constructors' `nan` rule and
  D9's empty-set rule), as do sequences of different lengths in `dot`
* **cancellation** (D13, built at M13f 2026-09-26; `intervals/ops.py::cancel_minus`, `::cancel_plus`):
  1788's `cancelMinus` and `cancelPlus` as the methods `A.cancel_minus(B)` and `A.cancel_plus(B)`.
  `cancel_minus` is the **Minkowski difference**, the largest `X` with `B + X ⊆ A` (the `+` above),
  defined on any multi-intervals, open or closed ends, ±inf points included; `cancel_plus(B)` is
  `cancel_minus(-B)`. since `+` is the set of values of the defined pairs, the largest `X` is
  exactly the set of the `x` with `{x} + B ⊆ A`. for connected, closed, bounded operands with
  `wid A ≥ wid B` it is 1788's `[a1 - b1, a2 - b2]`; where 1788 answers entire as "no answer"
  (`A` narrower than `B`, an unbounded operand) ours is a real set, often `∅`
  (`cancel_minus((-inf, -1], [-1, 5])` = `(-inf, -6]`), and an empty `B` fits every `x`, so the
  answer is `[-inf, inf]` (for `A = ∅` too, where 1788 answers `∅`)
    * the algorithm (derived in the docstring): a finite `x` fits iff `B`'s infinite points are in
      `A` (a finite `x` leaves them where they are) and every piece `q` of `B`'s reals, shifted by
      `x`, lies inside one piece `p` of `A`'s reals, the pieces being the connected components:
      an intersection over the `q` of a union over the `p` of `[p1 - q1, p2 - q2]`, each end
      closed unless `p` is open there and `q` closed; an unbounded side of `q` needs the same side
      of `p` unbounded. `inf` fits iff `inf ∈ A` or `B = [-inf]` (`inf + -inf` has no value, so
      `{inf} + [-inf]` is empty and fits); `-inf` mirrors it
    * **rounding**: computed exactly and rounded once, like `fma`: to nearest in `MultiInterval`,
      outward in `OutwardMultiInterval`, where it is the tightest float enclosure of the exact `X`.
      that is 1788's answer (`cancelMinus [0x1.FFFFFFFFFFFFP+0] [0.1]` is the two doubles around the
      difference, `cancelMinus [max] [-max]` is `[max, infinity]`) and the class's promise: every
      `x` that fits is in the result. it is **not** a certificate that `B + X ⊆ A`, which a
      moved end can break by an ulp; exact operands (a float as `Fraction(f)`) give that. the
      result class is the receiver's, as for `fma`; no warning is emitted, since `∅` and
      `[-inf, inf]` are real answers

### elementary and step functions (M12, M13d, M13e, M16e)

* `functions.py`: sqrt, exp, exp2, exp10, log (with an optional base), log2, log10, sin, cos, tan,
  asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, and atan2 (M12); expm1, log1p (1788's
  logp1), cbrt, rootn(n), cot, sec, csc, acot, coth, csch, sech, acoth, hypot and pow (M13d, D11);
  methods of the class, python's name where python has one and 1788's otherwise. the
  result is the set of values attained, ±inf ordinary points where the function has a limit there
  (`exp(-inf)` = 0, `tanh(inf)` = 1, `atan(inf)` = pi/2, `acot(-inf)` = pi, `coth(-inf)` = -1)
    * **domain**: points with no value are dropped with one `DomainClippedWarning` (below 0 for sqrt,
      the logs and an even root, below -1 for log1p, outside [-1, 1] for asin acos atanh, inside
      (-1, 1) for acoth, below 1 for acosh, ±inf for the six trigonometric functions). a domain's
      end is a point of it wherever the function has a limit there from inside: `log(0)` = -inf,
      `atanh(±1)` = ±inf, `log1p(-1)` = -inf, `acoth(±1)` = ±inf, `rootn(0, -2)` = inf. the domain
      reaches those ends from one side only, so no second limit disagrees (1788 drops them:
      `log([0])` is empty there, `[-inf]` here)
    * **shape**: each function is monotone and continuous on each piece once split where its
      direction changes (cosh and sech at 0), so a piece maps to the piece between its ends' images.
      sin and cos add ±1 for an extremum strictly inside a piece (found by `elementary.floor_over_pi`,
      exact: only cos at 0 has a rational extremum); tan splits a piece holding a pole into both
      sides, with both infinities attained, as `1/x` does at a zero inside a piece. cot, csc and sec
      (`functions.py::_Function.reciprocal_trig`) cut a piece at its poles and extrema into
      monotone segments, each between the values at its ends (a pole inside gives both
      infinities, an extremum its ±1, both attained; three poles inside hold a period, so the whole
      range)
    * **a pole at 0 with a side each way** (M13d): coth, csch, cot, csc and an odd negative root
      follow `reciprocal`: a piece ending at 0 takes the one-sided limit there, closed iff the piece
      holds 0 (`coth([0, 1])` = `[coth 1, inf]`, `coth((0, 1])` = `(coth 1, inf)`), a piece with 0
      inside gets both infinities, and the point 0 alone has no value: it contributes nothing and
      warns once (`IndeterminateResultWarning`, as `1/[0]`). `A ∪ B` can then gain an infinity that
      neither image had (`[-1, 0) ∪ [0]`), the same trade D2's rule makes for `1/x`
    * **acot** is `pi/2 - atan x`, continuous and falling from pi to 0 (fi_lib's, whose vectors are
      the only ones), not `atan(1/x)` with its jump at 0
    * **pow(A, B)** (`functions.pow_`, 1788's pow): defined for x > 0, and x = 0 where y > 0
      (`0 ** y` = 0); every other pair is dropped with one `DomainClippedWarning` (a negative base,
      `0 ** y` for y <= 0). the base splits at 0 and 1, the exponent at 0: the power is 0 at x = 0,
      1 at x = 1 or y = 0, and elsewhere monotone in each coordinate, so a box's ends are two
      corners (as atan2's). ±inf are points where there is a limit: `inf ** y` is inf for y > 0 and 0
      for y < 0, `x ** ±inf` is 0 or inf by the side of 1; `1 ** ±inf` and `inf ** 0` have none, so a
      box that is one of them contributes nothing and warns (`IndeterminateResultWarning`). a
      corner's value is attained iff the corner is in the box or on a closed infinite edge, along
      which the power is constant. exact operands give an exact end where the power is rational
      (`[4] ** [1/2]` = `[2]`; `x ** (a/b)` is rational iff x is a b-th power), else the tightest
      float enclosure; float operands round once, to nearest or outward
    * **hypot(A, B)** (`functions.hypot`): the sums of squares as an exact set (float operands as the
      rationals they are), then one square root, rounded once like `fma`
    * **values**: `elementary.py` computes every value in pure python, correctly rounded in all three
      directions: fixed-point interval enclosures with bounded errors, refined by ziv's strategy
      until both ends round alike. it terminates because the rational values are exactly the known
      ones (`exp(0)`, `sqrt` of a square, `log2` of a power of 2, `2 ** int`, ...) and every other
      value is irrational (lindemann-weierstrass, gelfond-schneider). so results are the same on
      every platform; libm is not correctly rounded (this laptop's UCRT `acosh` is 2 ulp off near 1,
      measured 2026-09-25). `exp2`/`exp10` of an int past `elementary.EXACT_POWER_LIMIT` (100000)
      are rounded rather than built exactly, and so is a rational power longer than that many bits
      (`elementary.exact_pow`), which ziv's loop still settles, since a value that long in lowest
      terms is neither a double nor a midpoint of two. M13d's functions avoid cancellation where
      they need it: expm1 from exp at extra precision near 0, log1p as the log of the exact `1 + x`,
      coth and csch through expm1, acot as `atan(1/x)`, roots and pow as `exp(ln x / n)` and
      `exp(y ln x)`; pow decides overflow and underflow from a bracket of `ln x` good to a factor of
      3.1, before any series
    * **the backend** (M16e, 2026-09-28): which code picks the rounded double, the pure path
      (`python`) or gmpy2/mpfr (`gmpy2`, `intervals/_gmpy2.py`). **the default is the pure path**; the
      environment variable `INTERVALS_BACKEND`, read once at `import intervals`
      (`intervals/backend.py`), selects: unset, `''` or `python` the pure path (gmpy2 never imported);
      `gmpy2` gmpy2, and an `ImportError` at import if it is missing or below the floor (gmpy2 2.3 with
      MPFR 4.2); `auto` gmpy2 if it imports and `2.3 <= version < 3`, else the pure path, silently;
      anything else a `ValueError`. no public setter (rounding is a property of the type, never an
      ambient mode); `intervals.backend.name()` says which, and is not exported from `intervals`
    * **the backend's contract: the same doubles and the same flags, faster.** it answers only "which
      double": every decision that is not a rounding (`elementary.exact`, `exact_pow`, `_beyond` and
      the range shortcuts of `rounded_pow`, the pi limits at ±inf, every flag and attainment in
      `functions`, `reverse` and the applicator) runs first and stays pure. it replaces the `_ziv`
      loop of `elementary.rounded`, `rounded_pow`, `rounded_angle` (the `(q, m)` of `functions._angle`)
      and `rounded_inverse_trig` (k = 0), and the rounding of the five `ops.OUTWARD` descriptors (add sub
      mul div reciprocal, keyed on the descriptor object). each answers a float or None (then the pure
      path runs), so a partial backend is correct by construction
    * **why one MPFR call is the correctly rounded double**: MPFR is correctly rounded in every direction,
      and a `gmpy2.ieee(64)` context is binary64 exactly (the subnormals applied with the ternary
      value). so the input must be exact: an MPFR function is called only on an mpfr equal to x, built
      at x's own bit length in a private context (x dyadic: every float, every int, `Fraction(3, 8)`).
      declined, hence pure: a non-dyadic x (`1/3`), `log` to a base, `acoth`, `rootn` with `n <= 0` or
      `n >= 2**31` (gmpy2 takes n as a C `unsigned long`, 32 bits on windows, where it raises
      `OverflowError` from `2**32`; `2**31` is a margin that holds on every platform), `k pi + ...` with
      k != 0, the `pow{n}` descriptors, an operand past `2**20` bits in its numerator or denominator
      (past MPFR's exponent range, `2**30` on windows, a dyadic flushes to 0 with a ternary value of 0,
      to inf with 1, silently). native at a non-dyadic rational: atan, acot and the angles of atan2, as
      atan2 of two exact ints; the hook's mixed operands, as an exact mpq rounded once
    * **the backend's own rules**: every mpfr it builds names a context (a bare `mpfr(x)` reads the
      user's global context); `+ 0.0` is every function's last operation, after a sign (MPFR gives
      `-0.0` where the pure path gives `0.0`); a ternary value of 0 on an elementary result is a missed
      exact case and raises `ArithmeticError` as the pure loop does, in every direction (the pure loop
      answers where both ends of its enclosure round alike); a nan (an argument outside the domain)
      raises. untested on free-threaded builds; `backend._use`, the tests' switch, is a module global
    * **atan2(y, x)**: the angle in [-pi, pi]. no -0, so the negative x axis is at pi and points just
      below it near -pi; `atan2(v, ±inf)`, `atan2(±inf, u)` follow python (the limits). `(0, 0)` and
      `(±inf, ±inf)` have no angle: a box that is one of them contributes nothing and warns
      (`IndeterminateResultWarning`), a larger box takes the limits along its edges (D2's rule).
      each box splits into negative, zero and positive parts, where the angle is monotone in both
      coordinates
* `steps.py`: floor, ceil, trunc, round (ties to even, as python), round_ties_away and sign, plus
  `round(A, ndigits)` on a grid of `10 ** -ndigits`. one engine: a piece attains every grid value
  from the one just inside its start to the one just inside its end, listed up to
  `steps.ENUMERATION_CAP` (1000, shared across pieces), else their hull and a `HullWarning`.
  `modulo.floor` delegates here. `math.floor/ceil/trunc` and `round()` call these (the dunders
  return sets)
* `ops.minimum/maximum` (the class's `minimum()`, `maximum()`; builtin `min` needs a bool from `<`):
  descriptors with their own attainment, since min is flat where the other operand is out of reach.
  `ops.fma`: `add(mul(a, b), c)` computed exactly, rounded once
* **reverse ops** (M13e, D12; `intervals/reverse.py`, exported from `intervals`; built 2026-09-26:
  `sqr_rev(c, x=REALS)`, `abs_rev(c, x=REALS)`, `pown_rev(c, n, x=REALS)`, `cosh_rev(c,
  x=REALS)` here, `mul_rev`, `sin_rev`, `cos_rev`, `tan_rev`, `pow_rev1`, `pow_rev2` below): each
  is `{t ∈ x : f(t) has a value and f(t) ∈ c}` for the library's own f at a point,
  an exact multi-interval, where 1788 answers its hull. `x`
  defaults to `[-inf, inf]`, so ±inf are points with f's value there: `inf ** -2` = 0, so
  `pown_rev([0], -2)` = `[-inf] ∪ [inf]`; a point with no value (0 for n < 0, as `1/[0]` is empty)
  is in no preimage; `t ** 0` = 1 everywhere, so `pown_rev(c, 0, x)` is `x` or `∅`
    * **the engine** (the module docstring; built for the later reverse ops to reuse): a *branch*
      (`reverse.Branch`) is a piece of f's domain where f is continuous and strictly monotone,
      given from the value side: its image (each end closed iff attained) and its inverse g, exact
      where rational and correctly rounded otherwise (`reverse.named` wraps `elementary`'s `sqrt`,
      `rootn`, `acosh`). g is an order isomorphism of the image onto the branch, so the preimage of
      `c` is g applied piece by piece to `c ∩ image`, flags kept (`reverse.branch_preimage`). these
      four are even or odd in one branch on `[0, inf]` (`P ∪ -P`, `P(c) ∪ -P(-c)`)
    * **rounding**: an irrational end is its tightest float enclosure, open; a float end of `c` is
      a float operand, to nearest in `MultiInterval` (flags kept) and outward in
      `OutwardMultiInterval` (a moved end open), the functions' rule (`reverse._end` is
      `functions._Function.end`'s). the union is intersected with `x` **after** rounding, so the
      result never leaves `x`, but for one case, to nearest only (D26, 2026-09-30): a part of the
      exact preimage inside `x` that rounds wholly onto one double is that double, as a point, even
      an end `x` excludes (`reverse._keep_squeezed`; decision log "2026-09-30 revision"). the result is an `OutwardMultiInterval` if either operand is one
    * an empty operand gives `∅` and an `EmptySetPropagationWarning`, as the functions do; an empty
      answer from non-empty operands (no solution) warns nothing. a number is a point; `n` must be
      an `Integral` (not bool; numpy's ints since M16d, "numpy" below), else `TypeError`
    * **`mul_rev(b, c, x=REALS)`** (M13e, second part, built 2026-09-26): `{t ∈ x : t * y ∈ c for
      some y ∈ b}` with the library's `*`, so `t` is in iff `{t} * b` meets `c`. `0 * ±inf` has no
      value, so 0 is in iff `0 ∈ c` and `b` has a finite point; a finite `t != 0` comes from
      `v / y` (finite `v != 0` of `c`, finite `y != 0` of `b`), from `y = 0` if `0 ∈ b` and `0 ∈ c`
      (every finite `t`), and from `y = ±inf` where `c` holds the infinity it makes; `t = ±inf`
      where `c` holds the infinity it makes with a nonzero `y` (`reverse._mul_preimage`). 1788's
      reals have `0 * y = 0` for every `y` and no infinite points; `mul_rev([0], [0])` is `(-inf,
      inf)`, not `[-inf, inf]`. the quotients are the library's division (`ops.div`), so an end is
      rounded exactly where `c / w` would round it, and a point `b = [w]` (finite, nonzero) gives
      `c / w` in both classes. the class, warnings and `∩ x` after rounding as above
    * **`sin_rev(c, x=REALS)`, `cos_rev`, `tan_rev`** (M13e, third part, built 2026-09-26; D12):
      `{t ∈ x : f(t) ∈ c}` for the library's sin, cos, tan at a point, which have no value at ±inf
      and tan none at its poles, so neither is ever in a preimage (`tan_rev([inf])` is `∅`). the
      engine's branches, one per k (`reverse._Periodic`): sin `k pi + (-1)^k asin v`, cos `k pi +
      acos v` or `(k + 1) pi - acos v` by parity, tan `k pi + atan v`, each end correctly rounded by
      `elementary.rounded_inverse_trig`. a bounded `x` gets the exact pieces (`sin_rev([1/2, 1], [0,
      20])` has 4); per piece of `x`, as `steps.step` per piece of its operand, a part past
      `steps.ENUMERATION_CAP` pieces or an unbounded part is its hull, with one `HullWarning`, so
      the unary form over the default `x` answers `(-inf, inf)` wherever `c` has a solution. a `c`
      holding sin's or cos's whole image `[-1, 1]` is every finite t, one piece, never hulled; tan's
      poles leave a gap in every period, so no `c` escapes the hull there. an irrational end is its
      tightest float enclosure, open; a pole cannot be cut out of a piece, the two ends'
      enclosures around it overlapping
    * **`pow_rev1(b, c, x=REALS)`, `pow_rev2(a, c, y=REALS)`** (M13e, fourth part, built
      2026-09-26; D11): the bases `{t ∈ x : t ** y ∈ c for some y ∈ b}` and the exponents `{s ∈ y :
      t ** s ∈ c for some t ∈ a}`, `**` the library's pow at a point (`functions.pow_`: 0 for `0 **
      y`, y > 0; 1 for `1 ** y`, y finite, and `t ** 0`, t finite and > 0; inf or 0 for `inf ** y`,
      y != 0, and for `t ** ±inf`, t not 1; nothing for a negative base, `0 ** y` with y <= 0, `1 **
      ±inf`, `inf ** 0`). a case per special point (a base 0, 1 or inf; an exponent 0 or ±inf,
      `reverse._pow1_preimage`, `::_pow2_preimage`), and the rest from boxes monotone in both
      variables, ends at two corners: `v ** (1/w)` through pow's own box rule
      (`functions._power_box`), so an end is rounded exactly as `**` rounds `v ** (1/w)`, and
      `log_t v` (`reverse._log_box`), `elementary`'s correctly rounded log to a base, exact where
      rational (`log_4 2` = 1/2). the float rule is pow's, per operand: any finite float end of
      either operand makes every end float. the class, warnings and `∩ x` after rounding as above

### empties and warnings

* empty operands propagate: `A + ∅ = ∅`, `A / ∅ = ∅`, etc. this is set theory (the image of an empty
  set), and 1788 and the other libraries agree. emit `EmptySetPropagationWarning` with a default
  `'ignore'` filter installed at import; solver code opts in with
  `warnings.simplefilter('error', EmptySetPropagationWarning)` as a tripwire (not in a contractor
  loop, where `∅` is the normal "no solution here" answer). the import-time `'ignore'` filters are
  process-global state, installed with `append=True` so they sit at the *end* of the filter list:
  a filter the user installed before importing the library still wins, as do `pytest -W`, a
  `filterwarnings` ini entry and a later `simplefilter`
* the test suite turns the library's warnings into errors, so a property test that generates
  indeterminate or empty cases on purpose (isotonicity over degenerate `[0]`) opts out with a
  `filterwarnings` mark; the warning itself is pinned by its own `pytest.warns` test
* `DomainClippedWarning`, same pattern, wherever an op drops input points (`sqrt([-1, 4])`, `log`,
  modulo by a divisor touching zero). this is the cheap stand-in for 1788 decorations
* `IndeterminateResultWarning` as in "domain and semantics"
* `HullWarning` where an exact answer is replaced by its hull (floor/floordiv over the cap or an
  unbounded piece, and per D12 `sin_rev`, `cos_rev`, `tan_rev` past 1000 pieces or over an
  unbounded piece of `x`, so their unary form warns wherever `c` has a solution, but for a `c`
  holding sin's or cos's whole image `[-1, 1]`, one piece); shown by default, like
  `IndeterminateResultWarning`. all four subclass
  `IntervalWarning`, which the suite's `filterwarnings` entry turns into errors. one warning per
  call, attributed to the caller's frame (`applicator.py::warn`)
* no op raises on well-typed operands: empty, indeterminate and clipped cases warn. exceptions are
  for malformed construction (`[2,1]`, `nan`, bad types) and for questions with no answer: `bool()`
  of `{}` or `{T, F}`, `inf`/`sup` of `∅`, `allen()` of non-contiguous operands, `float()` of a
  non-point, `expand()` by an infinite or negative distance, a non-int exponent
* **no ambient modes**: no zero mode (there is no signed zero), no rounding flag (a type property), no
  `config.py`, no context managers. v1's mutable global `INFINITY_IS_NOT_FINITE` is deleted

### ieee 1788

* never a runtime mode (closed intervals only, connected only, infinity never attained, decorations
  everywhere — a flag would be `INFINITY_IS_NOT_FINITE` ×10 and multiply the test matrix). instead a
  conformance adapter in the test suite over itf1788 vectors:
    * **input rule**: a 1788 unbounded bound maps to our open-at-inf, because 1788 never attains
      infinity. measured 2026-09-24 (M9): reading unbounded bounds as closed at inf turned 0 of the
      847 vectors red, because D2's corner rule gives the same closed hull either way. re-measured
      2026-09-25 (M12): 4 of the 2932 vectors go red, all `isMember ±infinity [entire]`, which only
      the open reading answers false. it is the faithful reading, pinned by its own test
    * **precision rule** (added at M9): operands are the literals' nearest doubles held exactly,
      and our exact result is rounded outward to doubles before comparing, so a vector checks
      soundness and sharpness against 1788's tightest enclosure. functions return their tightest
      enclosure themselves. the round-to-nearest float path is not what the vectors test
    * **outward rule** (added at M12): every interval-valued vector runs a second time with the
      operands as floats in an `OutwardMultiInterval`, compared with no rounding by the adapter, so
      the library's own outward rounding must produce 1788's tightest enclosure
    * **output rule**: closed-hull **both** our result and the expected value before comparing
      (the input rule would otherwise read 1788's unbounded expected bound as open at inf, while
      our hulled result is closed there); absorbs multi-interval vs connected (`[1,2]/[-1,1]`:
      1788 entire, ours `[-inf,-1] ∪ [1,inf]`, hull = entire → match). a bool, a number or an
      overlap state is compared as it is
    * **numeric rule** (added at M13b): `mid`, `rad`, `wid`, `mag`, `mig`, `midRad` return exact
      numbers for the first pass's exact operands, which the adapter rounds as 1788 does (`mid` to
      nearest, `wid`/`mag` up, `mig` down, `rad` around the rounded midpoint:
      `tests/itf1788/test_itf1788.py::_numeric`, `::_mid_rad_1788`). a second pass
      (`::test_vector_float`) gives them float operands in both classes and compares the library's
      own numbers as they are. a `ValueError` (the empty set) is `NaN`
    * **reduction rule** (added at M13h): a reduction's result must already be the double 1788
      specifies (rounded to nearest) and is compared as it is; a `ValueError` from it is 1788's
      `NaN` (`tests/itf1788/test_itf1788.py::REDUCTIONS`, `::_reduce`)
    * **signal rule** (added at M13g): for the ops in `tests/itf1788/test_itf1788.py::SIGNALLED`
      (`b-textToInterval`, `b-numsToInterval`, and since M13g part 2 `d-textToInterval`,
      `d-numsToInterval`, `setDec`, `intervalPart`) ours and 1788's are compared as (closed hull,
      signal) pairs (`::_signalled`). an `UndefinedOperationError` raised is 1788's bare answer to
      invalid input, `[empty]` with `signal UndefinedOperation`; a `PossiblyUndefinedOperationWarning`
      emitted is `signal PossiblyUndefinedOperation`. the constructors have no interval operand, so
      they are not in the outward pass. `::test_signals_are_checked` fails if an op whose vectors
      carry a signal is not in `SIGNALLED`
    * residual divergence table: degenerate infinities, domain-clipped functions, decoration
      expectations, (added at M12) cut-based relations, and (added at M13f, approved with D13)
      **cancellation as a Minkowski difference**: where 1788's `cancelMinus`/`cancelPlus` answer
      entire as "no answer", ours is the real set of the fitting `x`, and for `[empty] [empty]` the
      whole line where 1788 answers `∅`; and (added at M13e and M13g, approved 2026-09-27 with D18)
      **tighter than the vector** and **exact parsing decides validity**. (`1/[0]` is not a row: both give empty.) keyed on the
      statement with its decorations stripped (`tests/itf1788/test_itf1788.py::key`) since M13a.
      measured 2026-09-26 (M13d; the current count, at M13's merge, is the "counts at M13's merge" bullet
      below): 19 files, 9542 statements; 7314 vectors of 83
      ops (every op in `OPS` has vectors), 6301 of them interval-valued and run twice, the 167
      numeric ones run twice more with float operands (13949 vector test items); 114 keys and 0
      unknown failures: 10 degenerate infinities (11 vectors), 5 cut-based relations (7 vectors),
      47 cancellations as a Minkowski difference (94 vectors, each also an outward item;
      `tests/itf1788/test_itf1788.py::_CANCELLATION_ROWS` and the two `[empty] [empty]` rows) and
      52 decoration expectations, the `[nai]` operands of implemented ops, generated in code (52
      vectors, 6 outward items and 12 float items). counted and skipped: 2228 statements of 28 ops
      not implemented yet (the reverse ops, the largest `powRev1` 429, then the text constructors
      and the decoration ops), each assigned to an M13 sub-task. M13d added the 1939 vectors of
      `pow` (1431) and the 13 other functions, all matching in both passes with no row. history:
      at M13f 5375 vectors of 69 ops, 4167 statements of 42 ops skipped. at M13a (2026-09-26) 4767 vectors
      of 54 ops, 4775 statements of 57 ops skipped; M13h added the 15 reduction vectors (4782 of 58,
      4760 of 53 skipped, 49 keys); M13b the 167 numeric ones (4949 of 64, 4593 of 47 skipped, 55
      keys); M13c the 184 of `less`, `strictLess`, `interior` (5133 of 67, 4409 of 44 skipped, 67
      keys); M13f the 242 of `cancelMinus`, `cancelPlus`. as of 2026-09-25 (M12): 2932 vectors of 54 ops from 7 files, 2438 of them
      interval-valued and run twice; 18 rows (`tests/itf1788/test_itf1788.py::DIVERGENCES`): 11
      degenerate infinities (`log`/`log2`/`log10` of an operand meeting the domain only at 0, `atanh`
      of one meeting it only at ±1) and 7 cut-based relations (`overlap [1,2] [2,3]` is `meets` in
      1788 and `overlaps` here, since the two share the point 2). counted and skipped: `pow` (real
      exponents, open), `less`, `strictLess`, `interior`, `isNaI`, `mid`, `rad`, `wid`, `mag`, `mig`;
      the reverse-op and cancel files are not vendored
    * **reverse ops** (added at M13e, 2026-09-26): the unary form (`sqrRev`) is the call with `x`
      omitted, so `x` is `[-inf, inf]`; the `*Bin` form passes `x` through the input rule. compared
      by the output rule, closed hulls: 1788's reverse op is by definition the hull of the
      preimage, so the closed hulls agree iff ours has that hull; the pieces inside are held by
      `tests/test_reverse.py`. of the 476 vectors of `sqrRev`, `absRev`, `pownRev`, `coshRev` and
      their `*Bin` forms, 420 match in both passes; 52 (26 keys,
      `tests/itf1788/test_itf1788.py::_POWN_REV_ROWS`) are **degenerate infinities**, the unary
      `pownRev` with n < 0 and 0 in `c`, where ±inf join (each matches with 1788's entire as `x`,
      pinned by `tests/test_reverse.py::test_the_rows_differ_only_at_the_infinities`); and 4 (2
      keys, `::_POWN_REV_LOOSE_ROWS`, `rev.itl:276`, `:277`) are under **a new category, "tighter
      than the vector", proposed at M13e, approved by the owner 2026-09-27 (D18)**: 1788's end for
      `2 ** (1074/7)` is one double outside the tightest enclosure, which ours is (arb, in
      `tests/test_reverse.py::test_pown_rev_is_tighter_than_the_vector`)
    * **reverse multiplication** (added at M13e, second part, 2026-09-26): `mulRev` and
      `mulRevTen` are `mul_rev(b, c)` and `mul_rev(b, c, x)`, compared by the output rule;
      `mulRevToPair` is `mul_rev(b, c)` too, under **the pair rule**
      (`tests/itf1788/test_itf1788.py::PAIRS`, `::_pair`): each of our pieces closed (rounded
      outward in the first pass), compared in order with the pair's non-empty intervals, piece by
      piece, which is stricter than comparing the unions. the pair vectors run in the outward pass
      too (`INTERVAL_VECTORS` includes them). all 539 vectors of the three ops match in both passes
      but the 6 with a `[nai]` operand (generated rows); no new row, no new category (2026-09-26)
    * **periodic reverse ops** (added at M13e, third part, 2026-09-26): `sinRev`, `cosRev`, `tanRev`
      are the call with `x` omitted, whose answer where `c` has a solution is the hull `(-inf, inf)`
      with a `HullWarning` (ignored in a vector), 1788's entire; the `*Bin` forms pass `x`. compared
      by the output rule. of the 136 vectors, 124 match in both passes; 12 (7 keys,
      `tests/itf1788/test_itf1788.py::_TRIG_REV_LOOSE_ROWS`) are rows under the **proposed
      category "tighter than the vector"**: one end of 1788's hull is one or two doubles outside
      the tightest enclosure of `k pi ± asin/acos/atan(v)`, which ours is (arb, in
      `tests/test_reverse.py::test_trig_rev_is_tighter_than_the_vector`)
    * **power reverse ops** (added at M13e, fourth part, 2026-09-26): `powRev1` and `powRev2` are
      `pow_rev1(b, c, x)` and `pow_rev2(a, c, y)`, every vector giving the domain, compared by the
      output rule. of the 804 vectors (`pow_rev.itl`), 802 match in both passes; 2
      (`tests/itf1788/test_itf1788.py::_POW_REV_LOOSE_ROWS`, `pow_rev.itl:609`, `:642`) are rows
      under the **proposed category "tighter than the vector"**: for `a` in `[1/4, 1]` and `c = [2,
      inf)` the tightest hull is `[-inf, -0.5]`, which ours is, and 1788 answers `[entire]` and
      `[-infinity, 0.0]`, far looser, though its own vectors with `c = [2, 4]` answer -0.5 at that
      end (decided exactly in `tests/test_pow_rev.py::test_pow_rev2_is_tighter_than_the_vector`)
    * **all the reverse ops** (M13e done 2026-09-26; D12): the 1955 vectors of the 19 ops run, none
      skipped: 1879 match in both passes, 52 (26 keys) are the degenerate infinities of the unary
      `pownRev`, 18 (11 keys) the proposed "tighter than the vector", 6 have a `[nai]` operand
      (generated rows, M13g's). until M13's exit asserts `SKIPPED` empty,
      `tests/itf1788/test_itf1788.py::test_only_m13g_ops_are_skipped` holds every skipped
      statement to M13g's 9 ops, so a reverse op dropped from `OPS` goes red (removed at M13's
      merge, 2026-09-27, with M13g's pin: `::test_nothing_is_skipped` asserts `SKIPPED` empty)
    * (M13's merge, 2026-09-27) **decorated reverse ops**: given a `DecoratedInterval` operand, each
      reverse op is 1788's decorated one, the core's set on the intervals decorated trv
      (`intervals/reverse.py::_decorated`); every interval operand is then a `DecoratedInterval` or a
      number, a bare `MultiInterval` a `TypeError`. the 19 reverse ops are in
      `tests/itf1788/test_itf1788.py::PROPAGATED` (`::REVERSE`), so every decorated reverse vector
      (481, 2026-09-27) runs on `DecoratedInterval` operands with its decoration compared in both
      passes, but the 4 with a `[nai]` operand (rows, D16). a decorated pair is (pieces,
      decoration) on both sides, 1788's decoration being its non-empty intervals'
      (`::_pair_outcome`). 1788 decorates mulRevToPair's first interval as the decorated division
      `c / b` where `0 ∉ b`, though mulRev, the same set's hull, is trv there; ours is one op,
      `mul_rev`, trv: 52 rows under **decoration expectations** on a decoration alone, in both
      passes (`::DECORATION_ONLY`, keyed with the decorations; the set must match, only the
      decoration differs, `::check`)
    * (added at M13g, approved with D16, owner 2026-09-26) **no NaI: invalid input raises**: the
      package has no NaI, since a 1788 constructor given invalid input raises
      (`UndefinedOperationError`) and nothing else makes one. every statement that needs a NaI is a
      row: a `[nai]` operand or result (the 52 generated rows that were under "decoration
      expectations" until M13g, now 53 with `isNaI [nai]`) and every `isNaI` (generated,
      `tests/itf1788/test_itf1788.py::_NO_IS_NAI`); measured 2026-09-26, 66 keys, 68 vectors; at
      M13g's close (2026-09-26) 70 keys, 73 vectors, the decorated ops' `[nai]` texts and operands added
    * (M13g; approved by the owner 2026-09-27, D18) **exact parsing decides validity**: 1788
      lets a text constructor that rounds each bound first answer a literal whose bounds are within an
      ulp with `PossiblyUndefinedOperation`; ours reads the bounds exactly, so it returns the valid
      one with no warning (`ieee1788-exceptions.itl:18`) and raises on the three whose lower bound
      exceeds the upper as rationals (`libieeep1788_class.itl:136`-`138`). 4 keys, 4 vectors
      (`::_EXACT_VALID`, `::_EXACT_INVALID`); the three `d-textToInterval` twins will need the same
      (they have it since part 2: 7 keys, 7 vectors at M13g's close, 2026-09-26)
    * (M13g part 2, 2026-09-26) **decorated rule**: for the ops in
      `tests/itf1788/test_itf1788.py::DECORATED` (`d-textToInterval`, `d-numsToInterval`, `newDec`,
      `setDec`, `intervalPart`, `decorationPart`) a decorated operand is a `DecoratedInterval` and a
      decorated result is compared as (closed hull, decoration), so the decoration is checked
      (`::_ours`, `::_expected`); a raised `UndefinedOperationError` is the decorated flavour's
      `[nai]` with `signal UndefinedOperation`, so those vectors match and are no row
      (`::_nai_is_a_raise`). `newDec`, `setDec` and `intervalPart` run outward too, the set in an
      `OutwardMultiInterval`; only the constructors (`::CONSTRUCTORS`) do not. the other ops'
      decorated vectors: the propagation rule below (M13g part 3). the three `d-textToInterval` twins
      are rows under the category above (7 keys now), and three vectors whose literal is
      bounded as a rational but past the doubles (`libieeep1788_class.itl:165`, `:201`, `:204`:
      com here, 1788's `dac` for its binary64 hull `[max, inf]` or entire) are rows under
      **decoration expectations** (`::_BOUNDED_EXACTLY`)
    * (M13's merge, 2026-09-27) **counts at M13's merge**, the current count, replacing the counts at
      M13e and at M13g (`tools/itf1788_census.py`): 19 files; 9542 vectors of 111 ops, every
      statement of the 19 files (every op in `OPS` has vectors), 8306 of them interval-valued and
      run twice, the 167 numeric ones run twice more with float operands (17848 vector test items,
      plus 334 float items); 1624 vectors carry a decoration: 1105 of 61 propagating ops (481 of them
      the reverse ops', 174 pairs), checked with it in both passes, 398 of `BARE_PART` ops (their
      interval parts), 121 of the decorated ops. 185 keys (109 listed, the rest generated) with 271
      vectors, and 0 unknown failures: 76 no NaI (79 vectors), 47 cancellations (94 vectors), 36
      degenerate infinities (63 vectors), 11 tighter than the vector (18 vectors), 7 exact
      parsing (7 vectors), 5 cut-based relations (7 vectors), 3 decoration expectations (3
      vectors); plus 64 rows on a decoration alone, under decoration expectations: 12 in the plain
      pass only (`::PLAIN_ONLY`) and 52 in both passes (`::DECORATION_ONLY`). skipped: none, which
      `tests/itf1788/test_itf1788.py::test_nothing_is_skipped` asserts (M13's exit)
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = the
  community test framework and its `itl` vector DSL. all 19 `.itl` files of the maintained fork,
  oheim/ITF1788 at `b6ee1e2`, are vendored unmodified with its `LICENSE`, `NOTICE` and
  `COPYING.LESSER` (D15, M13a, 2026-09-26; they replaced the 7 from nehmeier/ITF1788 `e0e0d7e`).
  each file keeps its own licence: Apache 2.0 for the 11 `libieeep1788_*`, LGPL-2.1-or-later for
  `mpfi`, `fi_lib`, `c-xsc`, all-permissive for `ieee1788-constructors`, `ieee1788-exceptions`,
  `atan2`, `abs_rev`, `pow_rev` (`tests/itf1788/README.md`). they are test data: the wheel ships
  only `intervals/`. `git hash-object` of all 22 files equals the fork's blob at the pin (checked
  2026-09-26), and `tests/itf1788/.gitattributes` marks them `-text` so a checkout keeps the bytes
* decorations (`com/dac/def/trv`) are **not in the core; they are in a wrapper** (D16, built at
  M13g, 2026-09-26). they answer "was f defined and continuous on the whole input", which the result
  set cannot (`sqrt([-1,4])` = `[0,2]` either way), and only solver existence proofs need that. so
  `MultiInterval` stays undecorated, with `DomainClippedWarning` as its cheap stand-in, and
  `DecoratedInterval` (the next two bullets) wraps one with 1788's decoration, brought forward from
  the solver stack for the itf1788 vectors. 1788's `ill` and NaI are not built (owner, 2026-09-26,
  Q8): invalid input raises `UndefinedOperationError` instead
* (M13g part 2, 2026-09-26, D16) **the decorated type**, `intervals/decorated.py`: `DecoratedInterval`
  is a `MultiInterval` (any subclass, kept) with a `Decoration`, an enum `COM`, `DAC`, `DEF`, `TRV`
  ordered `TRV < DEF < DAC < COM` (so the weaker of two is `min`), with **no `ill` and no NaI**.
  immutable, hashable, equal iff both parts are, never equal to its bare set. 1788's ops:
  `DecoratedInterval(x)` is `newDec` (com for a non-empty set with no point at and no piece reaching
  ±inf, decided exactly, so `[10**400]` is com; dac for any other non-empty set; trv for ∅);
  `DecoratedInterval(x, d)` requires `d` to fit and raises `UndefinedOperationError` otherwise;
  `set_dec(x, d)` is 1788's `setDec`, which demotes instead (∅ gets trv, com on an unbounded set dac:
  `min(d, newDec's)`); `.interval` and `.decoration` are `intervalPart` and `decorationPart`;
  `text_to_decorated_interval` and `nums_to_decorated_interval` are the `d-` constructors, the bare
  ones plus the literal's decoration or newDec's. a decoration is a `Decoration` or its lower-case
  name; `ill` and any other name raise `UndefinedOperationError`, anything else is a `TypeError`.
  `str` is the set in our syntax then `_com`. propagation: the next bullet
* (M13g part 3, 2026-09-26, D16) **decoration propagation** (1788-2015 §11), `intervals/decorated.py`:
  `DecoratedInterval` has the core's point functions (`+ - * /`, `**` as D11, `reciprocal`, `abs`,
  `minimum`, `maximum`, `fma`, `hypot`, `atan2`, the elementary functions, `log(base)`, `rootn`, the
  step functions with `round(ndigits)`, `%`, `//`, `divmod`) and set operations (`& | ^ ~`,
  `difference`, `complement`, `hull`, `closed_hull`, `interior`, `cancel_minus`, `cancel_plus`; and,
  after the M13g review 2026-09-26, `union`, `intersection`, `difference`, `symmetric_difference`
  n-ary as the core's, `positive`, `negative`, `finite`, `expand`, the restriction `x[a:b]`). each
  computes the core's set on the intervals and decorates it (`::_propagate`): the op's local
  decoration on the box of the operands' sets is trv unless every point is in 1788's domain of the
  op, a set of **reals** (so an attained ±inf is outside every domain, even where the core gives it
  a limit), def unless the op restricted to the box is continuous, dac unless it is continuous at
  every point of the box (relative to the domain: `sqrt` at 0, `pow` at `x = 0` are com, as the
  vectors have them) and every operand is bounded, else com; the result is the min of that, every
  operand's decoration and newDec of the result (so com needs a bounded result). only the step
  functions (dac iff constant on each piece, com iff moreover no closed end is a jump), atan2 (the
  negative x axis: dac unless the box reaches it from below, then def), `%` and `//` (per pair of
  pieces, `floor(x / y)` one integer; com iff `x / y` never reaches it) have jumps inside their
  domains. **a multi-piece box is decided on the set**, so on each piece (pieces are apart): floor on
  `[1/4, 1/2] ∪ [5/4, 3/2)` is com though not constant on the hull. every decision is made on the
  exact set (float ends as the rationals they are); only newDec of the result sees rounding. set
  operations and cancellation are trv, as 1788 decorates intersection, convexHull and cancel*; the
  booleans and numbers are not on the wrapper (1788 defines them on the interval part:
  `.interval`). an operand is a `DecoratedInterval` or a real number (newDec's point); a bare
  `MultiInterval` is a `TypeError`. conformance: every decorated vector of these ops is checked with
  its decoration in both passes (`tests/itf1788/test_itf1788.py::PROPAGATED`); the booleans' and
  numbers' take the interval part (`::BARE_PART`). the exact plain pass keeps com where 1788's
  binary64 result overflows (`add [1,2]_com [5,max]_com` is `[6, 2 + max]` here, bounded): 12 rows
  under **decoration expectations** in the plain pass only (`::PLAIN_ONLY`, keyed with the
  decorations; the outward pass matches), measured 2026-09-26. the reverse ops (M13e, not here) are
  trv in 1788, `::_trivial`

### the 1788 layer (M16b, H3's second part, 2026-09-28)

* **a wrapper, never a mode** (the 2026-08-16 principle): `intervals/ieee1788.py` gives 1788's
  answers over the library and changes nothing in it. every set is computed by the library
  (`OutwardMultiInterval`, and `DecoratedInterval` over one); the layer converts in by 1788's input
  rule and out by its output rule, and has its own logic only where 1788 *defines* another answer
  than the library's set (cancellation, overlap, `mulRevToPair`'s decoration). no library module
  was edited. not exported from `intervals` and not imported by it: `from intervals import
  ieee1788`
* **one class for both flavours**: `ieee1788.Interval(lo, hi, decoration)`, immutable and hashable,
  a set in **1788's form** (empty, or one piece whose finite ends are closed python floats and whose
  infinite ends are open: `[1, +infinity]` is `[1.0, inf)`) and a `Decoration` or `None` (bare). the
  form is the class invariant, asserted under `__debug__` in `Interval._init`, which both
  `Interval.__init__` and the internal constructor `Interval._make` call, as `MultiInterval._wrap`
  asserts `kernel.is_valid`. a call's flavour is its
  operands', which must agree (1788 has no mixed operations: `TypeError`); a real number is a point
  of that flavour (newDec's if decorated); a library value (`MultiInterval`, `DecoratedInterval`) is
  no operand (`TypeError`; the operators return `NotImplemented`), since it would bypass the input
  rule. `x.to_set()` is the library value the layer computes on; `x.decoration`
* **the input rule** is the constructor: `Interval(lo, hi)` is 1788's `numsToInterval`,
  `literals.nums_to_interval` (exact, infinite ends open) then the output rule, so a non-double end
  is rounded outward (`Interval(Fraction(1, 10))` is `[0.09999999999999999, 0.1]`, `Interval(0.1)`
  the double). with a decoration it is **strict**, decided on the binary64 result
  (`Interval(1, 2 ** 1024, 'com')` raises, its hull `[1.0, inf)` being unbounded), as
  `DecoratedInterval(x, d)` and a literal `[1,]_com` are, where `text_to_decorated_interval`
  demotes the same overflow to dac because 1788's `d-textToInterval` does (the three
  `_BOUNDED_EXACTLY` rows). `set_dec` is the forgiving one. `Interval(None, hi)` is a `TypeError`
* **the output rule** is `from_set(s)`, public, for any library value: (1) drop the attained
  infinities (`s ∩ (-inf, inf)` on the cuts, explicitly: 1788's functions are functions of reals,
  so `log((-inf, 0])` = `[-inf]` is 1788's empty); (2) the hull; (3) its ends outward to doubles
  (`rounding.round_value`); (4) 1788's form; (5) a decorated value's decoration capped by newDec of
  the binary64 result (`[6, 2 + max]_com` exactly is `[6.0, inf)_dac`). `from_set(x.to_set()) ==
  x`. it encloses the set it is given: a to-nearest `MultiInterval`'s float ends are read as exact,
  so only an `OutwardMultiInterval` or an exact result encloses a computation
* **every function** is `from_set(<library op>(<operands' sets>))`, flavours checked, with the
  library's `EmptySetPropagationWarning`, `DomainClippedWarning`, `IndeterminateResultWarning` and
  `HullWarning` silenced inside the call (`warnings.catch_warnings` + four `simplefilter('ignore',
  c)`; process-global filter state, so not thread-safe, as `decorated._quietly` is not). 1788
  signals none of them (`[0] / [0]` is empty with no signal). `PossiblyUndefinedOperationWarning`
  goes through, so the base class `IntervalWarning` is never the one silenced;
  `UndefinedOperationError` is raised
* **where 1788 defines another answer**, the layer gives 1788's and the library keeps its own, so
  one 1788 name has two answers in the package, each module's docstring saying which:
  `cancel_minus` is entire as 1788's "no answer" (`a` empty with `b` bounded: empty; both
  non-empty and bounded with `wid a >= wid b` compared exactly as `Fraction`s: the library's
  Minkowski difference, then 1788's `[a1 - b1, a2 - b2]` outward; else entire; trv when decorated)
  where `MultiInterval.cancel_minus` is the Minkowski difference (D13); `overlap` returns
  `Overlap`, an enum of 1788's 16 states on the ends as extended reals (`[1, 2]`, `[2, 3]` `meets`)
  where the library's `allen` is on cuts (`overlaps`); attained infinities are dropped in the layer
  only. `mul_rev_to_pair(b, c)` is `(div(c, b), empty)` where 0 is not in `b` (1788's wording: the
  first decorated as the division) and otherwise the pieces of the library's `mul_rev` with a real
  point, in order, each through the output rule, trv (at most two for 1788-form operands; a third
  raises, never hulled). the library's `mul_rev` stays one op, trv
* **numbers** are python floats of the interval part: `inf` of the empty set `+inf`, `sup` `-inf`;
  `inf` of an interval whose lower end is 0 is `-0.0` and `sup` of one whose upper end is 0 `+0.0`,
  1788's rule (`libieeep1788_num.itl:34`; nothing re-enters the library, whose one zero stays);
  `rootn(x, 0)` raises `ValueError` for every `x`, the empty set included (the library's rule,
  "rootn(n) for every int n other than 0", kept; no vector has degree 0), while `pown_rev(c, 0)`
  answers (entire or empty), as the library's does;
  `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad` of the empty set raise `ValueError` (D9's answer
  where 1788 says NaN; the default built, pending Q13 (b)). booleans are of the interval parts;
  `is_member(nan, x)` and `is_member(±inf, x)` are false. the reductions are the library's own
  objects (`ieee1788.sum_ is intervals.sum_`; `sum_square` is `sum_sqr`)
* **names**: 1788's, transliterated to snake_case mechanically (`mulRevToPair` ->
  `mul_rev_to_pair`), a trailing underscore on a python builtin (`abs_`, `min_`, `max_`, `pow_`,
  `sum_`); `NAMES` maps 1788's own spelling to each function: 104 names, 1788's 102 and
  itf1788's `'d-numsToInterval'`, `'d-textToInterval'` for the decorated constructors, 104
  distinct functions. every 1788 op the library
  implements and nothing else: not the recommended `exp2m1`, `exp10m1`, `log2p1`, `log10p1`,
  `compoundm1`, `rsqrt`, the `*Pi` functions, NaI and `isNaI` (D16), the exact conversions, other
  formats than binary64 (an `AttributeError`, not a stub)
* **the class's own operators** are python's obvious spellings of a function: `+ - * /` (and
  reflected), unary `- +`, `abs()`, `**` as D11 reads it (an integral real exponent `pown`, any
  other real and an `Interval` `pow_`; the function `pown` itself takes an `int` only), `&`
  (`intersection`), `|` (`convex_hull`), `in` (`is_member`), `==`/`hash` structural (a bare and a
  decorated interval are never equal), `bool` (non-empty). **no ordering**: 1788 has four orders and
  none is the obvious one. `repr` evaluates back (`Interval(float('-inf'), 2.0)`); `str` is a 1788
  literal with python's shortest decimal of each end (`[0.1, 2.0]_com`, `[entire]`, `[empty]_trv`),
  read back by `text_to_interval` as an enclosure within one double at each end.
  `__array_ufunc__ = None`, the package's rule for every type when the layer was designed; since
  M16d the three core classes have numpy's hook ("numpy" below) and the layer's `Interval` still
  refuses numpy: the rule for the layer is owed (`HANDOFF.md` "still owed")
* **conformance: a third pass** (`tests/itf1788/test_ieee1788.py`) runs all 9542 vectors through
  the layer and compares **exactly** (no hull, no rounding, no input rule): an interval as its
  float ends and decoration after asserting 1788's form, a pair member by member, numbers as
  floats, booleans, `Overlap` and `Decoration` values as they are. operands are the literals'
  nearest doubles in `Interval(lo, hi, d)`, strict (the C++ tests' convention, not 1788's text
  reading); a `Fraction` becomes its float and an `int` stays an `int`. warnings are recorded and
  every one must be a `PossiblyUndefinedOperationWarning`; the readings, in order: an
  `UndefinedOperationError` is `signal UndefinedOperation`, a `PossiblyUndefinedOperationWarning`
  `signal PossiblyUndefinedOperation`, then a `ValueError` from a number or a reduction is `NaN`
  only where the vector expects `NaN`. its rows are the adapter's under three categories only, taken
  from the adapter's lists by reason: **94 keys, 104 vectors** (76 no NaI, 11 tighter than the
  vector, 7 exact parsing; 2026-09-28). the adapter's other rows (degenerate infinities, cut-based
  relations, cancellation as a Minkowski difference, decoration expectations incl. Q9's 52 pair
  rows and the 12 plain-only ones) all match through the layer: each stays a true statement about
  the library, and has a second reading, "1788's own answer is `ieee1788.<op>`, which matches"

### the solver stack (M15, H3's first part, 2026-09-27; M16a, several variables, 2026-09-28)

built: forward-mode automatic differentiation and an interval newton solver, the demonstration of
what multi-intervals are for (M15), then its n-variable form (M16a, the last bullets). what stays
of H3 is under "later" below.

* **`Dual`** (`intervals/autodiff.py`): a value and a derivative, each a `MultiInterval` (either
  class) or each a `DecoratedInterval`, never one of each. `Dual.variable(x)` seeds `[1]`,
  `Dual.constant(x)` `[0]`; a number or a bare set in an op is a constant. its ops are the
  arithmetic dunders, `reciprocal`, `abs`, `**` (a number exponent as `MultiInterval.__pow__` reads
  it; a `Dual` or set exponent as pow, `u ** v (v' log u + v u' / u)`; `u ** 0` the constant 1) and
  every elementary method with `log(base)` and `rootn(n)`, each with its chain rule computed by the
  library's own ops, so the derivative's set encloses `{f'(x) : x ∈ X, f differentiable at x}`:
  exactly for int and Fraction, outward in `OutwardMultiInterval` (to nearest in `MultiInterval`
  with float ends, which is not an enclosure). the quotient rule is `(u' v - u v') / v ** 2`, the
  square a pown, so `1 / x` gives `-1 / x ** 2` and never a positive value across the pole. no step
  functions, `%`, `//`, `minimum`, `maximum`, `fma`, `hypot`, `atan2`: not methods of `Dual`.
  `derivative(f, x)` is `f(Dual.variable(x)).derivative` (`[0]` if `f` returns a number)
* **an enclosure of f' is not a proof that f is differentiable** (`abs` at 0 has the value
  `sign(0) = 0`). with decorated parts the two decorations are that proof: every op, and every op
  of the chain rule's formula, is defined and continuous on the input iff both are dac or com, so
  `f` is C¹ there. each formula is undefined exactly where its op is not differentiable (`sqrt`,
  `rootn`, `cbrt`, pow at 0, `asin` at ±1, `acosh` at 1: a division by 0, trv; `abs` across 0:
  `sign` is def), which `tests/test_autodiff.py` pins op by op
* **`newton(f, x, *, tol=1e-10, max_steps=10_000)`** (`intervals/solver.py`) returns `Root(interval,
  unique)`s, disjoint, in order, each a connected piece of `x`, and every zero of `f` in `x` is in
  one: a branch and prune over the pieces of `x` (a multi-piece `x` is fine), in
  `OutwardMultiInterval` (int and Fraction stay exact). per piece: prune if `0 ∉ f(piece)`; where
  both decorations of `f` on a `Dual` of `DecoratedInterval`s are dac or better and the piece is
  bounded, a newton step `piece ∩ (m + mul_rev(F', -f(m)))` from `m`, the midpoint (as a float if
  one is in the piece); `mul_rev`, not `/`, because `[0] / [0]` is `∅` (D7) where the mean value
  theorem needs every `t` (`t * 0 = 0`). where `0 ∈ F'` the step is two pieces: the gap is cut in
  one step, where a connected interval type gets two intervals (1788's `mulRevToPair`) or their
  hull. **unique** when the newton set is non-empty, inside the piece's interior, and `0 ∉ F'` (not
  implied by the others for a multi-piece `F'` with an isolated 0); kept by every later step. a
  point piece where `f` is exactly `[0]` is a unique zero, and so is a closed end where it is (a
  zero on a split point is a closed end, where no newton set fits inside the interior)
* termination: newton goes on while it halves a piece, past `tol` for at most 8 steps (quadratic at
  a simple zero, so that proves it; linear at a multiple one, 3/8 a step for `x ** 2` at 0); a
  proved zero is narrowed until a step no longer narrows it; else bisection at the midpoint, or,
  on a piece spanning more than a factor of 16 in magnitude, at 0, ±1 or ±2 ** the mean binary
  exponent, with no newton step (so `[-inf, inf]` reaches the scale of its zeros in about a dozen
  splits: `x ** 2 - 2` on it in 57 evaluations, measured 2026-09-27); an unproved piece is output
  once its width is at most `tol` or it cannot be split, and past `max_steps` the whole stack is
  output as it is (still every zero enclosed)
* `f` takes one argument and uses the library's ops on it, with numbers as its constants (it is
  called with a decorated `Dual`, which refuses a bare set, and with a point as an
  `OutwardMultiInterval`). the library's warnings inside `f` are silenced: a piece the solver makes
  can be an indeterminate point (`1/[0]`), which is no news to the caller
* the direction tag under "later" was not needed: the C¹ gate keeps newton off anything holding
  ±inf as a point, and a range check on such a piece needs no tag
* **`gradient(f, xs)` and `jacobian(F, xs)`** (`intervals/autodiff.py`, below `derivative`; exported
  from `intervals`): n passes of `f`, pass j calling `f(c_1, ..., Dual.variable(x_j), ..., c_n)` with
  `c_k = Dual.constant(x_k)` and reading the derivatives: column j of the jacobian. `Dual` is not
  touched (no vector mode: a tangent tuple inside `Dual` would edit the 42 chain rules M15 pinned).
  `xs` is a list or a tuple of n sets or numbers, every set bare or every set decorated (checked up
  front: `f` need not combine the two, and `lambda x, y: x` would never meet the bare `y`); a number
  is a point of the first set's kind (a `MultiInterval` if every entry is a number). `f` takes n
  positional arguments and returns a `Dual` or a number (a constant: its partials `[0]`); `F` returns
  a list or a tuple of m of them, and `jacobian(F, xs)[i][j]` is `∂F_i/∂x_j`, rows as tuples, no
  matrix type. at n == 1 `gradient(f, [x]) == (derivative(f, x),)`, the same pass. decorated, the
  entries carry the C¹ proof in n variables, relative to the box: a value and every partial dac or
  better say every op and every op of its chain rule's formula was defined and continuous on the
  box, so `F` is C¹ on the box relative to the box, which is what the mean value theorem on the box
  needs (one-sided at its faces). not on an open set holding the box: `x ** 1.5` over `[0, 1]` is com
  with nothing below 0 in its domain, and `abs` over the point `[0]` has a dac derivative (review
  F1)
* **`solve(F, xs, *, tol=1e-10, max_steps=10_000)`** (`intervals/solver.py`, below `newton`) returns
  `RootBox(box, unique)`s (`box` a tuple of n connected `OutwardMultiInterval`s), pairwise disjoint in
  some coordinate, inside `xs`, sorted by the components' `sort_key`s, and every zero of `F` in `xs`
  is in one. `xs` as for `newton` per coordinate (a `DecoratedInterval` refused); n == 1 is `newton`
  exactly. a branch and prune over the product of the pieces of `xs`, per box: prune by range; a
  point is unique iff `F` is exactly `[0]` there; **the step** on a box with no wide component
  (unbounded, or spanning more than a factor of 16 in magnitude) where the n decorated passes over
  the box's **closed hull** `H` are all dac or better (krawczyk's theorem is for a compact box, and
  a box from `mul_rev` or a bisection has open ends): `J` over `H`, a point `m` (the float midpoint,
  else the exact one, so a component two doubles wide can still be stepped), `Y` the float inverse
  of `mid J` by gauss-jordan with partial pivoting (the identity where a midpoint is not a finite
  float or the matrix is singular: any real `Y` keeps the step valid), `Mx = Y J`, `b = Y F(m)`, then
  * **krawczyk's test**: every `K[i] = m[i] - b[i] + Σ_j (δ_ij - Mx[i][j]) (H[j] - m[j])` non-empty
    and inside `H[i]`'s interior proves exactly one zero in `H`, so in the box (`int H` is inside the
    box whatever its ends). `g(x) = x - Y F(x)` maps `H` into `K` by the mean value theorem row by
    row: brouwer gives a fixed point. uniqueness is argued on the exact `K*`, built from the closed
    range of the true partials over the compact `H`: `K* ⊆ K ⊆ int H` and `K*` compact make it
    strictly narrower than `H`, so `ρ(|I - Y A|) < 1` for every real `A` of the mean value theorem,
    `Y` is regular and `g` contracts. (the computed `K` may be as wide as `H`, its ends open: the
    strict inequality is not read off `K`)
  * **the narrowing, preconditioned interval gauss-seidel with `mul_rev`** (hansen and sengupta),
    on the box: row by row `Z[i] = Z[i] ∩ (m[i] + mul_rev(Mx[i][i], -b[i] - Σ_{j≠i} Mx[i][j] (Z[j] -
    m[j])))`, each row using the rows before it. `mul_rev`, not `/` (D7). where `0 ∈ Mx[i][i]` the row
    is two pieces: the sweep stops and the box splits there, the other components as narrowed so far.
    krawczyk proves and gauss-seidel narrows: krawczyk has no division, so a partial holding 0 teaches
    it nothing, which is where multi-intervals split
  * then `newton`'s rules: a unique box is narrowed while the step narrows it; an unproved one goes on
    while the step halves its width, past `tol` for at most `_PAST_TOL = 8` steps; else bisection: a
    wide component first, round robin from the last split coordinate + 1 (`x + y, x - y` on `REALS²`:
    15 calls, 2749 without), then the other components widest first, falling through past a component
    that cannot be split
* **before an unproved box is output**: (1) **its simplest point**, the simplest rational of each
  component's closed hull (`_simplest_between`, continued fractions), when inside the component (not
  on an unbounded one): if `F` is exactly `[0]` there it is output alone as a unique point, and the
  rest of the box goes back as up to 2n boxes. a zero at a simple rational sits on a split face or on
  a component a row made degenerate, where no `K` fits inside an interior. (2) else **krawczyk on the
  box inflated within its region**: each stack entry carries its region `R`, the part of the input
  it stands for, with every zero of `R` in the box. the work list starts with `R` = the box; a
  gauss-seidel narrowing keeps `R`; a split (gauss-seidel's or a bisection's) cuts the old `R` in the
  split coordinate only, between the neighbouring boxes, and the simplest point's rest boxes get `R`
  cut at the point. `H'' = ` the closed hull of (the box widened by twice its width, at least 1e-12
  relative, on each side) ∩ `R`, with its own jacobian and C¹ gate: `K(H'') ⊆ int H''` gives one zero
  in `H''`, in `int H'' ⊆ R`, so in the box, and the box ⊆ `H''` holds no other. why: one row that
  pins a coordinate (`y - 1/4`, `3y - 1`) makes the box degenerate or an ulp wide there at once, and
  krawczyk on the box can then never prove the textbook simple zero
* the warnings inside `F` are silenced as in `newton`. `F` is called with decorated `Dual`s, with boxes
  of `OutwardMultiInterval`s and with points
* known limits: a zero on a split face that is not a simple rational in every coordinate, and a
  singular zero, end as unproved boxes of width `tol`, often with unproved slivers beside them
  (sound, not proved). the cost is the library's arithmetic: a box costs n + 2 calls of `F` (the
  plain call, n decorated passes, the point), plus up to n + 2 when it is output unproved. a
  continuum of zeros is bisected to `tol` everywhere, and each of its boxes gives its simplest point
  and up to 2n rest boxes, so the output is up to 2n + 1 times the boxes of width `tol` (not a
  cascade: a rest box's closed hull holds the point, which is then its own simplest point and not
  in it). an exact end beyond the doubles (`[10 ** 400, 10 ** 401]`) makes the float
  preconditioner overflow `b` to `(1.8e308, inf)`, so such a box is bisected, not stepped (review
  F2, F3; numbers in M16a's record)
* **the direction tag stays under "later", now for n variables**: the step runs only on a box whose
  components are all bounded, where `F` is dac on the closed hull, so every real value and partial
  over it is bounded. the enclosures may still hold an open end at ±inf by overflow (`exp` over
  `[700, 720]` is `(1.0142320547350045e+304, inf)`, dac): every set op keeps every real point, so the
  real `A` and the real `F(m)` of the mean value theorem stay inside `J`, `Mx`, `b` and `K`, and no
  degenerate `[±inf]` arises from a finite real (overflow is open at inf, and D2's corner makes
  `(a, inf) * [0]` `[0]`). on a box holding ±inf as a point `F` is only evaluated for the range
  prune, where `1/(1/[inf])` is `∅` (D7) and the tag would make it `[inf]`: that changes which
  function `F` is at ±inf, not whether a zero of it is found (a zero is a point where `F` has the
  value 0, and with the tag `[inf] - 5` holds no 0 either). no test: a test that the solver behaves
  the same with and without a type that does not exist cannot be written;
  `tests/test_solve.py::test_overflow_box` pins the overflow case

### numpy (M16d, H3's second part, 2026-09-28)

numpy is optional: never imported at load (`intervals/numpy_compat.py` imports numpy inside the two
hooks, which only numpy calls), not in `[project]` dependencies nor the `[test]` extra (CI's jobs
install it beside the extra). a multi-interval is a *scalar* to numpy, one value of a number-like
type, never an array of numbers, so interop is four rules and one refusal:

* **a numpy scalar is a python number** to every op: numpy registers `np.floating` as
  `numbers.Real` and `np.integer` as `numbers.Integral`, so a numpy scalar goes through `_coerce`
  and `cuts.py::normalize_value` like any python number. `np.float64` is a `float`, rounded to
  nearest in `MultiInterval` and outward in `OutwardMultiInterval`; `np.float32` and `np.float16`
  are the doubles they hold, exactly (`np.float32(0.1)` is `0.10000000149011612`); the numpy ints
  are ints; `np.bool_` and `np.complex*` are refused, as python's bool and complex
* **a foreign real is its exact value** (`cuts.py::normalize_value`) where it has one to give. a
  foreign real is a `numbers.Real` that is no int, float or Fraction: numpy's scalars, gmpy2's
  numbers. a `numbers.Rational` (gmpy2's `mpq`) is exact by type, as a `Fraction` is; any other
  real is the float it equals where it is a double (so nothing a double can hold changes type), else
  its exact `as_integer_ratio()` as a `Fraction` or an int; a real with no `as_integer_ratio()` is
  `float()` of it, as before M16d. so an `np.longdouble` wider than a
  double (x86-64 linux, where CI runs: 64-bit significand) and a wide `mpfr` are exact, where
  `float()` rounded them to nearest and an `OutwardMultiInterval` result did not hold its operand
  (and `np.longdouble('1e4000')` became `inf`). the float path pays one `isinstance(value, float)`
* **integer arguments take any `Integral` but bool**, as `ops.py::power` does: `rootn(n)`,
  `round(ndigits)`, `round_ties_away(ndigits)`, `pown_rev(c, n)` (`functions.py::_check_degree`,
  `steps.py::step`, `reverse.py::pown_rev`), and the `log` base takes any real but bool, through
  `normalize_value` (`functions.py::_check_base`). `Dual.rootn` takes `int(n)` before `n - 1`
* **ufuncs** (`MultiInterval`, `OutwardMultiInterval`, `DecoratedInterval`, `Dual`:
  `__array_ufunc__ = numpy_compat.array_ufunc`): a numpy scalar on the left of an operator reaches
  numpy's ufunc and so the hook, which runs python's protocol **on our dunders only**
  (`np.float64(2) + A` is `A.__radd__(np.float64(2))`, the call python made before the hook
  existed; both operands ours, python's own operator, subclass first, so `np.add(M, O)` is `M + O`,
  an `OutwardMultiInterval`); `==` and `!=` fall back to identity, as python's do
  (`np.float64(2) == M(2)` is False). the other ufuncs are the method computing the set image of
  the ufunc's pointwise function: `sqrt cbrt exp exp2 expm1 log log2 log10 log1p sin cos tan sinh
  cosh tanh floor ceil trunc sign reciprocal` the method of that name, `arcsin ... arctanh` the
  1788 names `asin ... atanh`, `rint` `round` (ties to even, as numpy's), `square` `x ** 2` (pown,
  not `x * x`), `minimum maximum hypot` the method of whichever operand is ours, `arctan2(y, x)`
  `y.atan2(x)`; with both operands ours, the operators' subclass rule: `np.hypot(M, O)` is
  `O.hypot(M)` and `np.arctan2(M, O)` takes y as an `OutwardMultiInterval` first, so an outward
  operand keeps its rounding in either order (`M.hypot(O)`, called directly, is still M's), and
  with no subclass between them the first operand's method (`np.hypot(M, D)` a TypeError, as
  `M.hypot(D)`); and the unary operators (`negative positive absolute fabs`, `invert` the complement
  `~`). a class without the method (`Dual` has no `floor`) is a TypeError. **not mapped**, so a
  TypeError: `fmod` (C's truncated remainder, `fmod(-7, 2)` is -1 where `-7 % M(2)` is `[1]`),
  `fmin`/`fmax` (numpy's point is to ignore a nan operand; a nan is refused here), `float_power`,
  `deg2rad` and the like, `isnan` and the other predicates of a point, `matmul`, every other ufunc,
  every ufunc method but `__call__` (`reduce`, `outer`, `accumulate`, `at`) and every keyword
  (`out=`, `where=`, `dtype=`, `casting=`). the table is keyed by the ufunc *objects*, so a foreign
  ufunc sharing a name (scipy's) is not numpy's
* **an ndarray meeting one of ours** (`np.linspace(0, 1, 3) + A`, either order, any ufunc of the
  table with two operands) is elementwise into an object array of the array's shape, each element a
  python number (as `tolist()` gives it) meeting the object on the scalar path; an element with no
  answer raises (a nan element: `ValueError`, as `nan + A`; a bool or complex array: `TypeError`).
  `==` and `!=` never broadcast: `f == A` is False and `A in f` False, as before. a list is not an
  array here: `np.add([1, 2], A)` is a TypeError
* **arrays hold a `MultiInterval` as one element** (`MultiInterval.__array__ = numpy_compat.array`,
  a 0-d object array): `np.array([A, B])` has shape `(2,)` (before, numpy took A for the sequence
  of its pieces: a 64-deep array or a ValueError). `np.asarray(A, copy=False)` is a ValueError.
  `DecoratedInterval` and `Dual` have no `__len__` and were scalars to numpy already. a float dtype
  is numpy's cast, `float()` of each element, **to nearest in both classes**:
  `np.asarray(O(Fraction(1, 3)), dtype=float)` is the double nearest 1/3, as `float(O(...))` is,
  so a degenerate outward set flows into float code rounded to nearest
* **object arrays run numpy's own loops, not the table**: numpy calls the python operator or a
  method named after the ufunc on each element, so on `arr = np.array([A, B])` `arr + 1`,
  `np.sum(arr)`, `np.sin(arr)` work, `np.arcsin(arr)` is a TypeError (no method `arcsin`),
  `np.square(arr)` is `x * x` (looser than `np.square(A)`), and `arr == A` is False and `A in arr`
  False although `arr` holds `A` (identity: `==` is structural and does not broadcast).
  `np.round(A)` and `np.around(A)` are TypeErrors (not ufuncs: numpy's fallback looks for `rint`).
  `np.frompyfunc(MultiInterval.asin, 1, 1)(arr)` or an operand of ours reaches the table.
  numpy's own comparisons (inside its loops and functions) call `bool()` of a `TruthSet`, so they
  answer where the comparison is certain and raise where it is not: `np.sign(np.array([M(1, 2)]))`
  is `[1]` (numpy's number, not a set) and `np.minimum(np.array([M(0, 1)]), 2)` holds the element
  itself, while `np.sign(np.array([M(0, 1)]))`, `np.minimum(np.array([M(0, 1)]), 0.5)` and
  `np.clip(M(1, 2), 0, 1)` are ValueErrors (BOTH), `np.max`/`np.maximum` over an empty set a
  ValueError (no truth value), and `np.isclose(M(2), 2)` a TypeError (`TruthSet & bool`). `out=None`
  never reaches the hook (numpy drops it): `np.sin(A, out=None)` is `np.sin(A)`
* **pandas** (3.0.6, probed 2026-09-28, not tested): a `Series` meets ours as an ndarray does, so
  `pd.Series([1.0, 2.0]) + A` and `A + pd.Series(...)` are object Series, elementwise (a TypeError
  before M16d), and `pd.Series([1.0]) < A` a Series of `TruthSet`s; pandas broadcasts `==` itself,
  so `pd.Series([1.0]) == A` is a bool Series of `False`, as before M16d (not numpy's single
  `False`). a Series or a DataFrame column holds sets as elements (`pd.Series([A, B]) + 1`,
  `.sum()`, `np.sin(series)` run numpy's loops), and a numpy masked array keeps its mask
  (`np.ma.masked_array([1.0, 2.0], mask=[0, 1]) + A` masks the second element)
* **the array API standard and `__array_function__` are not built**: the standard is a namespace
  for arrays of fixed-size numbers, with elementwise `bool` comparisons and float special cases;
  ours are ragged sets with structural `==`, `TruthSet` comparisons and set images without nan.
  an interval *array* type is the D23 alternative, "later" if ever

### package layout

modules export pure functions over cut tuples; one class file on top binds the dunders. no mixins.
imports only point downward.

    intervals/
        errors.py          warning and exception classes
        numpy_compat.py    numpy's hooks: array_ufunc (the __array_ufunc__ of MultiInterval,
                           DecoratedInterval, Dual) and array (MultiInterval.__array__); numpy
                           imported only inside them, never at load (M16d); below the three classes
        cuts.py            Side, Cut, below()/above(), mirror, -0.0 normalization
        kernel.py          normalize sweep; union/intersection/complement/difference; membership;
                           interior (M13c); size; Builder (collect, sort once, sweep)
        fmt.py             format and parse cut tuples; regexes compiled at module level
        relations.py       TruthSet, pointwise compare, relation predicates, allen(),
                           the interval orders weakly_less strictly_less (M13c),
                           allen_matrix allen_relations of every pair of pieces (M16c)
        rounding.py        rounding an exact value to a double: nearest, down, up
        backend.py         which code picks a rounded double: INTERVALS_BACKEND, python (default),
                           gmpy2 or auto; imports _gmpy2 only when selected (M16e)
        _gmpy2.py          the gmpy2/mpfr backend: the same doubles as elementary.py and the
                           outward hook, faster, or None (then the pure path) (M16e)
        applicator.py      op descriptor, corner evaluation, closure pass, rounding hook
        ops.py             neg pos absolute reciprocal add sub mul div, power (int exponents),
                           minimum maximum as descriptors; the OUTWARD descriptors; fma;
                           cancel_minus cancel_plus (M13f)
        modulo.py          v3 far-edge mod / floordiv / divmod_ (floor from steps)
        steps.py           floor ceil trunc round round_ties_away sign: enumerate or hull
        elementary.py      correctly rounded elementary functions at one exact point
        functions.py       the elementary functions and atan2 over cut tuples
        reverse.py         the reverse ops, {t in x : f(t) in c}: sqr_rev abs_rev pown_rev cosh_rev
                           mul_rev sin_rev cos_rev tan_rev pow_rev1 pow_rev2 (M13e, D12); on
                           decorated operands trv (M13's merge); above decorated
        reductions.py      sum_ sum_abs sum_sqr dot over numbers: exact, rounded once (M13h)
        numeric.py         mid rad wid mag mig mid_rad of a set: exact, or rounded as 1788 (M13b)
        literals.py        1788's interval literals, parse_literal; text_to_interval and
                           nums_to_interval, 1788's bare constructors (M13g); above the class
        decorated.py       Decoration, DecoratedInterval (a MultiInterval and a decoration),
                           set_dec and the d- constructors (M13g); above literals
        autodiff.py        Dual, derivative: forward-mode autodiff over sets (M15); gradient,
                           jacobian: n passes, one variable seeded each (M16a); above decorated
        solver.py          newton, Root: interval newton over multi-intervals (M15); solve,
                           RootBox: a square system in n variables (M16a); above autodiff and
                           reverse
        multi_interval.py  the class and OutwardMultiInterval: immutable cut tuple; _coerce
                           (numbers and intervals only — strings go through an explicit
                           parse()); one-line dunders. arithmetic
                           owns `+ - * / // % **` and unary `- + abs`; set algebra is `| & ^ ~`
                           plus named methods (`difference` has no operator, because `-` is
                           arithmetic subtraction, as in v1)
        ieee1788.py        1788's inf-sup binary64 intervals, bare and decorated, over the
                           library (M16b); not imported by intervals
        time_interval.py   the same kernel over datetime/timedelta values (deferred, D4)
        __init__.py        public API, constants (EMPTY, REALS, ...)
    tests/
        oracles.py         sampling + attainment oracles, derived from the pointwise table only
        strategies.py      hypothesis strategies and probe points shared by the test modules
        exhaustive_modulo.py  exhaustive modulo differential, run by hand, not in the gate
        exhaustive_ops.py  exhaustive differential for + - * / reciprocal neg abs **, by hand
        itf1788/           vendored .itl files, itl.py parser, adapter + divergence table
        conftest.py        the fuzz profile (M14); does nothing unless HYPOTHESIS_PROFILE is set
        test_<module>.py   (ops split into test_ops_examples.py and test_ops_properties.py;
                           test_minmax_fma.py and test_outward.py for the rest of M12;
                           test_oracle_flint.py, the arb oracle for the functions, M14;
                           test_reductions.py, M13h; test_numeric.py, M13b;
                           test_orders.py, the orders and the interior, M13c;
                           test_cancel.py, cancellation, M13f;
                           test_literals.py, test_decorated.py and test_propagation.py,
                           1788's literals, the decorated type and propagation, M13g;
                           test_autodiff.py and test_solver.py, M15;
                           test_gradient.py and test_solve.py, several variables, M16a;
                           test_ieee1788_layer.py and itf1788/test_ieee1788.py, the 1788
                           layer and its conformance pass, M16b; test_relations.py gained
                           the allen matrix, M16c; test_numpy_compat.py, numpy, M16d;
                           test_backend.py, the backend differential, M16e;
                           test_pown_huge.py, pown with a huge exponent, pown-huge)

* only the two class files know the class; everything below takes and returns tuples. this removes
  the mixin return-type problem, keeps fmt below the class, makes every kernel function
  oracle-testable with tuples, and makes the time-interval port a thin second wrapper
* the consistency check is `MultiInterval._wrap` asserting `kernel.is_valid` under
  `if __debug__:` (v1 runs an O(n) scan at the top of nearly every public
  method; that is the real hot cost). correctness lives in the tests
* immutability retires the `inplace=` dual API. `interval.py` (the alternative debug implementation)
  is not ported; the sampling oracle does that job better
* v1 is **archived, never deleted**: all v1 modules moved unchanged to `archive/v1/` at M10
  (2026-09-25), with the old README, and stay as the reference until v2 works

### testing

* soundness fuzz for every op: `op(x, y) ∈ op(A, B)` for sampled `x ∈ A, y ∈ B`, on exact operands
  and on floats under identity and outward rounding
  (`tests/test_ops_properties.py::test_sound_float_identity_rounding`, `::test_sound_float_outward_rounding`),
  and outward rounding with the flags as given over subnormals, near-overflow magnitudes and ±inf
  (`tests/test_extreme_floats.py::test_outward_rounding_sound_on_extreme_floats`)
* v1 as a differential oracle: set operations (`tests/test_kernel.py::test_matches_v1`) and
  `A % scalar`, where ours ⊆ v1 and v1 − ours ⊆ {0} (`tests/test_modulo.py::test_matches_v1_mod_scalar`,
  `::test_v1_phantom_zero`)
* attainment checks for closure, on int/Fraction operands only
* algebraic properties that pin the cut encoding cheaply: `~~A == A`, De Morgan, the size tiling
  invariants, and for arithmetic:
    * isotonicity for every op: `A ⊆ B ⇒ f(A) ⊆ f(B)`
    * `f(A ∪ B) == f(A) ∪ f(B)` for every op without a pole: add, sub, mul, div in its dividend, neg,
      pos, abs, pow with n ≥ 0, and mod in both arguments (`tests/test_ops_properties.py::EQUAL_UNION`,
      `::test_power_union`, `tests/test_modulo.py::test_distributes_over_union`). splitting at zero
      does not break it; a pole does. for reciprocal, div's divisor and pow with n < 0 only
      `f(A ∪ B) ⊇ f(A) ∪ f(B)` (i.e. isotonicity): under the
      direction-from-the-piece rule equality fails with *any* value of `1/[0]`. counterexample
      `A = [-1, 0)`, `B = [0]`: `1/(A ∪ B)` = `[-inf, -1]`, but `1/A ∪ 1/B` = `(-inf, -1]`
    * `1/(1/A) == A` for every A with no degenerate piece at `0`, `inf` or `-inf` that is not
      unbounded at both ends while holding exactly one of ±inf. such an A gains the other infinity:
      `1/(-inf, inf]` = `[-inf, inf]` (0 is reached from both sides), which maps to itself. corrected
      2026-09-24; pinned by `tests/test_ops_properties.py::test_reciprocal_involution`
* the sampler must draw a closed ±inf endpoint with positive probability, and the attainment
  oracle must decide ±inf symbolically; otherwise the infinite-endpoint closure rule is untested
* **interior sharpness**, not just endpoints: soundness fuzz passes on any superset, and endpoint
  attainment checks cannot see a spurious interior point. so check that every gap of the true
  result is a gap of ours — on exact operands, sample points of `op(A, B)` and confirm each is
  attained (pins e.g. `[-1, 1] * [inf]` = `[-inf] ∪ [inf]`, not `[-inf, inf]`)
* sabotage each check once (flip one merge comparison) and watch it go red before trusting it
* itf1788 conformance through the adapter above
* elementary functions (`tests/test_elementary.py`): correctly rounded in all three directions
  against an independent oracle (the decimal module, whose exp, ln, log10 and sqrt are correctly
  rounded, and taylor series in decimal for the trig functions), and every enclosure holds the value
  at several working precisions. the constant series' error bounds are pinned directly; the taylor
  loops' bounds are covered by the interval rounding's slack around them, so no sampled value shows
  one missing, and they are argued in their docstrings instead
* an independent oracle for the functions (D14, M14, 2026-09-26): `tests/test_oracle_flint.py`
  checks all 19 against python-flint's arb, whose every value is a ball proven to contain the true
  one, a test-only dependency in the `[test]` extra. at a drawn float, int or Fraction point each
  directed end must be sound and sharp (no double strictly between it and the value, i.e.
  correctly rounded), nearest the right one of the two, a rational value exact and a closed point;
  at set level an exact operand's irrational value is the open one-ulp piece and an outward one is
  sharp and closed only where attained; plus fixed hard points, domain ends and limits, atan2 and
  `elementary.rounded_angle`/`floor_over_pi`. a comparison arb cannot decide retries at more bits
  (`test_oracle_flint.py::PRECISIONS`), then is rejected and counted (0 at default settings
  and under fuzz ×10, 2026-09-26)
* the reductions (M13h, 2026-09-26; `tests/test_reductions.py`): a random differential against
  Fraction arithmetic, each result checked from the definition of rounding on its neighbouring
  doubles (`::is_rounded`, not the package's rounding) in all three directions, float sums also
  against `math.fsum`, and the special values (±inf, `nan`, `0 * inf`) by rule, with the itf1788
  vectors as `@example`s
* the numeric functions (M13b, 2026-09-26; `tests/test_numeric.py`): exact operands against the
  definitions, `mag`/`mig` through `abs(A)`; float operands over the whole double range, each number
  checked from the definition of rounding (`tests/test_reductions.py::is_rounded`) and `rad` as the
  smallest covering double; soundness at sampled points (`mig <= abs(x) <= mag`, `x` within
  `mid ± rad`, `abs(x - y) <= wid`); isotonicity; the hull against the set; both classes equal. the
  itf1788 vectors can see no hull/set difference (each is one interval), so D9's set reading of
  `mig` is held by these properties alone
* the interval orders and the interior (M13c, 2026-09-26; `tests/test_orders.py`): both orders
  against 1788's quantified definitions, decided by brute force on a grid over 1788's reading of
  the hulls; on the ends (the hull and closed hull change nothing); soundness at sampled points;
  the interior against its definition at probe points and its laws (open, inside the set,
  idempotent, isotone, distributes over `&`), and `A.within(B.interior)` against a grid
  neighbourhood oracle. no vector can see the interior of a multi-interval or of a closed end at
  inf, so those are held by these properties alone
* cancellation (M13f, 2026-09-26; `tests/test_cancel.py`): the defining property decided
  completely on exact operands, `x ∈ A.cancel_minus(B)` iff `{x} + B ⊆ A` (the library's `+`) at
  every difference of an end of `A` and one of `B`, a point between each two, beyond each end and
  ±inf, which is soundness and maximality at once; `B + X ⊆ A` at set level;
  `C ⊆ (B + C).cancel_minus(B)`; isotone in `A`, antitone in `B`; a point `B` is subtraction;
  1788's formula; float operands: outward encloses the exact `X` tightly, nearest is `X` rounded
  once; soundness at sampled points in both classes. no vector has an open finite end or a
  multi-piece operand, so those are held by the properties alone
* the reverse ops (M13e, 2026-09-26; `tests/test_reverse.py`, `tests/test_pow_rev.py`): the
  defining property decided on exact operands, `t` in the result iff `t ∈ x` and `f(t) ∈ c` (for
  the binary ones, `{t} * b` or `{t} ** b` meets `c`, the library's op), at every end, the
  inverses of the ends, a point between each two, beyond, ±inf and around every float end, which
  is soundness and maximality at once, a rounded end's one-double slack aside; the largest set
  (`T ⊆ rev(f(T) ∪ more)`); isotone in every operand, distributing over unions, `x` only
  intersecting; symmetry (even, odd, `-b`, `1/c`); the relations between the ops (`sqr_rev` is
  `pown_rev(., 2)`, `pow_rev1([n], c)` is `pown_rev` on the bases, `pow_rev2([t], c)` is
  `c.log(t)`, `mul_rev([w], c)` is `c / w`); float operands in both classes; soundness at
  sampled points; D12's cap and hull for the periodic ones; arb for the irrational ends, and arb or
  exact arithmetic for every row under "tighter than the vector". no vector has an infinite point
  in an operand, an open end or a multi-piece operand, so those are held by the properties alone
* 1788's literals, the decorated type and propagation (M13g, 2026-09-26): `tests/test_literals.py`
  checks `nums_to_interval` against 1788's definition at probe points (soundness and maximality),
  every spelling of a value reading back exactly, the uncertain form against a `decimal` oracle, and
  that any text over the literal alphabet is one interval or raises `UndefinedOperationError`;
  `tests/test_decorated.py` newDec and `set_dec` against 1788's definitions written out in the test,
  with maximality (every decoration up to newDec's fits, none past it); `tests/test_propagation.py`
  every decorated op against an oracle decided by brute force from 1788's definitions on a grid
  (domain, continuity, jumps), float operands against the exact values of the same doubles, the min
  law and antitonicity in the box. no vector has a multi-piece operand or an attained infinity, so
  the set reading of decorations is held by these properties alone
* the decorated reverse ops (M13's merge, 2026-09-27; `tests/test_propagation.py`): each of the ten
  ops on decorated grid sets, `x` omitted or given, is the core op's set on the intervals, in its
  class, decorated trv (`::test_each_reverse_op_is_trv`); a bare set beside a decorated one is
  refused, a number is a point, the class is kept, the core's warning reaches the caller once
* autodiff and newton (M15, 2026-09-27): `tests/test_autodiff.py` checks every op of `Dual`, and
  random expression trees over them, against arb's taylor series (`arb_series`, python-flint): at
  an exact point of a drawn interval, f and f' are in the value's and the derivative's sets, and at
  a point the derivative is within 1e-10 relative, so a loose formula is caught; the decorations
  are dac inside each op's domain and not where it is not differentiable.
  `tests/test_solver.py`: on polynomials from drawn zeros, every zero is enclosed (under any `tol`
  and `max_steps`), a unique `Root` holds exactly one distinct zero, the roots are disjoint and in
  order; by example the first step's split, a non-C¹ function whose derivative's values would lose
  a zero (`abs(x) + x / 2 - 1/4` on `[-1, 3]`), close zeros, poles, unbounded and multi-piece input
* several variables (M16a, 2026-09-28): `tests/test_gradient.py` (the jacobian against arb series,
  one column per seeded variable) and `tests/test_solve.py` (constructed systems `A G(B x + c)` with
  every real zero known: exact fractions, or `r + Σ a sqrt(q)` decided by arb)
* the 1788 layer (M16b, 2026-09-28): the third conformance pass,
  exact, over all 9542 vectors (`tests/itf1788/test_ieee1788.py`); `tests/test_ieee1788_layer.py`
  checks every lifted function against a table of library calls written apart from the module's
  (`::LIBRARY`) and 1788's form on drawn operands of both flavours (ends from a pool with 0, ±1,
  subnormals, ±max, ±inf and drawn doubles), `from_set` against the output rule decided with
  `Fraction` and `math.nextafter` on drawn library sets (several pieces, attained ±inf, ints and
  Fractions past the doubles), cancellation against 1788's rule written in the test (equal widths
  drawn, and widths equal as floats but not exactly), `overlap` on the 729 pairs of a grid against a
  table keyed on the signs of the ends' comparisons and its converse, `mul_rev_to_pair` against the
  library's pieces on drawn bare operands (so the law "where 0 is not in b the pieces of `mul_rev`
  are `div(c, b)`" is drawn there) and, with com drawn, its sets tied to the bare pair's and its
  decorations pinned (div's where 0 is not in b, trv otherwise), the flavours (a call of numbers
  alone is bare), the refusal of library values, the warnings (silent, `PossiblyUndefinedOperationWarning` through, the
  filters unchanged after a call), names, `repr`/`str` read back, the operators
* the per-piece allen matrix (M16c, 2026-09-28; `tests/test_relations.py`): every entry against the
  `n x m` loop over the pinned `allen()` (`::allen_loop`), on operand pairs often derived one from
  the other (itself, its complement, hull, gaps, interior), so shared cuts, MEETS and MET_BY are
  common; the set view against the matrix's entries; the converse, a set against itself and the set
  relations as identities over the matrix; the empty shapes; 13 worked rows both ways; the plain
  loop's contract on unnormalized operands; the set view's cost pinned by counts, calls to
  `allen()` (at most `n + m - 1`, exactly the intersecting cells, never the matrix) and cut
  comparisons (`::_CountingCut`, at most `10 (n + m)`), and its refusal of out-of-order operands
* numpy (M16d, 2026-09-28; `tests/test_numpy_compat.py`, which skips without numpy): numpy never
  imported at load (a subprocess); 9 numpy scalar types in every operator derived from the classes'
  reflected dunders, both sides, against the python number of the same value (result, exception
  type and warning categories); every ufunc of the table against its method, both operands ours by
  the operators' subclass rule; the unmapped ufuncs, ufunc methods and keywords `TypeError`s; the
  elementwise path against the scalar path per element, numpy's float flags left alone; `==` never
  broadcasting; the foreign-real rule against stubs (`::Wide`, `::Rat`, `::Plain`), every float32
  bit pattern and the long double (which discriminates on linux CI only); numpy ints as integer
  arguments. in `tests/test_autodiff.py`, M15's `Dual ** r` hole against arb at 400 bits, bare and
  decorated (`::test_pow_number_exponent_derivative_encloses`)
* **the backend differential** (M16e, `tests/test_backend.py`): at every point it draws, each
  primitive three ways, the pure path under `backend._use('python')`, `_gmpy2`'s function directly
  (the same double and sign bit, and None exactly where the module's table says,
  `::declines_rounded` and its kin, written from this design), and the dispatch under
  `_use('gmpy2')` (which sees an argument dropped on the way); 15 edge classes (`::EDGES`, `::HARD`,
  `::_bound_cases`), the whole list again under a hostile gmpy2 global context; the `repr` of every
  set-level method under both backends; `::test_use_switches` guards that the two really ran
  different code, and `::test_use_restores` that the files after it run on the pure path again.
  gmpy2 is in `[test]`, so nothing skips. the rest of the suite runs on the pure
  path (the default); the build ran the whole gate once more forced to gmpy2
* a **fuzz profile** (M14, 2026-09-26): `HYPOTHESIS_PROFILE=fuzz` makes `tests/conftest.py` run
  every hypothesis test randomized, with no deadline, at `FUZZ_MULTIPLIER` (default 10; 100 until
  2026-09-27) times its own `max_examples`; unset, the conftest does nothing, so the gate keeps `default` locally and the
  derandomized `ci` under GitHub Actions. `.github/workflows/fuzz.yml` runs it on every push to
  `master` and on `workflow_dispatch` (not weekly since 2026-09-29), carrying `.hypothesis/` (the
  profile's own example database) between runs and uploading it with the log on a failure;
  `tools/prepush.sh` runs the same locally before a push

### later (not in v2.0)

* a **direction tag on a degenerate zero piece** if a solver ever needs `1/(1/[inf]) == [inf]`:
  metadata that `==` and hash ignore, created only by limits (`1/[±inf]`, `exp([-inf])`), consumed
  only by branch-at-zero functions. never a position in the order — that is what the signed-zero seam
  was. M15: not needed in one variable; M16a: not needed in n variables either (the argument is
  "the solver stack" above, its last M16a bullet; the overflow case pinned by
  `tests/test_solve.py::test_overflow_box`)
* a decorated wrapper type, with the solver. owner 2026-09-25: brought forward to
  `v2-implementation-plan.md` M13g, for the itf1788 decoration vectors; the core stays undecorated.
  built 2026-09-26 (`DecoratedInterval`, "ieee 1788" above); the solver uses it since M15 (the C¹
  gate)
* H3, the solver stack: **built**. forward-mode autodiff and newton's method 2026-09-27 (M15), and
  2026-09-28 (M16, on the owner's "get the rest of h3 done") the solver in several variables
  (M16a), a thin `ieee1788.py` with 1788's `mulRevToPair` as `ieee1788.mul_rev_to_pair` (M16b, Q9),
  the per-piece Allen matrix (M16c), numpy interop (M16d) and gmpy2/mpfr as a faster backend for
  `elementary.py` and the outward hook, not a tighter one (M16e). what stays here from it:
    * an interval *array* type (the array API standard's namespace), if ever; not the numpy
      interop, which is built (M16d; D23 (a), `HANDOFF.md` Q15(a))
    * the backend's non-dyadic part: an mpfr ziv loop for a monotone f at a bracketed x, for the
      points the backend declines today (a `Fraction(1, 3)`, `log` to a base, `pow_rev2`'s `log_t
      v`, `acoth`, `rootn` with n < 0, the periodic reverse ops' `k pi + f(v)`); not measured
      (M16e; `HANDOFF.md` Q16(c))
    * allen's composition table (from the relations of `(a, b)` and `(b, c)`, those possible for
      `(a, c)`), re-derived for the cut reading with points, where some classical compositions
      shrink (a point cannot OVERLAP); `allen_relations` is its input. a design of its own (M16c)
    * vector-mode autodiff (a tangent tuple inside `Dual`, one pass for a jacobian, M15's chain
      rules edited), the alternative if a measured solve is too slow (M16a; `HANDOFF.md` Q12(a))
    * 1788's recommended operations the layer does not have, not built by design (no vectors):
      `exp2m1`, `exp10m1`, `log2p1`, `log10p1`, `compoundm1`, `rsqrt`, the `*Pi` functions, the
      exact text and interchange conversions, and every inf-sup type but binary64 (M16b)

## decision log

### 2026-10-03 revision: owner answers to Q9-Q20 and the owner's-call items

the owner accepted every recommendation of `references/owner-questions-2026-10-03/` on 2026-10-03
("i'll accept everything fable said"): nine reports, one per group of questions, each option's pros,
cons and when it is the better choice (`README.md` there is the table). where a report said
"consider", "optional" or "later if asked", nothing was decided beyond that. in short:
* **kept as built, now owner-confirmed**: D19 (Q11, M15) and D20 (Q12, M16a) whole; D21 (Q13, the
  1788 layer) but (b); D22 (Q14, allen) whole; D23 (Q15, numpy) (a), (b), (d), (e); D24 (Q16, the
  backend) (a)-(c), (f). Q9 and Q10 close as built in the 1788 layer (`ieee1788.mul_rev_to_pair`; the
  constructors' binary64 pass); `layer-numpy`: the layer keeps `__array_ufunc__ = None`
* **Q20 (D29)**: a crossed piece in the exact class stays the piece between the two values, each end
  keeping its flag (`[1e-06, 1/1000000)`), as plain `*` already gives a float end beside an exact one
* **Q19 (D29)**: the outward class stays typed per end, so it is isotone within one grid and, across
  grids, `f(A)` lies within the tightest double cover of `f(B)`; documented, and a public method puts
  every end on the double grid, outward, for a user who needs isotonicity. rejected: doubles only in
  the outward class (drops exact points such as the solver's `1/6`), and rounding whenever a float is
  mixed in (not isotone either: a purely exact `B` stays exact)
* **Q17 and Q18 (D28)**: pown of exact operands past one exact-result limit of about 2 ** 22 bits,
  shared by pown, pow and exp2/exp10, is the tightest open float enclosure (outward) or the value
  rounded to nearest, with a default-ignored warning; `EXACT_POWER_LIMIT` stays the float-corner
  threshold. pown to nearest is correctly rounded for every n (libm's `float ** int` was off by an ulp
  on this laptop; it was the one libm value in the library). `repr` must not raise past python's
  4300-digit limit: hex past it, which `parse` reads
* **Q13(b)**: the 1788 layer's numbers of the empty set are `nan`, 1788's answer; the library keeps
  D9's `ValueError`
* **Q15(c), (f), (g), (h)**: `==`/`!=` against an ndarray is elementwise; `fmin`/`fmax` are
  `minimum`/`maximum`; numpy joins `[test]` and the README numpy section becomes doctests; a method
  mixing the two classes returns the class the operators do (a defect: `M(0.1).hypot(O(0.1))` was a
  `MultiInterval`, against README "rounding")
* **Q16(d), (e)**: `[fast]` pinned to the window `auto` takes, as `[test]`; one CI gate job on the
  forced gmpy2 backend, no gmpy2 fuzz
* **the four 1788 departures never asked (D27)**: confirmed deliberate; the stale "domain-clipped
  functions" category goes
* **Q6-shift**: `A << n` is `A * 2 ** n` and `A >> n` is `A * 2 ** -n`, exact, through `*`; negative n
  allowed; `//` stays the floor. **Q6-rest closed**: neither `random_multi_interval` nor a public
  `apply()` in 2.0. **vectors-ext (b) closed**: glibc's rows are conformance inputs, not hard cases;
  pown takes CORE-MATH's pow rows with integral exponents instead
* small, from the same reports: `allen_relations`' set is documented as extensional and every
  cut-tuple relation asserts normalized operands; the solver documents `tol` as absolute and
  `max_steps` as boxes, splits symmetrically inside (0, 1], and `newton`'s width test cannot overflow
  (open item `newton-width`)
* left open, as the reports said: `Root`/`RootBox` as dataclasses if a third state appears; an `rtol`;
  the outward fma, `%`, hypot and `cancel_minus` per corner (tighter, optional); rootn run on cbrt's
  worst-case inputs (optional); a public strategies module after 2.0 if asked

### 2026-09-30 revision: to nearest, a reverse op meets `x` before rounding (D26)

* the fuzz job found `pown_rev(c, -1, (-inf, -2))` empty to nearest for `c = (-2.2e-309, 0)`, whose
  exact answer `(-inf, -4.49e308)` is wholly past -MAX: the preimage rounds to the point `[-inf]` and
  `x`, open at -inf, dropped it (plan §2 "fuzz-rev-inf"). 1788 intersects with `x` first and then
  encloses (`[-inf, -MAX]` there, no infinite point); python's float rounds such a value to -inf
* chosen (owner): 1788's order in the reverse ops, python's rounding kept: `x` meets the preimage
  before the rounding, so a part of it inside `x` that rounds wholly onto one double is that double,
  as a point: `[-inf]` here, an end `x` excludes (the one case where a result leaves `x`); or a point of
  `x` at an end the rounding kept open (`mul_rev(10, (1, 2), [0.1])` is `[0.1]`, was `{}`). the nearest class stays IEEE round-to-nearest everywhere, so forward ops
  are unchanged (`(M(1e308) * 10) & M.parse('(0, inf)')` is `{}`: `[inf]` is the point inf).
  rejected: saturating an overflow to `(MAX, inf)` in the nearest class (1788's enclosure rule; it
  would stop matching python's float), and keeping the loss as a documented rule
### 2026-09-29 revision: fuzz on push, not on a schedule

* the owner: the fuzz runs "fully autonomously or not at all"; nobody reads a scheduled run's
  failure email. so `fuzz.yml` runs on every push to `master` (and `workflow_dispatch`), not weekly;
  the same fuzz run happens locally before each push (`tools/prepush.sh`), and the session that
  pushed watches both workflows to the end (`tools/ci_watch.sh`, a babysitter agent), debugging a red
  run from its log and artifact. the push procedure is in `CLAUDE.md`
* the fuzz database was never kept on CI: hypothesis loads its `ci` profile at import under
  GitHub Actions, whose database is None, and the `fuzz` profile, registered without one, inherited
  it, so the first two GitHub runs saved no example and the carried `.hypothesis` held only
  hypothesis's constants cache (their artifacts have no `examples/`). found 2026-09-29 by replaying
  the second run's artifact locally, which replayed nothing. the profile now names its database;
  `tests/test_fuzz_profile.py` pins it (red without it, under a simulated CI)

### 2026-09-29 revision: `-` and `+` keep each cut's type (fuzz-symmetry)

* a point can hold one value in two types: `[0, 1/2] & [0.5, 1]` is the point with a float 0.5 below
  and an exact 1/2 above, and the fuzz drew the other way round. the applicator reads a point by its
  low cut alone (`applicator._ends`), so `-` of the fuzz's point gave `[-1/2]` exact, while the reverse ops read each end by its own
  type (`reverse._end`): `pown_rev(-c, -7)` was not `-pown_rev(c, -7)`, found by M14's first GitHub
  fuzz run (plan §2 "fuzz-symmetry")
* `-` is now the cut mirror and `+` the identity, each cut keeping its type, as `-` already did for a
  piece with two ends: exact, an involution, and the `-` `test_trig_rev_symmetry` had to build for
  itself (`reverse.negate`). the other forward ops still read a point at its low cut; the reverse ops
  still read each end by its own type. a mixed point is sound either way; only its rounding differs

### 2026-09-29 revision: pown-huge, built

* pown (`A ** n` for an integral n, and every path to it: `ieee1788.pown`, `Interval ** n`,
  `DecoratedInterval`, `Dual`, `np.power`) of a float corner with a huge n never finished: the outward
  descriptor built the exact `Fraction(x) ** n`. it now builds it only within `EXACT_POWER_LIMIT` bits
  and otherwise rounds with `elementary.rounded_pow`, the route pow already took, so pown and pow give
  the same doubles; the design, its proof and the tests are plan §2 "pown-huge". "rounding"'s "fn is
  exact" holds for pown up to that limit; past it attainment is against a marker equal to nothing,
  which the proof shows is the exact value's answer
* the nearest class past `|n| = 2 ** 53` is `rounded_pow` to nearest: python's `float ** int` lost the
  parity (a wrong sign) or read an int past the double range as an overflow (`M(0.5) ** 10 ** 400` was
  `[inf]`). up to 2**53 it is python's `float ** int` as before (whether that promises correct
  rounding is open, Q-nearest-libm)
* the descriptor's name is bounded (`pow<a 20001-bit int>`): `f'pow{n}'` raised python's 4300-digit
  ValueError in every class
* exact int/Fraction operands are unchanged and still huge by nature (Q-exact, for the owner)

### 2026-09-28 revision: python 3.12 minimum (owner, D25)

* CI's first run of M15 and M16 (36402681261) was red on python 3.11 alone: CPython 3.11's `Fraction.__pow__` answers a non-rational exponent with `float(a) ** b`, so `Fraction(1, 3) ** A` reached `A.__rpow__` as `0.3333333333333333 ** A`, unsound in `OutwardMultiInterval` (1/9 missed) and indistinguishable, inside the library, from a float base. the owner chose python >= 3.12 over documenting `Fraction ** interval` as unsupported on 3.11 (`pyproject.toml`, `ci.yml`); pinned by `tests/test_outward.py::test_a_fraction_base_stays_exact`

### 2026-09-28 revision: M16e, the gmpy2/mpfr backend (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now" (`HANDOFF.md` H3; plan §2 M15: "gmpy2/mpfr stay out"). built as M16e,
one of M16's five streams. the choices, each the build's default, open for the owner (D24, Q16):
* **the pure path is the default**; gmpy2 only with `INTERVALS_BACKEND=gmpy2` (forced) or `auto`.
  the design had automatic-when-importable; its critique held that the conservative reading wins:
  the pure path is the reference, the local gate and itf1788 then keep checking it, and a user's
  gmpy2 (linked to whatever MPFR their distribution ships) never changes code paths unasked
* **the backend only picks the double** (the contract above), so it is correct by construction
  wherever it declines, and bit-identical where it answers (the differential)
* **declined where one MPFR call is not one rounding**: non-dyadic inputs (but atan2 of two ints and
  the hook's exact mpq), `log` to a base, `acoth`, `rootn` with n < 0, `k pi + ...`; and short of
  where MPFR's limits bite: `rootn` from `n = 2**31` (a margin: gmpy2 raises from `2**32` on
  windows), an operand past `2**20` bits
* **`gmpy2>=2.3,<3` in `[test]`** (the differential never skips, and CI installs the series `auto`
  takes) and a new extra `[fast]` (`gmpy2>=2.3`, unpinned); no CI change:
  every gate job installs `.[test]`, so `tests/test_backend.py` runs everywhere, and the rest of the
  suite runs on the pure path there as here
* **`auto`'s window is `2.3 <= version < 3`** (the series verified); forced takes any gmpy2 at the floor

### 2026-09-28 revision: M16d, numpy interop (H3's second part), built

the owner asked (2026-09-27) for the rest of H3; numpy was one of its five streams. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q15; D23 in
v2-implementation-plan.md):
* **`__array_ufunc__`, not the array API**: the owner's 2025-12 line names "numpy compat via
  data-apis.org/array-api or `__array_ufunc__` or the interoperability page"; a multi-interval is an
  element, not an array, so the hook; the array API would be a new interval-array type (Q15(a))
* **a foreign real is exact** (Q15(b)): a `numbers.Rational` by type, any other real by value where
  `float()` would round it; fixes the outward class for `np.longdouble` on linux and gmpy2's `mpq`
  and `mpfr` operands; a double stays the float it is
* **an ndarray meeting ours is elementwise into an object array**, `==`/`!=` never broadcast
  (Q15(c))
* **no numpy-named alias methods** (`arcsin`, `rint`, ...) on the classes (Q15(d)), so numpy's
  object loops and `np.round` refuse them
* **`np.invert(A)` is the complement `~A`** (Q15(e)); `fmin`/`fmax` not mapped (Q15(f))
* **numpy not in the `[test]` extra** (Q15(g)); the README's numpy section is prose
* **both operands ours in a method ufunc: the subclass decides** (Q15(h)), as for the operators, so
  `np.hypot(M, O)` is outward (the review found the first operand's class decided)
* found on the way and fixed (not choices): M15's `Dual ** r` computed `r - 1` in r's own
  arithmetic, unsound in the outward class with plain python floats (`Dual.variable(O(1e300)) ** 0.1`
  missed its derivative, bare and decorated; `Dual.variable(O(-1)) ** 2.0 ** 60` had its sign
  flipped); a number exponent is now an int (integral) or a point set of u's kind first, which
  also changes the nearest class (an exact derivative for an integral float exponent; pow, not
  pown, for an r like 1e-20 whose float `r - 1` is -1.0)

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

### 2026-09-28 revision: M16b, the 1788 layer (H3's second part), built

the owner asked (2026-09-27) "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now"; H3's rest was built as M16 in five streams, this one M16b. the
2026-08-16 decision ("if real conformance is ever needed, it's a thin wrapper class in
`ieee1788.py`, never a mode on MultiInterval") is now built, as designed there. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q13; D21 in
`v2-implementation-plan.md`):
* **shape**: `intervals/ieee1788.py`, one class `Interval` for both flavours, snake_case 1788 names
  with 1788's camelCase in `NAMES`, not exported from `intervals`
* **numbers of the empty set raise** (`ValueError`, D9's answer, as the owner chose for the
  reductions, Q2) where 1788 says NaN; `inf`/`sup` of it are `±inf` either way
* **Q9 and Q10 answered by the layer, pending Q13 (c), (d)**: 1788's pair with its decoration is
  `ieee1788.mul_rev_to_pair` and the library's `mul_rev` stays one op, trv; the constructors' binary64
  run is the layer's pass, no class argument on the library's constructors. the adapter's 52
  `DECORATION_ONLY` rows and its reasons are left as they are (true of the library).
  why the library's `mul_rev` stays trv rather than decorated as the division: in the library ±inf
  are points, so where 0 is not in `b` the two sets differ (`mul_rev([1, inf], [inf])` is `(0,
  inf]`, `[inf] / [1, inf]` is `[inf]`; measured 2026-09-27, re-checked 2026-09-28), and a
  decoration copied from the division would claim com or dac for a set that is not the division's; a
  multi-piece `b` gives more than two pieces, so a pair has no meaning on the core; and 1788
  decorates its own `mulRev`, the hull of the same set, trv (`libieeep1788_rev.itl:988`), so one
  library op cannot match both
* **where 1788 and the library disagree, the layer answers 1788's way** (cancellation's "no answer",
  `meets`, attained infinities dropped), the library its own
* **the sign of zero follows 1788 in the layer**: `inf` gives `-0.0` for a lower end of 0 and `sup`
  `+0.0` for an upper end of 0 (1788-2015's rule; the critique's n7: "recorded, not an owner
  question" did not fit the layer's principle of answering 1788's way). no vector can see it
  (`tests/itf1788/itl.py::parse_number` reads `-0.0` as `Fraction(0)`); pinned by
  `tests/test_ieee1788_layer.py::test_the_sign_of_zero_is_1788s`. the package's one zero (2026-09-22
  revision) is unchanged: the ends of an `Interval` are never `-0.0`, and nothing re-enters the
  library from `inf`/`sup`

### 2026-09-28 revision: M16a, the solver in several variables (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done". M16a is its stream nd-solver; the choices are the
build's defaults, open for the owner (Q12; D20):
* **n passes, `Dual` untouched**: a gradient or a jacobian is n evaluations of `F`, each with one
  variable seeded. vector mode (a tangent tuple inside `Dual`, one pass) edits M15's pinned chain
  rules; it stays the alternative if a measured solve is too slow (Q12(a))
* **`solve` and `RootBox`** (names, Q12(b)); `Root` stays one set for `newton`. n == 1 delegates to
  `newton`, so one variable keeps M15's behaviour exactly
* **krawczyk proves, gauss-seidel with `mul_rev` narrows**, on the closed hull, with a float
  preconditioner (identity fallback); uniqueness for n >= 2 by krawczyk only (hansen and sengupta's
  own test is not used: its proof was not checked)
* **the simplest point and the inflation, before an unproved box is output**, each measured to be
  needed: without the first, the cusp's `(0, 0)` and `(1, 1)` end unproved; without the second,
  `(x ** 2 - 2, y - 1/4)` proves none of its two zeros (critique B1). the inflation is clipped to the
  box's region, carried on the stack, without which it can claim a box holding no zero
* **the C¹ gate in n variables**: every value and every partial of the n decorated passes dac or
  better, over the closed hull
* **bisection**: wide components first, round robin; then the widest (kearfott's smear measured on
  the prototype, 148 against 146 calls, 345 against 340, not taken)
* **no direction tag** (the solver stack, above)

### 2026-09-27 revision: M15, autodiff and interval newton (H3's first part), built

the owner asked for H3 first; its suggested first pick (2026-09-26 revision below) was built. the
choices the build made, each the session's default, open for the owner (`HANDOFF.md` Q11; D19 in
v2-implementation-plan.md):
* **public, in the package**: `intervals/autodiff.py` (`Dual`, `derivative`) and
  `intervals/solver.py` (`newton`, `Root`), exported from `intervals`, as the 2025-12 layout sketch
  named them (`autodiff.py`, `solver.py`), rather than newton "as a test" only
* **C¹ is proved by decorations** (dac or better on the value and on the derivative), the decorated
  type's use in the solver that the "later" list kept back; where it fails the piece is only pruned
  and bisected. the alternative, trusting the caller that `f` is differentiable, is the silent
  failure `tests/test_solver.py::test_not_c1_would_lose_a_zero` shows
* **the step is `mul_rev`**, the reverse op, never `/`: `[0] / [0]` is `∅` (D7)
* **one variable**: a `Dual` carries one derivative; several variables (a gradient, a jacobian and
  a krawczyk or newton step in n dimensions) are not built. **superseded 2026-09-28**: built as M16a (the 2026-09-28
  revision above), `Dual` untouched
* `tol=1e-10` absolute, `max_steps=10_000`; float midpoints; the magnitude split at a factor of 16

### 2026-09-27 revision: owner answers on M13's proposed categories (D18)

* "tighter than the vector" (M13e: 11 keys, 18 vectors, the two grossly loose `pow_rev.itl:609`,
  `:642` included) and "exact parsing decides validity" (M13g: 7 keys) are approved residual
  categories; the `(PROPOSED)` markers are gone from `REASONS` and from the rows' reasons
* the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY`
  12) stay under "decoration expectations"; `set_dec` keeps demoting as 1788's `setDec` does
* the adapter's `tests/itf1788/test_itf1788.py::is_decorated` unpacked an `Interval` result (a
  NamedTuple) into its fields, so a decoration on the result alone went unseen; 0 vectors were
  affected in what runs (the 103 it missed are all of the six decorated ops, which take the decorated
  path by op), but the census undercounted: 1624 vectors carry a decoration, not 1521 (121 of the
  decorated ops, not 18; measured 2026-09-27). fixed and pinned by
  `::test_is_decorated_sees_the_result_alone` (the old line: 1 red)

### 2026-09-27 revision: M13 merged, M13's exit met

M13e (the reverse ops) and M13g (decorations, constructors and signals), built in parallel on
branches `m13e` and `m13g`, are merged on `m13-merge` (details in v2-implementation-plan.md, "exit
for M13"). M13 is done: every statement of the 19 files is a vector of an op in `OPS`, and
`tests/itf1788/test_itf1788.py::test_nothing_is_skipped` asserts `SKIPPED` empty. what the merge
itself decided, each the conservative reading:
* **the reverse ops take decorated operands** ("ieee 1788" above, decorated reverse ops): the
  core's set, decorated trv, as 1788 decorates a reverse op's result
  (`intervals/reverse.py::_decorated`); a bare `MultiInterval` beside a `DecoratedInterval` is a
  `TypeError`, as for the wrapper's other ops
* **mulRevToPair's better decoration is not built**: 1788 decorates the pair's first interval as
  the decorated division `c / b` where `0 ∉ b`, but its mulRev, the same set's hull, trv. ours is
  one op, `mul_rev`, which cannot be both; it is trv (sound: trv claims nothing). 52 vectors are
  rows under "decoration expectations", on the decoration alone (the set must match). whether to
  add a pair op with 1788's decoration is an owner question
* the counts at M13e and at M13g are replaced by one re-measurement ("counts at M13's merge");
  the two categories proposed at M13e and M13g were approved the same day (next revision up)

### 2026-09-26 revision: owner answers to the open questions

the owner answered `HANDOFF.md`'s questions and items on 2026-09-26:
* **Q3 and Q7 confirmed**: the readings of M13c (equal infinite ends as 1788 writes them;
  `.interior` in the reals) and M13d (`0 ** y` for y <= 0 outside the domain, `1 ** ±inf` and
  `inf ** 0` indeterminate, an integral Fraction exponent is pown, acot continuous) stand as
  recorded in their entries below
* **Q4 closed, no inward variant.** 1788 has none either: `cancelMinus` at level 2 is the hull of
  the exact level-1 answer, an outer enclosure, which `OutwardMultiInterval` already gives.
  reopen if a solver needs a certified inner answer; exact operands give one today
* **Q5 on hold** (the time layer, M8): no rush
* **Q6: `<<` and `>>` will be ported**; `random_multi_interval` and a public `apply()` are kept as
  to-dos, undecided. v1 applied python's int shifts endpoint-wise (`archive/v1/multi_interval.py::__lshift__`),
  so floats raised; the meaning on real sets (`A << n` as `A * 2**n`, and `>>` as exact division
  or as python's floor) is chosen when built
* **H1**: release 2.0.0 when everything is fully done, which narrows D17's "whenever"
* **H4**: delete `archive/v1/` after v2 is stable
* **H5**: the v1 README's reading list and illustration to-do are kept, moved to
  `references/todo-from-v1-readme.md` so they outlive `archive/v1/`
* **H2**: push `v2` approved
* **Q1, after 1788's context** (in 1788 every signal is a flag, like IEEE 754's: the result is
  returned, empty for bare and NaI for decorated, and the program goes on, since the result itself
  carries the news: empty and NaI propagate). the owner's choice for python, where errors raise:
    * **`UndefinedOperation` raises**, a `ValueError` subclass: a 1788 constructor given invalid
      input (`numsToInterval(2, 1)`, `"[ foo ]"`) stops, as `MultiInterval(2, 1)` already does.
      so NaI, which 1788 makes only from such input, arises only when made on purpose
    * **`PossiblyUndefinedOperation` warns** (an `IntervalWarning` subclass) and returns 1788's
      result. an exact parser can always decide validity, so it may never fire; the one vector
      that expects it (`ieee1788-exceptions.itl:18`) is settled when M13g is built
    * **`IntvlPartOfNaI` raises** (owner, 2026-09-26, after first leaning to a warning): NaI
      propagates like NaN through decorated ops, but `intervalPart` is the one place it stops,
      turning "invalid" into `∅`, which then reads as a real "no values"
    * **no NaI at all** (owner, 2026-09-26, Q8): with both raising, nothing in the library makes
      NaI. its only other uses, a per-element "invalid" in batch work and reading another 1788
      library's `[nai]`, come with numpy or data import, if ever, and it can be added back then.
      the decorated type has com/dac/def/trv, no `ill`, so `IntvlPartOfNaI` cannot arise; the
      statements needing a NaI become rows under a new divergence category, "no NaI: invalid
      input raises", approved with the decision (details: v2-implementation-plan.md M13g)
* **Q2 settled: the reductions keep raising `ValueError`** where 1788 answers the float `nan`
  (they are operations on floats, not intervals, so an empty interval is no answer either);
  `math.fsum` raises on `inf + -inf` too
* **H3**: numpy interop and a gmpy2/mpfr backend are recorded, not built now ("later (not in
  v2.0)" above). the session's suggested first pick when the solver stack starts: Newton's
  method with forward-mode autodiff, the demonstration of what multi-intervals are for. **superseded
  2026-09-27** by the owner's "get the rest of h3 done": numpy interop and the gmpy2/mpfr backend
  were built as M16d and M16e (the 2026-09-28 revisions above)

### 2026-09-26 revision: M13e, the reverse ops, built

M13e is complete (2026-09-26; details in v2-implementation-plan.md, M13e, parts 1 to 4 and the
close-out). D12 is now in "current design" whole: elementary and step functions (the reverse ops,
their exact pieces and D12's cap for the periodic ones), empties and warnings (`HullWarning` for
the periodic ones past the cap or over an unbounded piece of `x`) and ieee 1788 (the reverse ops,
reverse multiplication, the periodic and the power reverse ops, and all of them together). the
choices the plan left open are the four parts' entries below; the close-out made none. in short:
* **ten functions in `intervals/reverse.py`**, exported from `intervals`, each the exact set
  `{t ∈ x : f(t) ∈ c}` (for `mul_rev`, `pow_rev1`, `pow_rev2`, `∃` over the other operand) with the
  library's own f, ±inf points like any other, where 1788 answers its hull; an irrational end is its
  tightest float enclosure, open, and the domain is intersected after rounding
* **the itf1788 vectors**: all 1955 of the 19 ops run in both passes; 1879 match, 52 are
  degenerate infinities (the unary `pownRev` with n < 0 and 0 in `c`), 6 `[nai]` rows move with
  M13g, and 18 (11 keys) are under **"tighter than the vector", still PROPOSED and awaiting the
  owner**: 1788's hull is looser than the tightest, which ours is
* **a pin until M13's exit**: `tests/itf1788/test_itf1788.py::test_only_m13g_ops_are_skipped`, since
  dropping a reverse op from `OPS` turned no test red before it (its statements only become skips)
* itf1788 now 9269 vectors of 102 ops; 273 statements of 9 ops skipped, all M13g's (2026-09-26,
  `tools/itf1788_census.py`). the totals in "ieee 1788" above are M13d's until the merge with
  M13g's branch re-measures them

### 2026-09-26 revision: M13e, fourth part (pow_rev1, pow_rev2), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 4). `pow_rev1` and
`pow_rev2` moved into "current design" (elementary and step functions: reverse ops; ieee 1788: power
reverse ops). the choices the plan left open, each the most conservative reading, are now current
design too:
* **the library's own pow defines them**, ±inf as points: `pow_rev1([-2], [0, 1])` is `[1, inf]`
  (`inf ** -2` = 0), `pow_rev1([inf], [0])` is `[0, 1)` and `pow_rev2([0], [0])` is `(0, inf]` (`0 **
  inf` = 0); `0 ** y` for y <= 0, `1 ** ±inf` and `inf ** 0` have no value, so they solve nothing.
  1788 has no infinite points and every 1788 vector gives the domain through the input rule, so no
  vector sees the difference
* **the operand order is 1788's** (`b` or `a` first, then `c`, then the domain), and the domain is
  named `x` for the bases and `y` for the exponents, defaulting to `[-inf, inf]`
* **the float rule is pow's, per operand**: a finite float end in either operand makes every end
  float (to nearest, flags kept; outward, a moved end open), as `**` does; part 1's ops take each
  end of `c` on its own (the functions' rule). so `pow_rev1([n], c)` equals `pown_rev(c, n)` on `[0,
  inf]` when `c`'s ends are all exact or all float, not when they mix
* to nearest, an end past the largest double is inf, closed, the library's nearest rule (`M(1e200)
  ** M(1000.0)` is `[inf]` too); `OutwardMultiInterval` keeps it open
* the class, the coercion of a number, the warnings (an empty operand only; a negative base or a
  point with no value warns nothing) and `∩` the domain after rounding are part 1's
* **a bug found while building**: `MultiInterval(2).log(4)` hung, since `elementary._exact_log`
  looked only for an int k with `4 ** k == 2` and ziv's loop never settles on the rational 1/2.
  `log_b x` is now exact wherever rational, through each operand's perfect-power decomposition
  (`elementary._perfect_power`; since M13e's review, 2026-09-27, euclid on the exponents,
  `elementary._log_ratio`, as the root search took minutes on `3 ** 20000 + 1`); `pow_rev2` needs it,
  as every `powRev2` vector's ends are rational
* **a proposed category gains rows**: two `powRev2` vectors expect a hull far looser than the
  tightest (`[entire]` and `[-infinity, 0.0]` where it is `[-inf, -0.5]`); they are rows under
  "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 804 vectors: 802 match in both passes, 2 are those rows. itf1788 now 9269 vectors of 102 ops;
  273 statements of 9 ops skipped, all M13g's (2026-09-26, `tools/itf1788_census.py`)

### 2026-09-26 revision: M13e, third part (sin_rev, cos_rev, tan_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 3). D12 for the
periodic ops moved into "current design" (elementary and step functions: reverse ops; ieee 1788:
periodic reverse ops). the choices the plan left open, each the most conservative reading, are now
current design too:
* **a `c` holding `[-1, 1]` is not hulled** for sin and cos: every finite t is a solution, one
  piece, so `sin_rev([-1, 1])` is `(-inf, inf)` with no warning; D12 hulls only answers with too
  many pieces. tan has no such `c` (its poles), so `tan_rev((-inf, inf))` warns
* **the poles and ±inf are in no preimage**, part 1's rule for a point with no value, although the
  set op `tan` attains ±inf around a pole inside a piece: `tan_rev([inf])` is `∅`. a piece around a
  pole keeps it, since it is irrational and the enclosures of the two branches' ends overlap
* **the cap per piece of `x`**, as `steps.step` has it: earlier pieces stay exact, the piece that
  would pass 1000 pieces and every unbounded piece are hulled, one `HullWarning` per call; the
  hull stays inside its piece of `x`
* **the result is the union of every branch's enclosure, then `∩ x`**, so one branch more is listed
  on each side of a piece of `x`, for an `x` that starts inside a neighbour's rounding slack
* a `c` whose gaps are single irrational points (`[-1, 1)` for sin) counts as having gaps: a wide
  `x` is hulled with the warning even though listing would give the same set
* far out, where the doubles are coarser than a period (above about 1.8e16), neighbouring
  enclosures merge into one piece; still an enclosure
* **a proposed category gains rows**: six more `*Bin` vectors (12 with decorated copies) expect an
  end one or two doubles outside the tightest enclosure; ours is the tightest (arb). they are rows
  under "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 136 vectors: 124 match in both passes, 12 are those rows. itf1788 now 8465 vectors of 100
  ops; 1077 statements of 11 ops skipped (`powRev1`, `powRev2`, and M13g's)

### 2026-09-26 revision: M13e, second part (mul_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 2). `mul_rev` moved
into "current design" (elementary and step functions: reverse ops; ieee 1788: the pair rule). the
choices the plan left open, each the most conservative reading, are now current design too:
* **the library's own `*` defines it**: `t` is in `mul_rev(b, c, x)` iff `{t} * b` meets `c` and
  `t ∈ x`, so `0 * ±inf` (no value) contributes nothing: `mul_rev([inf], [0])` is `∅` and
  `mul_rev([1, inf], [3], [0, 10])` is `(0, 3]`, 0 excluded. 1788 has no infinite points, so no
  1788 vector sees the difference
* **rounding is the division's**: every finite end other than 0 is a quotient of an end of `c` by one
  of `b`, computed by `ops.div`, so it is exact for int and Fraction ends and rounded (to nearest,
  flags kept, or outward, a moved end open) exactly where the division would round it. the other
  candidate, `fma`'s rule (all ends rounded once if any operand has a finite float end), would
  make `mul_rev([3], (-2, 0.0))` differ from `(-2, 0.0) / 3` in its exact end `-2/3`
* **x defaults to `[-inf, inf]`**, as for the other reverse ops; no 1788 vector is touched, since
  the input rule never gives `c` an infinite point and ±inf join a preimage only through one
* **`mulRevToPair` is `mul_rev` under a pair rule** in the adapter, piece by piece (stricter than
  as a union, which the plan allowed); the one exported function serves all three 1788 ops
* the 539 vectors: all match in both passes but the 6 `[nai]` ones (generated rows under
  "decoration expectations", which M13g moves). no new category

### 2026-09-26 revision: M13e, first part (sqr_rev, abs_rev, pown_rev, cosh_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, in progress). D12's
reading for these four moved into "current design" (elementary and step functions: reverse ops;
ieee 1788: the reverse ops rule). the choices the plan left open, each the most conservative
reading, are now current design too:
* **`x` defaults to `[-inf, inf]` with f's own values at ±inf** (the plan's `x=REALS`), so the
  infinities are points of a preimage like any other: `pown_rev([0], -2)` is `[-inf] ∪ [inf]`,
  1788's `[empty]`. the 1788 vectors this touches are rows under "degenerate infinities"
* **a point where f has no value is in no preimage**: 0 for n < 0, where the library's `1/[0]` is
  empty, even though the set op `1/[-1, 1]` attains ±inf around it. so the result is pointwise,
  and `T ⊆ rev(f(T))` holds for every `T` without such points
* **an empty operand warns** (`EmptySetPropagationWarning`, off by default), as the functions do;
  an empty answer from non-empty operands does not warn: "no solution here" is a normal answer
* **the result class is `OutwardMultiInterval` if either operand is one**, as for the dunders;
  the functions are module-level (`intervals.sqr_rev`), not methods, as the plan's signatures say
* **the adapter compares the closed hulls** (the output rule unchanged): 1788's reverse op is the
  hull of the preimage by definition
* **a proposed new residual category, "tighter than the vector" (not yet approved)**: two
  `pownRev` vectors (`rev.itl:276`, `:277`) expect an end one double outside the tightest enclosure
  of `2 ** (1074/7)`; ours is the tightest (checked against arb). they are rows under it, and the
  owner decides whether the category stands
* the 476 vectors: 420 match in both passes, 52 (26 keys) are degenerate infinities of the unary
  `pownRev`, 4 (2 keys) the proposed category. itf1788 now 7790 vectors of 91 ops; 1752
  statements of 20 ops skipped (the rest of M13e, and M13g)

### 2026-09-26 revision: M13g, decorations, constructors and signals, done

M13g is done on branch `m13g` (2026-09-26); its record is v2-implementation-plan.md M13g, "done",
and parts 1 to 3 below are its steps. **D16 is now current design** ("ieee 1788" above): the core
`MultiInterval` stays undecorated, and 1788's decorations live in the wrapper `DecoratedInterval`,
with `Decoration` com/dac/def/trv, 1788's constructors (`text_to_interval`, `nums_to_interval` and
their decorated twins, `set_dec`) and propagation through every point function and set operation of
the core. 1788's signals are python's: `UndefinedOperationError` (a `ValueError`) raises,
`PossiblyUndefinedOperationWarning` (an `IntervalWarning`) would warn and is never emitted, since
the exact parser decides validity. no NaI and no `ill` (Q8): every statement that needs one is a
row under "no NaI: invalid input raises". every itf1788 statement of an M13g op is a vector (matching,
or a row), every decorated vector runs through the decorated type with its decoration checked, and
the only statements still skipped are the 19 reverse ops' (M13e), pinned by
`tests/itf1788/test_itf1788.py::test_only_the_reverse_ops_are_skipped`. **still open for the owner**
(each built as the conservative reading, flagged in the part entries): the PROPOSED category "exact
parsing decides validity" (7 rows); whether the 15 rows where an exact value past the doubles keeps
com (`_BOUNDED_EXACTLY`, 3; `PLAIN_ONLY`, 12) stay under "decoration expectations" or get a category
of their own; and `set_dec` demoting as 1788's `setDec` does rather than raising. the reverse ops
join the decorated type with M13e, all trv (`decorated.py::_trivial`)

### 2026-09-26 revision: M13g part 3, decoration propagation, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part
3") and "ieee 1788" above. `DecoratedInterval` now propagates 1788's decorations through every point
function of the core and decorates set operations trv; the adapter checks the decoration of every
decorated vector of those ops. the choices the plan left open, each the conservative reading, flagged:
* **a multi-piece box is decided on the set, not on the hull**: continuity on the set is continuity
  on each piece, the pieces being apart (the owner's task text asked for this reading)
* **an attained ±inf is outside every domain**: 1788's functions are functions of reals, so an
  operand holding ±inf as a point gets trv even where the core takes a limit (`exp([0, inf])` is trv,
  `exp([0, inf))` dac)
* **com needs the result bounded as returned**: the exact result for exact operands (so
  `[1,2]_com + [5,max]_com` stays com here, 12 plain-pass rows under "decoration expectations"), the
  rounded one for float operands (`OutwardMultiInterval` gives `[6.0, inf)`, dac, as 1788;
  `MultiInterval` rounds `2.0 + max` to nearest, max, and keeps com). the owner question of part 2
  (`_BOUNDED_EXACTLY`: this category, or an exact-values one) covers these rows too
* continuity at a point is relative to the op's domain (`sqrt` at 0, `acosh` at 1, `pow` at `x = 0`
  with `y > 0` are com), as the vectors require
* `%`, `//`, `divmod` and `round(ndigits)`, not in 1788, get the same rule from their definitions;
  `sign` and every step function at a jump point is dac at best
* booleans, numbers and relations stay off the wrapper (`.interval` first); a bare `MultiInterval`
  operand is a `TypeError`, a real number newDec's point
* the core's warnings reach the caller once; the decoration's own core calls are silenced

### 2026-09-26 revision: M13g part 2, the decorated type, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 2").
the decorated wrapper of D16 is `DecoratedInterval` (`intervals/decorated.py`, "ieee 1788" above),
named after M8's `DateTimeInterval`, with a `Decoration` enum and no `ill`. the choices the plan left
open, each the conservative reading, flagged:
* **the constructor is strict, `set_dec` is 1788's**: `DecoratedInterval(x, d)` raises
  `UndefinedOperationError` for a decoration that does not fit, as the literal `"[1,]_com"` does;
  `set_dec` demotes as 1788 defines it, with no signal (`libieeep1788_class.itl:283`-`288` expect
  `[empty]_trv` and `_dac`), and raises only for `ill`. raising for every unfitting `setDec` would
  have turned six matching vectors into rows needing a new category, for input 1788 calls valid
* **bounded is decided on the exact set**, so `[1.0E+400]_com` stays com: three vectors are rows
  under the existing "decoration expectations" (owner question: or the PROPOSED exact-parsing
  category, widened)
* a decoration name is exact lower case (`'COM'` raises): a python argument is not 1788 text
* **a core bug fixed**: `MultiInterval.is_finite` and `.finite` raised `OverflowError` on an exact end
  past the doubles (`MultiInterval(10**400)`), since `math.isfinite` converts to float; they compare
  with ±inf now (`tests/test_multi_interval.py::test_finiteness_of_an_exact_end_past_the_doubles`)
* not built: decorations propagated through the core's ops (1788's decorated arithmetic); every
  other op's vectors still drop their decorations

### 2026-09-26 revision: M13g part 1, signals and bare constructors, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 1").
the signals as the owner chose them (Q1): `UndefinedOperationError(ValueError)` raises,
`PossiblyUndefinedOperationWarning(IntervalWarning)` would warn and return. `intervals/literals.py`
reads 1788's interval literals, a syntax separate from `MultiInterval.parse`, **exactly**: a decimal
is the rational it spells, so validity is decided exactly and the warning is never emitted today.
`text_to_interval` and `nums_to_interval` give a bare `MultiInterval` with an infinite end open.
"no NaI: invalid input raises" is in `REASONS` (D16) with the 52 former `[nai]` rows and every
`isNaI`; **a new category, "exact parsing decides validity", is PROPOSED for the owner** (the 4
vectors expecting `PossiblyUndefinedOperation`, "ieee 1788" above). the decorated type is still to
come

### 2026-09-26 revision: M13d, power and the rest of the elementary functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13d). D11 moved into "current
design" (arithmetic: power; elementary and step functions: pow, hypot and the twelve functions;
ieee 1788: the counts). the choices D11 and the plan left open, each the most conservative
reading, are now current design too:
* **a Fraction with an integral value is pown**, as D11 says for a float: `A ** Fraction(4, 2)` is
  `A ** 2`. any `MultiInterval` exponent is pow, even a degenerate integral one, as D11 says
* **`0 ** y` for y <= 0 is outside the domain**, dropped with the `DomainClippedWarning` negative
  bases get, not an `IndeterminateResultWarning`: D11 names the domain as x > 0 plus x = 0 where
  y > 0. the one-sided limit `0 ** -1` = inf, which the M12 rule for a domain's end would allow, is
  not taken, since 1788's domain excludes those points outright; so `[0, 1] ** [-1]` is `[1, inf)`
  while pown's `[0, 1] ** -1` is `[1, inf]`
* **±inf in pow are points where the power has a limit** (`inf ** 2` = inf, `(1/2) ** inf` = 0),
  and `1 ** ±inf`, `inf ** 0` are indeterminate points with atan2's treatment: a box that is one of
  them contributes nothing and warns, a larger box keeps the values of its other points
* **the poles at 0 of cot, csc, coth, csch and an odd negative root follow `reciprocal`** (a
  one-sided limit at a piece's end, closed iff the piece holds 0; the point 0 alone empty with an
  `IndeterminateResultWarning`), the plan's "split a piece as tan's do" for the poles inside
* **acot is continuous**, `pi/2 - atan x` with values in (0, pi), fi_lib's reading (its 30 vectors
  are all on positive operands, where the two readings agree); `atan(1/x)`, with a jump at 0 and
  values in (-pi/2, pi/2], is the other convention
* **rootn(n) for every int n other than 0**, 1788's: n < 0 is the root of `1/x`; an even root's
  domain is x >= 0 with `rootn(0, -2n)` = inf at the domain's end; an odd negative root has the
  two-sided pole at 0; `rootn(x, 0)` is a `ValueError`, a non-int degree a `TypeError`
* **hypot is rounded once** from the exact sums of squares, like `fma`; its class is the receiver's
* the 1939 vectors of the 14 ops match in both passes with no divergence row. 41 of them end at or
  are the pole 0 of cot, csc, coth or csch: `[0]` is empty in both, and a piece ending at 0 has the
  same closed hull whether the infinity is attained or not. `0 ** y` for y <= 0 is outside both
  domains (`pow [0.0,0.0] [0.0,0.0] = [empty]`); no vector has `acoth` at ±1, and the input rule
  never gives a closed infinity. itf1788 now 7314 vectors of 83 ops; 2228 statements of 28 ops
  skipped (M13e, M13g)

### 2026-09-26 revision: M13f, cancellation, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13f). D13 moved into "current
design" (arithmetic, ieee 1788, testing), and its new residual category, "cancellation as a
Minkowski difference", is in the divergence table. the choices D13 and the plan left open, each
the most conservative reading, are now current design too:
* **float operands round outward in `OutwardMultiInterval`**, to the tightest enclosure of the exact
  `X`, and to nearest in `MultiInterval`, computed exactly and rounded once like `fma`. the vectors
  require it (`cancel.itl:218`, `:221`: 1788's answer is the hull of the exact difference) and it
  is that class's promise, every `x` that fits is in the result. inward rounding would certify
  `B + X ⊆ A` but break the outward pass of 16 matching vectors and the class's meaning; it is
  **open to the owner** as a separate method or type if a certified inner answer is wanted.
  today exact operands (a float as `Fraction(f)`) give it
* **an empty `B` gives `[-inf, inf]`, for `A = ∅` too**: every `X` has `∅ + X = ∅ ⊆ A`, so the
  largest is the whole line. 1788 answers `∅` for `cancelMinus [empty] [empty]` (and entire for a
  non-empty `A`, which matches); the two `[empty] [empty]` vectors are rows under the new category
  with their own reason. special-casing `∅ ⊖ ∅ = ∅` would contradict "the largest `X`"
* **the infinite points follow the library's `+`**: `inf + -inf` has no value, so `{inf} + [-inf]`
  is empty and `inf` fits any `A` when `B = [-inf]` (`[0, 1].cancel_minus([-inf])` = `[inf]`)
* **methods on the receiver's class**, as `fma`: `OutwardMultiInterval` only when `self` is one; a
  number is coerced to a point; no warning, not even for an empty operand
* the 242 vectors: 148 match in both passes; 94 (47 keys) are rows under the new category, 45 where
  1788 has no answer and 2 for `[empty] [empty]`. itf1788 now 5375 vectors of 69 ops, 4167
  statements of 42 ops skipped

### 2026-09-26 revision: M13c, the interval orders and the interior, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13c). D10 moved into "current
design" (comparisons, set operations and size, testing). the choices D10 and the plan left open,
each the most conservative reading, are now current design too:
* **"equal infinite ends count" exactly as 1788 writes it**: two starts at -inf, two ends at +inf.
  a start at +inf or an end at -inf (`[inf]`, `[-inf]`, points 1788 has no interval for) is not
  strictly less than itself; the looser "any equal infinite ends" would make `[inf]` strictly less
  than `[inf]`
* **`.interior` is the interior in the reals**: the infinite ends are opened too, so a closed end
  at ±inf drops out (`[5, inf]` → `(5, inf)`, `[inf]` → `∅`, `[-inf, inf]` → `(-inf, inf)`), the
  plan's "every end opened". the extended reals' interior would keep `(5, inf]`. degenerate
  pieces drop out; the class is kept (`OutwardMultiInterval`)
* **open or closed ends do not matter to the orders** (they are on the ends as values, the hull's),
  so `[0, 2]` is weakly but not strictly less than `[0, 2)`; the orders coerce a number to a
  point and raise `TypeError` on anything else, as `within` does
* **empty sets follow the vectors**: two empty sets are weakly and strictly less than each other,
  an empty and a non-empty one neither; `∅.within(B.interior)` is True
* the 184 vectors match with no rule and no listed row (12 generated `[nai]` rows); itf1788 now
  5133 vectors of 67 ops, 4409 statements of 44 ops skipped

### 2026-09-26 revision: M13b, the numeric functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13b). D9 moved into "current
design" (set operations and size, ieee 1788, testing). the choices D9 and the plan left open, each
the most conservative reading, are now current design too:
* **methods, not properties**: `mid()`, `rad()`, `wid()`, `mag()`, `mig()`, `mid_rad()` (the plan's
  spelling; `inf` and `sup` stay properties)
* **"float" is the operand's kind**: any finite float end makes every number a float, rounded once;
  otherwise int or Fraction. the one exception is D9's ±max float for a half-bounded `mid`
* **`rad` is 1788's radius around the rounded midpoint**, not half the width rounded up, because the
  vectors require it (`rad [1, 1 + 3ulp]` = `2ulp`); `mag` rounds up and `mig` down (exact on
  double ends); both classes give the same numbers, the direction being the function's
* **edge cases no source covered**: a point, `[inf]` included, is its own midpoint with radius and
  width 0; an exact end past max float keeps the midpoint inside the hull; the empty set raises
  `ValueError` naming the missing quantity, with no warning
* **the adapter gained a float pass for the six ops** (`test_vector_float`, both classes, compared
  as they are): the first pass's exact operands would never exercise the library's own rounding
* the 167 vectors match, no divergence row beyond the 6 generated `[nai]` ones; itf1788 now 4949
  vectors of 64 ops, 4593 statements of 47 ops skipped

### 2026-09-26 revision: M13h, the reductions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13h). no D row covered it, so
the choices below were made while building, each the most conservative reading of the plan, and
are now "current design" (arithmetic, ieee 1788, testing):
* **exported from `intervals`**: `sum_`, `sum_abs`, `sum_sqr`, `dot`, following M13e's plan for
  the reverse ops (`sum_` keeps its underscore so it never shadows the builtin)
* **the direction is a keyword-only string**, `rounding='nearest'` by default, `'down'`, `'up'`:
  no public API took a direction before, and `rounding.py`'s constants stay internal
* **the result is always a float**, also for int and Fraction operands (the plan's "rounded once",
  an exception to "int and Fraction are never rounded", since a reduction is 1788's float op)
* **no value raises**: a `nan` operand, `inf + -inf`, `0 * inf` raise `ValueError` where 1788
  answers `NaN`, as `nan` does in a constructor and as D9 decided for `mid` of the empty set; the
  adapter reads the error as `NaN`. no warning. open to the owner: returning `nan` instead (1788,
  ieee 754 and python's float `sum` do) would be a looser, compatible change later
* the 15 reduction vectors match with no divergence row; itf1788 now 4782 vectors of 58 ops, 4760
  statements of 53 ops skipped

### 2026-09-25 revision: M13a and M14's fuzz job and oracle built

built across 2026-09-25 and 26 (measured 2026-09-26), the first session M13 suggests; the details
and numbers are in v2-implementation-plan.md (M13a, M14). D14 and D15 moved into "current design"
(ieee 1788, testing):
* **vendored** all 19 files of oheim/ITF1788 at `b6ee1e2` and its three licence files, byte-exact,
  blob hashes checked; reading every header corrected D15: five files carry the all-permissive
  notice (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`)
* **the parser reads every statement** (9542): a strict tokenizer, brace-counted testcases,
  `NaN`, quoted text, `{...}` lists, `signal` clauses, two-value results. the old one ended a
  testcase at a list's `}` and silently dropped 11 statements of `libieeep1788_reduction.itl`
* **divergence keys drop decorations**, so the 18 old rows are 15 keys; the `[nai]` operands are
  rows generated in code under "decoration expectations" until M13g, each failing as stale once
  its vector matches. 4767 vectors, 49 keys, 0 unknown failures
* **fuzzing is a multiplier, not a profile's count**: a test's own `max_examples` overrides any
  profile, so under `HYPOTHESIS_PROFILE=fuzz` the conftest multiplies each test's own, once per
  function; the weekly `fuzz.yml` saves the example database even on failure. its first local run
  found `test_sound_float_identity_rounding[div]` red on mixed Fraction and float operands:
  python rounds `Fraction(1,3) / 2.75` twice, the library once, as "arithmetic" says it should. the
  fault was the test oracle's, which now rounds a mixed pair once (`tests/oracles.py::_once`)
* **the flint oracle requires correct rounding**, not a tolerance: no double may lie between an end
  and the value, decided by arb's proven comparisons, more bits when undecided; values arb cannot
  hold near the float range are compared through their logs. it found no bug

### 2026-09-25 revision: M13 and M14 planned (not built)

owner requests: vendor the whole itf1788 suite (the maintained fork, oheim/ITF1788 at `b6ee1e2`,
19 files) and implement every op in it, as M13; and fuzzing, as M14. the owner accepted every
recommended default, recorded as D9–D17 in v2-implementation-plan.md section 0 with the details.
none of this is "current design" yet: each point moves up into it when its sub-task is built.
in short:
* **D9** `mid`, `rad`, `wid` are of the hull; `mag`, `mig` of the set (`mig([-3,-2] ∪ [2,3])` = 2)
* **D10** `weakly_less`, `strictly_less` for 1788's `less`, `strictLess` (`<` stays pointwise); a
  `.interior` property, so 1788's `interior(A, B)` is `A.within(B.interior)`
* **D11** `**`: an integral exponent is pown as today; a non-integral or interval exponent is 1788's
  `pow`, negative bases dropped with `DomainClippedWarning` (so `MI(-3,1) ** MI(2)` = `[0, 1]`);
  `b ** A` through `__rpow__`; 3-argument `pow` dropped
* **D12** reverse ops with periodic answers: exact pieces up to 1000, else the hull + `HullWarning`
  (built 2026-09-26 at M13e; now in "current design", elementary and step functions, empties and
  warnings, ieee 1788)
* **D13** `cancel_minus` is the Minkowski difference (the largest `X` with `B + X ⊆ A`); where 1788
  answers "no answer" with entire, ours is a real set, under a new residual category
  "cancellation as a Minkowski difference"
* **D14** `python-flint` (Arb) is the independent oracle for the elementary functions, test-only
  (built 2026-09-26; now in "current design", testing)
* **D15** the fork's LGPL-2.1+ files (`mpfi`, `fi_lib`, `c-xsc`) are vendored unmodified as test
  data with their licence files; the wheel ships only `intervals/` (built 2026-09-26; now in
  "current design", ieee 1788. five files are all-permissive, not the two `ieee1788-*` alone)
* **D16** decorations, NaI and 1788's constructors in a separate decorated wrapper type, brought
  forward from "later"; the core stays undecorated. open: 1788's signals as warnings or exceptions
* **D17** M13 does not block 2.0.0, and the release is in no hurry

### 2026-09-25 revision: M12, the unblocked backlog built

the (b) items of the M11 backlog, built in one session by owner request ("build all the things that
are unblocked"); the choices made while building are recorded here and in
v2-implementation-plan.md (M12), and written into "current design" above:

* **functions do not go through the applicator**: an irrational value has no exact `Value` for the
  descriptor's `fn` to return, and sin/cos/tan split at irrational points. `functions.py` has its
  own small evaluator, as `modulo.py` does; it reuses `applicator.split_pieces` and `warn`
* **no libm**: every value comes from a pure-python, correctly rounded evaluator (`elementary.py`),
  so results are the same on every platform and a directed rounding is a true bound. this replaces
  "±1 ulp around trig is pragmatic, not rigorous — document"
* **an irrational value of an exact operand is its tightest float enclosure** (`sqrt([2])` is the
  open one-ulp piece), not a nearest float: an exact operand never loses its true value
* **a moved end is open, outward**: attainment is decided on exact values, so the ends that directed
  rounding moved are open; to nearest, flags stay conservative
* **the end of a function's domain is a point of it** where the one-sided limit exists
  (`log(0)` = -inf, `atanh(±1)` = ±inf); 1788 drops those points, and the 11 vectors where that
  shows are divergence rows (degenerate infinities)
* **atan2 on the negative x axis is pi** (there is no -0); `(0, 0)` and `(±inf, ±inf)` have no value
* **Allen relations stay cut-based** (`[1, 2]` and `[2, 3]` overlap): a new divergence category,
  "cut-based relations", for the 7 overlap vectors where 1788 says meets
* **the step functions share one engine and one cap** (`steps.py`); `modulo.floor` delegates to it
* names: `minimum`/`maximum` (numpy's names for the pointwise min and max; builtin `min` needs a
  bool from `<`), `round_ties_away` (1788's roundTiesToAway), `OutwardMultiInterval`

### 2026-09-23 revision: implementability review

a review of whether the plan could be built as written. the design stands; these holes would have
been built in, two of them past the planned tests. all are written into "current design".

* **corner-flag rule at infinity** (unsound): add/sub/mul are flat at ±inf, so "closed iff both
  corners closed" excluded attained infinities — `[inf] + (1, 2)` came out `∅`, `[1, inf] + [0, 1)`
  came out `[1, inf)`. infinite result endpoints now always go through the attainment predicate.
  the planned attainment tests ran "on int/Fraction only", and `inf` is a float, so they would not
  have seen it
* **mul splits at zero** (not sharp): `[-1, 1] * [inf]` hulled to `[-inf, inf]` although only
  `[-inf] ∪ [inf]` is attained. sound, so the fuzz passed, and endpoint checks cannot see an
  interior point. an interior-sharpness test is added
* **±inf is exact** whatever its python type, so the exact/float split and the rounding hook
  have a rule for it
* **`-` is arithmetic subtraction** (as v1, `multi_interval.py:1188`); set difference is a named
  method. the 2026-08 dunder list had given `-` to both
* **`==` does not coerce** scalars (hash contract)
* **relations return bool**; only pointwise comparisons return a `TruthSet`
* **warnings**: import-time filters use `append=True`; property tests that hit indeterminate cases
  opt out of the suite's warnings-as-errors
* **modulo with infinite operands** (D8): clip an infinite dividend, follow python for an infinite
  divisor
* **itf1788 output rule** hulls the expected value too
* **division** computed directly, not as `a * (1/b)` (double rounding on floats)
* implementation detail, in the implementation plan: `typing.NamedTuple` refuses a `__new__`
  override (`AttributeError: Cannot overwrite NamedTuple attribute __new__`, executed 2026-09-23 on
  python 3.13.15), so `Cut` normalizes in a subclass of the named tuple

### 2026-09-23 revision: D1–D6 settled

owner decisions on the review's open list (v2-implementation-plan.md section 0).

* **D1, closure at infinity**: flags propagate; an infinite endpoint is closed iff attained.
  "closure over attainable values and their limits" is gone: it made `1/(-1, 0)` = `[-inf, -1)`,
  which contradicted both `1/[1, inf)` = `(0, 1]` and the involution (`1/(1/(-1, 0))` came back
  `(-1, 0]`)
* **D2, indeterminate corners**: limit along the box, as reciprocal already did at zero. replaces
  "`[-inf, -1] * [0]` is the entire line". this is what forces D7's `∅` for `[0] * [inf]` and
  `[inf] - [inf]`; under the entire-line rule `∅` was allowed but not forced
* **D3**: exact division as Fraction, integral Fractions normalized to int
* **D4**: time layer deferred; v1's `time_interval.py` (and v1 `multi_interval.py` under it) stay
* **D5**: modulo for every sign combination is required before release, as its own milestone; not
  shipped Q1-only
* **D6**: `[a, inf]` literal — already the design, recorded as settled
* **v1 files** (owner, later the same day): archived to `archive/v1/`, never deleted, until v2
  works. this includes `time_interval.py`, which returns on top of the v2 class; the D4 bullet's
  "stay" means "stay in the archive"

### 2026-09-23 revision: degenerate indeterminate boxes return ∅

> **amended same day**: D1 and D2 were decided after this entry; see the entry above.

owner decision (D7 in v2-implementation-plan.md). a box that *is* an indeterminate point — `1/[0]`,
`[0] * [inf]`, `[inf] - [inf]`, `[0] / [0]` — returns `∅` and emits `IndeterminateResultWarning`,
where the 2026-09-22 design gave `[-inf] ∪ [inf]` or the entire line.

**why.** inclusion isotonicity (`A ⊆ B ⇒ f(A) ⊆ f(B)`) forces it: `[0]` is inside both `[-1, 0]` and
`[0, 1]`, so `1/[0] ⊆ [-inf, -1] ∩ [1, inf] = ∅`; `[0] * [inf] ⊆ [0] * [5, inf] ∩ [0, 1] * [inf]`,
empty under the sharp corner rule. solvers (bisection, forward-backward contraction) need isotone
ops; it matches 1788's empty and removes the `1/[0]` row from the itf1788 divergence table.

**cost.** `1/(1/[inf])` = `∅`, and `1/x` round-trips only for sets with no degenerate piece at `0`,
`inf` or `-inf` (`[0] ∪ [1, 2]` → `[1/2, 1]` → `[1, 2]`). the direction tag under "later" stays the
recovery path.

**separately.** `f(A ∪ B) == f(A) ∪ f(B)` is false for reciprocal under the direction-from-the-piece
rule with *any* value of `1/[0]`: `A = [-1, 0)`, `B = [0]` gives `1/(A ∪ B)` = `[-inf, -1]` but
`1/A ∪ 1/B` = `(-inf, -1]` (with D1's flag propagation). that law is only `⊇` for reciprocal/div.

### 2026-09-22 revision: signed zero dropped

**how it got in.** v1's reciprocal returns the whole line for anything touching zero → fix by
splitting at zero → the pieces must reach infinity, so ±inf became points → `1/0` evaluated at an
endpoint needs a sign to pick `±inf` → `-0` imported from ieee 754 as a distinct point → six-state and
four-state enums, then three boundaries at zero, glued/strict modes, `same_set()`, `config.py`.

**why it is out.** the fourth step assumes `1/x` is evaluated at a scalar. an interval library
evaluates sets, and the set already carries the direction: `1/[-1, 0]` is `[-inf, -1]` because the
piece is negative, which is also Hickey's and 1788's answer. the sign is informative only for a
degenerate `[0]`, where it records how the zero was produced — dependency tracking, which interval
arithmetic gives up everywhere else. outward rounding never manufactures a signed zero from underflow
on a non-degenerate interval either, since the bound facing away from zero rounds away. as a position
in the order it also cost real correctness: `[-1, 0]` necessarily contains `-0`, so `1/[-1, 0]` grew a
phantom `+inf` under "signs pick the branch"; `(-0.0, 1)` contained `0.0`; ints cannot spell `-0`;
`[-0]`, `[+0]` and `[-0, 0]` were `==` and hash-equal yet gave different reciprocals.

> **superseded 2026-09-23**: `1/[0]` is now `∅`, so `1/(1/[inf])` = `∅` and the involution also
> fails at a degenerate zero piece. see the 2026-09-23 entry.

**what is lost.** `1/(1/[inf])` is `[-inf] ∪ [inf]` instead of `[inf]` — Kahan's involution argument,
the one real case for the sign bit. it breaks at both infinities symmetrically and nowhere else, and
is recoverable later as metadata (see "later").

> **superseded 2026-09-23**: "now" column, rows `1/[0]` and `[0] * [inf]` — both are `∅` + warn

| case          | pure math      | ieee 754 scalar | python         | ieee 1788                     | cset (Hickey)         | 2026-08 plan         | now                     |
|---------------|----------------|-----------------|----------------|-------------------------------|-----------------------|----------------------|-------------------------|
| `1/[0]`       | undefined      | ±inf by sign    | raises         | empty                         | `{-inf, +inf}`        | `[+inf]` or `[-inf]` | `[-inf] ∪ [inf]` + warn |
| `1/[-1, 0]`   | undefined at 0 | n/a             | n/a            | `[-inf, -1]`, -inf unattained | `[-inf, -1]`          | `[-inf,-1] ∪ [+inf]` | `[-inf, -1]`            |
| `1/[-1, 1]`   |                |                 |                | entire                        | `[-inf,-1] ∪ [1,inf]` | same                 | same                    |
| `[0] * [inf]` | undefined      | nan             | nan            | not expressible               | entire                | `[0, inf]` by signs  | entire + warn           |
| `-0.0` input  | same number    | distinct, `==`  | distinct, `==` | absent                        | absent                | distinct point       | normalized to 0         |

**other changes in the same revision**: the cut side is an `IntEnum`, not "epsilon", and there is no
third "exact" side; the truth set has four states so empty operands raise instead of being vacuously
TRUE; `size` in points rather than half-points, and the v1 ray bug noted; rounding is a hook with an
identity default enabled by type; no context managers of any kind; 1788 decorations deferred to the
solver; the itf1788 adapter gets an input rule for infinity; mixins replaced by kernel functions plus
one class file; relations defined on cuts.

### 2025-12 wishlist and negative-zero enums

* the range will be the affine extended real numbers, meaning support for ±inf along with negative zero, i.e.:
  `[-inf] + (-inf, 0) + [-0, 0] + (0, inf) + [inf]`
* divide by zero is supported, and there will be warnings for indeterminism
* newton's method solver as a test
* generalized function applicator as long as its continuous and differentiable
* forward mode autodiff
* numpy compat via https://data-apis.org/array-api/latest/ or `__array_ufunc__` or https://numpy.org/doc/stable/user/basics.interoperability.html
* consider ieee 1788 compatible decorations, although multi intervals actually support a bit more so not sure if it matters

### negative zero

> **superseded 2026-09-22**: there is no signed zero in v2. see "current design" and the 2026-09-22 log entry.

* note that while we accept that `-0 == 0`, `-0` will be stored as a separate value, and they interact as follows:
    * `(..., 0)` & `[-0]` will merge to form `(..., -0]`
    * `[0]` & `(-0, ...)` will merge to form `[0, ...)`
    * `(..., -0)` & `[0]` will merge to form `(..., 0]` (but this is somewhat questionable)
    * `[-0]` & `(0, ...)` will **not** merge
    * `[-0]` & `[0]` will merge to form `[-0, 0]`
* we can think of `-0` as a hyperreal number of the form `(0, -1ε)`, while plain `0` is `(0, 0ε)`
* we need `-0` to handle infinities:
    * `(0, inf]/0 == [-inf, -0)/-0 == inf`
    * `[-inf, -0)/0 == (0, inf]/-0 == -inf`
    * `0/0 == -0/-0 == inf * 0 == -inf * -0 == [0, inf]` (raises `IndeterminateResultWarning`)
    * `-0/0 == 0/-0 == -inf * 0 == inf * -0 == [-inf, -0]`(raises `IndeterminateResultWarning`)

maybe this enum, but this introduces a `(-0, ...)` that just feels wrong:

* -3 open end neg zero
* -2 open end
* -1 closed neg zero
* 0 closed
* 1 open start neg zero
* 2 open start

or a simplified version that might be more intuitive, but this introduces a weird symmetry break at 
`[-0]|(-0,...)` that cannot exist and can never merge, and `(...,-0)|[0]` that merges but introduces a 
spurious `-0`, so taking the reciprocal now produces a `-inf` that should not have been there:

* -2 open end
* -1 closed neg zero
* 0 closed
* 2 open start

### update

found that there's an ieee spec that does something similar

### v2 consolidated decisions (2026-08-16)

> **partly superseded 2026-09-22**: cuts, applicator, retirements and the 1788-as-adapter principle survive; signed zero, zero modes, config.py, mixins and the `measure` name do not. each subsection is marked.

supersedes the enum discussion above (kept for history). derivation: chat sessions 2026-08-15/16.

#### representation: cuts (boundaries), not endpoints

* store each bound as the *boundary where the set stops*, not "point + open/closed flag"
* every real x has two boundaries: just below = `(x, -1)`, just above = `(x, +1)`; a piece is a pair of cuts
    * `[a, b]` = `(a,-1),(b,+1)` — `(a, b)` = `(a,+1),(b,-1)` — `[a, b)` = `(a,-1),(b,-1)` — `[x]` = `(x,-1),(x,+1)`
    * empty iff start_cut >= end_cut, so `[1,1)` and `(1,1)` normalize to empty; `[2,1]` is still ValueError
* sort = plain lexicographic tuple compare; a normalized multi-interval's flat cut list is *strictly* increasing
* **there is no merge rule**: pieces merge iff `next.start_cut <= current.end_cut`
    * `[1,2] | (2,3]` tiles exactly (both cuts `(2,+1)`) -> merge; `[1,2) | (2,3]` has `(2,-1) < (2,+1)` -> the point 2 is missing -> gap
    * no `diff <= k`, no adjacency table, no special cases
* complement reuses the same cut tokens with start/end roles swapped (no epsilon negation); mirror = `(-v, -s)`
* the hyperreal story is the *documentation*, not the implementation: `(v,+1)` is the boundary at v+ε; only order matters, so tuples suffice
* v1's `{-1, 0, 1}` epsilon had a redundant state: closed-start-at-x and closed-end-at-x are different boundaries; cuts never conflate them

#### signed zero: keep it

> **superseded**: dropped 2026-09-22 (see log entry for why).

* -0 is direction info on a zero boundary, not a user-facing number. IEEE hands us the sign bit anyway (`-1.0 * 0.0 == -0.0`), and it's exactly what keeps reciprocal sharp under our closure-of-limits semantics (`1/[-5,-0] = [-inf, -0.2]`)
* zero has *three* boundaries instead of two: `(0,-1)` below -0, `(0,0)` between -0 and +0, `(0,+1)` above +0. side 0 is only legal at value 0 (one constructor check)
* the merge rules from the section above now fall out of plain comparison, nothing written down:
    * `(..., 0) | [-0]` -> merges -> `(..., -0]`
    * `[0] | (-0, ...)` -> merges -> `[0, ...)` (in cuts `(-0, ...)` *is* `[+0, ...)` — same boundary, the "feels wrong" state never exists)
    * `[-0] | (0, ...)` -> does NOT merge (+0 genuinely missing)
    * `[-0] | [+0]` -> merges -> the full zero
    * deviation from the old plan: `(..., -0) | [0]` does NOT merge (-0 genuinely missing). this was the case marked "somewhat questionable" above — merging it is what manufactured the spurious -0 and the phantom -inf under reciprocal
* the seam at zero is irreducible: -0 and +0 are two *adjacent* points in an otherwise dense order — an order-theoretic fact no encoding can hide. its entire footprint is one extra token at value 0. dropping -0 entirely is the only way to a perfectly homogeneous line (and would cost sharp division)

#### zero gluing: float semantics by default

> **superseded**: no zero modes, no context managers.

model the UX on how python/IEEE already treat -0.0 (equal, same hash, sign preserved through arithmetic, visible only via copysign / division):

* `zero_mode='glued'` (default): `==`/`hash`/membership/merging treat -0 == +0; sign tags on zero endpoints are preserved until an actual cross-zero merge consumes them, so `1/[-5,-0]` stays sharp even in glued mode. gluing only ever *widens* by the other zero point -> sound, never wrong
* `zero_mode='strict'`: no gluing; `[-0] | (0,...)` stays two pieces; for solvers and branch cuts (1/x, log, sqrt). toggle via context manager (same pattern as the warnings design)
* repr always faithful (prints -0 when present, like python floats); str likewise

#### comparisons

> **amended**: the truth set has four states (empty operands raise); relations are defined on cuts, not sup/inf.

* `< <= > >=` return a tri-state truth set (TRUE / FALSE / BOTH, since "every a op every b" can be both); `__bool__` raises on BOTH (numpy precedent), so `if a < b:` either works or fails loudly — never guesses
* `==` and `__hash__` are structural set equality (required for dicts/tests); pointwise equality is a method, and is BOTH for any non-degenerate `a == a` (document this, it surprises everyone)
* explicit `sort_key` (lex on cuts) for structural ordering; `sorted()` raising on ambiguous intervals is a feature
* relation vocabulary (the adjacent/adjoining/intersecting/overlapping todo): named set-level predicates — disjoint, adjoins (end cut == start cut), overlaps, contains, within, equals, before (sup A < inf B), after — plus certainly_/possibly_ modal variants
* allen's 13 relations are only JEPD for *contiguous* intervals; expose `allen(a, b)` restricted to contiguous pieces (raise otherwise, or caller passes hulls explicitly). optional: per-piece relation matrix, or the set of relations holding between any piece pair (allen's algebra natively reasons over relation sets). note cuts make the taxonomy *finer* than classical allen: tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing-one-point (`[1,2] ∩ [2,3] = {2}`) are distinguishable

#### division semantics vs ieee 1788 (they are not wrong, just different)

> **amended**: direction comes from the set, not a sign bit; `1/[0]` = `[-inf] ∪ [inf]`.
> **superseded 2026-09-23**: `1/[0]` = `∅` + warning, and results are attained values with flags propagating through infinity, not a closure over limits (see the 2026-09-23 entries).

* ours: closure over attainable values/limits (cset-flavored, cf. hickey/van emden paper in README todo) -> zero denominators contribute their limit infinities, signed zeros pick the branch
* 1788: division is the *inverse relation of multiplication* over the reals ({z : x = z·y}); y=0 contributes no z because z·0=1 has no solution -> `1/[-5,0] = [-inf,-0.2]` with decoration dropped to `trv` (partiality is recorded, not ignored). solvers get the split result via `mulRevToPair` (two-interval extended division)
* consequence: itf1788 vectors will disagree on division-by-zero cases *by design*; keep a documented divergence table rather than chasing exact matches
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = community test framework for it (test files in a small DSL, "itl"; used by IntervalArithmetic.jl et al)

#### housekeeping

> **amended**: `measure` renamed again to `size` (in points, not half-points); rounding is a hook with identity default, enabled by type.

* rename `cardinality` -> `measure`: it's a lex-graded size (rays, open length, closed endpoint count), not cardinality. the ω/ε gloss is fine intuition; we only compare these tuples, never do arithmetic on them
* delete `INFINITY_IS_NOT_FINITE`: affine extended reals, `[inf]` degenerate allowed; cuts encode `[a, inf]` vs `[a, inf)` naturally, and a mutable global that changes set semantics is a footgun
* v2 core type immutable + hashable; incremental building via a small builder (bisect-insert is 82x faster than re-sort per compare.py)
* directed rounding moves FIRST, before newton/autodiff: `math.nextafter` outward rounding behind a rounding-policy hook, gmpy2/mpfr later as tight mode. every arithmetic op touches endpoint computation — retrofitting means touching everything twice. (libm is not correctly rounded, so ±1 ulp around trig is pragmatic-not-rigorous; document)
* floordiv: enumerate integer points below a size cap, else return hull + warning — never silently drop openness
* testing: random fuzz vs sampling oracle for soundness (`op(x,y) ∈ op(A,B)` for sampled x,y) + attainment checks for closure — the pattern already validated by the modulo v3 work; itf1788 as conformance suite with the divergence table above

#### ieee 1788 conformance: test adapter, not a runtime flag

> **amended**: the adapter also needs an input rule (1788 unbounded → our open-at-inf).

* 1788 is further away than a flag: closed intervals only (no open bounds exist in the standard), connected only (no multi-intervals — everything hulls), ±inf never attained, decorations everywhere. a semantics flag would be `INFINITY_IS_NOT_FINITE` again ×10, and every flag multiplies the test matrix (`zero_mode × 1788_mode × rounding`)
* instead: a conformance adapter in the *test suite* — parse itf1788 vectors, run our ops, closed-hull the result, compare, consult a small divergence table. hulling absorbs most divergence automatically (`[1,2]/[-1,1]`: 1788 says entire, we say `[-inf,-1] ∪ [1,inf]`, hull = entire → match). residual table: empty-vs-degenerate-infinity division cases, domain-clipped functions, decoration expectations
* principle: flags change what existing objects mean; wrapper types add meanings. if real conformance is ever needed, it's a thin wrapper class in `ieee1788.py`, never a mode on MultiInterval

#### package architecture

> **superseded**: kernel functions over cut tuples plus one class file; no mixins, no config.py, no zero_mode.

split by domain, layered so imports only point downward (no cycles):

    intervals/
        __init__.py      public API assembly; constants (EMPTY, REALS, ...)
        errors.py        warning/exception classes                 -- layer 0
        config.py        zero_mode + rounding policy               -- layer 0
        cuts.py          cut encoding: (value, side) helpers,      -- layer 1
                         zero-sign logic, negation, ordering
        core.py          MultiInterval: immutable flat cut tuple,  -- layer 2
                         normalization sweep, eq/hash/bool,
                         Builder for incremental construction
        sets.py          union/intersection/difference/complement, -- layer 3
                         membership, subset, adjoins/overlaps
        relations.py     TruthSet tri-state, certainly_/possibly_, -- layer 3
                         sort_key, allen() (contiguous only)
        rounding.py      outward-rounding eval (nextafter now,     -- layer 3
                         gmpy2/mpfr later) — the single hook
        apply.py         generic applicator (see below)            -- layer 4
        arithmetic.py    + - * / reciprocal as op descriptors      -- layer 5
        modulo.py        v3 far-edge mod/divmod/floordiv           -- layer 5
        fmt.py           str/repr grammar + parser                 -- layer 5
        functions.py     sqrt/log/exp/trig via apply         (later)
        autodiff.py, solver.py, numpy_compat.py             (later)
        time_interval/   DateTime/TimeDelta layers over core (port)
    tests/
        oracles.py       sampling + attainment oracles — promote out of
                         modulo_v3_prototype; they're the library-wide
                         test pattern, not a modulo tool
        itf1788/         vector runner + divergence table
        test_<module>.py

* one public class assembled from mixins: `sets.py` / `relations.py` / `arithmetic.py` each export a mixin, `core.py` composes `MultiInterval(SetOps, Relations, Arithmetic, Base)`. domain split without circular imports, dunders stay on one discoverable type
* **refinement of the zero-gluing section above**: `zero_mode` is a *construction-time* policy, not an ambient switch — eq/hash must not change when a context flag flips or dicts corrupt. normalization glues (or not) under the mode active at construction; after that the object is just data. `==`/`hash` are ALWAYS glued (float precedent: `0.0 == -0.0` unconditionally); strict structural comparison is a named method (`same_set()`). the "context manager" from above scopes which policy new objects are built with
* `_consistency_check` goes under `if __debug__:` (stripped by `-O`) — v1 runs an O(n) scan at the top of nearly every public method; that's the real hot cost, not corner counts. correctness lives in the oracle tests instead

#### generic applicator: keep it, but shape-then-attainment

* keep the generic endpoint applicator — not to save code, but because it's the ONE place where directed rounding, closure decisions, and domain splitting get woven into endpoint computation. per-op code would reimplement all three per op, forever. 4-vs-2 evals is noise next to interpreter overhead
* v1's epsilon propagation through corners is UNSOUND, not just inelegant: `[0,1] * (2,3)` — min corner `0*2=0` gets eps `0 or 1` = open, v1 returns `(0,3)`, but 0 is attained (`0 * 2.5`); true result `[0,3)`. the flat spot at zero makes the result independent of the open operand. same disease the modulo v3 work diagnosed (604 closure faults from epsilon propagation), here it excludes attainable values → containment violation
* restructure around the modulo lesson:
    1. locations first, all endpoints treated closed (corner min/max — correct for coordinatewise-monotone and bilinear ops)
    2. closure per endpoint separately: strictly-monotone ops → corner-flag rule is provably fine (fast path, zero cost); flat-spot ops (mul at 0, pow, min/max-like) → attainment check
    3. driven by a small op descriptor per operation: monotonicity directions (gives add/sub a 2-corner fast path for free), flat-spot predicate, rounded-eval pair from `rounding.py`

#### more v1 retirements

* empty operands in arithmetic propagate: `A / ∅ = ∅` (vacuous union over no divisors), same for `∅ / B`, `A + ∅`, etc. — this matches ieee 1788 (empty in → empty out) and the other interval libraries, and mirrors nan propagation in floats, so silent propagation is the standard-aligned default. **note (owner):** silent empty propagation is effectively implicit nan/null propagation and can be annoying to debug later — so emit a dedicated `EmptySetPropagationWarning` on division (maybe all arithmetic) with a default `'ignore'` filter installed at import; solver code opts into `warnings.simplefilter('error', EmptySetPropagationWarning)` as a tripwire. exceptions stay reserved for malformed construction (`[2,1]`)
* parsing moves out of `merge()` into `fmt.py`, regexes compiled at module level (hot-path compile in v1)
* one `_coerce(other)` in core instead of per-method isinstance ladders
* the `inplace=` dual API disappears with immutability — large chunk of v1 surface gone for free
* retire `interval.py` (the alternative debug implementation) — the sampling oracle does that job better, and a second implementation is a maintenance tax
* keep `__getitem__` slicing (`x[0:5]` as restriction); `in` = scalar membership + documented subset alias; subset stays a named method since `<=` is the tri-state comparator
* `__bool__` = non-empty, explicitly (set precedent) — distinct from tri-state comparisons, which raise on ambiguity
