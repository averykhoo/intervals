# `MultiInterval` v2 plan

two parts. **current design** is normative: if the code and that section disagree, one of them is a
bug. the **decision log** below it is history, kept verbatim, with a marker wherever a later decision
superseded it.
open work and open questions for the owner (including the ones raised in the decision log's
2026-09-25/26 entries) live in `HANDOFF.md`; the milestones are in `v2-implementation-plan.md`.

## current design (2026-09-23; brought up to date with the build at M12, 2026-09-25, and M13a,
M13b, M13c, M13d, M13f, M13h and M14's fuzz job and oracle, 2026-09-26)

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
  after, adjoins, overlaps, contains, within and allen; disjoint, equals and the modal variants
  are functions in `relations.py`. (`sup A < inf B` is wrong for `[1,2)` before `[2,3]`).
  relations return plain `bool` — they are set-level facts; only the pointwise `< <= > >=` return a `TruthSet`. so
  `before([1,2), [2,3])` is True and `before([1,2], [2,3])` is False, while `[1,2) < [2,3]` is
  `{T}` and `[1,2] < [2,3]` is `{T, F}`. for non-empty operands `before(A, B)` is exactly
  `(A < B).certainly`
* `allen(a, b)` on contiguous pieces only (raise otherwise). cuts make it finer than classical Allen:
  tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing one point (`[1,2] ∩ [2,3] = {2}`)
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
  end (`rounding.round_rational`). that is the tightest float enclosure, so gmpy2/mpfr would only be
  faster, not tighter. mixed with a `MultiInterval` on either side, the result is outward: the
  subclass overrides every reflected dunder, which python requires before it tries the right
  operand first. int and Fraction are exact and never rounded. `mod`, `floordiv`, `fma` and the step
  functions have no hook: they compute exactly and round once (`rounding.round_piece`), to nearest
  or outward by type. poles never go through the hook, and a float piece that rounding squeezes to
  one point keeps that point, closed
* **flags at rounded ends**: outward, attainment is decided on exact values (an `OUTWARD`
  descriptor's `fn` is exact too), so an end that directed rounding moved is **open** — nothing
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
  `TypeError`: dropped (not 1788; v1 had it on integers only)
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

### elementary and step functions (M12, M13d)

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
* **reverse ops** (M13e, D12; `intervals/reverse.py`, exported from `intervals`; built 2026-09-26
  for `sqr_rev(c, x=REALS)`, `abs_rev(c, x=REALS)`, `pown_rev(c, n, x=REALS)`, `cosh_rev(c,
  x=REALS)`; sin, cos, tan, mul and pow to come): each is `{t ∈ x : f(t) has a value and f(t) ∈ c}`
  for the library's own f at a point, an exact multi-interval, where 1788 answers its hull. `x`
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
      result never leaves `x`. the result is an `OutwardMultiInterval` if either operand is one
    * an empty operand gives `∅` and an `EmptySetPropagationWarning`, as the functions do; an empty
      answer from non-empty operands (no solution) warns nothing. a number is a point; `n` must be
      an int (not bool), else `TypeError`
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
  unbounded piece); shown by default, like `IndeterminateResultWarning`. all four subclass
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
    * residual divergence table: degenerate infinities, domain-clipped functions, decoration
      expectations, (added at M12) cut-based relations, and (added at M13f, approved with D13)
      **cancellation as a Minkowski difference**: where 1788's `cancelMinus`/`cancelPlus` answer
      entire as "no answer", ours is the real set of the fitting `x`, and for `[empty] [empty]` the
      whole line where 1788 answers `∅`. (`1/[0]` is not a row: both give empty.) keyed on the
      statement with its decorations stripped (`tests/itf1788/test_itf1788.py::key`) since M13a.
      measured 2026-09-26 (M13d), the current count: 19 files, 9542 statements; 7314 vectors of 83
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
      than the vector", PROPOSED at M13e and not yet approved by the owner**: 1788's end for
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
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = the
  community test framework and its `itl` vector DSL. all 19 `.itl` files of the maintained fork,
  oheim/ITF1788 at `b6ee1e2`, are vendored unmodified with its `LICENSE`, `NOTICE` and
  `COPYING.LESSER` (D15, M13a, 2026-09-26; they replaced the 7 from nehmeier/ITF1788 `e0e0d7e`).
  each file keeps its own licence: Apache 2.0 for the 11 `libieeep1788_*`, LGPL-2.1-or-later for
  `mpfi`, `fi_lib`, `c-xsc`, all-permissive for `ieee1788-constructors`, `ieee1788-exceptions`,
  `atan2`, `abs_rev`, `pow_rev` (`tests/itf1788/README.md`). they are test data: the wheel ships
  only `intervals/`. `git hash-object` of all 22 files equals the fork's blob at the pin (checked
  2026-09-26), and `tests/itf1788/.gitattributes` marks them `-text` so a checkout keeps the bytes
* decorations (`com/dac/def/trv/ill`) are **not in the core**. they answer "was f defined and
  continuous on the whole input", which the result set cannot (`sqrt([-1,4])` = `[0,2]` either way),
  and only solver existence proofs need that. when the solver comes, a decorated wrapper type; until
  then `DomainClippedWarning`

### package layout

modules export pure functions over cut tuples; one class file on top binds the dunders. no mixins.
imports only point downward.

    intervals/
        errors.py          warning and exception classes
        cuts.py            Side, Cut, below()/above(), mirror, -0.0 normalization
        kernel.py          normalize sweep; union/intersection/complement/difference; membership;
                           interior (M13c); size; Builder (collect, sort once, sweep)
        fmt.py             format and parse cut tuples; regexes compiled at module level
        relations.py       TruthSet, pointwise compare, relation predicates, allen(),
                           the interval orders weakly_less strictly_less (M13c)
        rounding.py        rounding an exact value to a double: nearest, down, up
        applicator.py      op descriptor, corner evaluation, closure pass, rounding hook
        ops.py             neg pos absolute reciprocal add sub mul div, power (int exponents),
                           minimum maximum as descriptors; the OUTWARD descriptors; fma;
                           cancel_minus cancel_plus (M13f)
        modulo.py          v3 far-edge mod / floordiv / divmod_ (floor from steps)
        steps.py           floor ceil trunc round round_ties_away sign: enumerate or hull
        elementary.py      correctly rounded elementary functions at one exact point
        functions.py       the elementary functions and atan2 over cut tuples
        reductions.py      sum_ sum_abs sum_sqr dot over numbers: exact, rounded once (M13h)
        numeric.py         mid rad wid mag mig mid_rad of a set: exact, or rounded as 1788 (M13b)
        multi_interval.py  the class and OutwardMultiInterval: immutable cut tuple; _coerce
                           (numbers and intervals only — strings go through an explicit
                           parse()); one-line dunders. arithmetic
                           owns `+ - * / // % **` and unary `- + abs`; set algebra is `| & ^ ~`
                           plus named methods (`difference` has no operator, because `-` is
                           arithmetic subtraction, as in v1)
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
                           test_cancel.py, cancellation, M13f)

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
* a **fuzz profile** (M14, 2026-09-26): `HYPOTHESIS_PROFILE=fuzz` makes `tests/conftest.py` run
  every hypothesis test randomized, with no deadline, at `FUZZ_MULTIPLIER` (default 100) times its
  own `max_examples`; unset, the conftest does nothing, so the gate keeps `default` locally and the
  derandomized `ci` under GitHub Actions. `.github/workflows/fuzz.yml` runs it weekly and on
  `workflow_dispatch`, never on push, carrying `.hypothesis/` between runs and uploading it with
  the log on a failure

### later (not in v2.0)

* a **direction tag on a degenerate zero piece** if a solver ever needs `1/(1/[inf]) == [inf]`:
  metadata that `==` and hash ignore, created only by limits (`1/[±inf]`, `exp([-inf])`), consumed
  only by branch-at-zero functions. never a position in the order — that is what the signed-zero seam
  was
* a decorated wrapper type, with the solver. owner 2026-09-25: brought forward to
  `v2-implementation-plan.md` M13g, for the itf1788 decoration vectors; the core stays undecorated
* forward-mode autodiff, newton's method as a test, numpy compat (array API / `__array_ufunc__`),
  gmpy2/mpfr as a faster backend for `elementary.py` and the outward hook (not a tighter one).
  owner 2026-09-26: numpy and gmpy2/mpfr recorded, not now

## decision log

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
  method with forward-mode autodiff, the demonstration of what multi-intervals are for

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
