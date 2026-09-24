# `MultiInterval` v2 plan

two parts. **current design** is normative: if the code and that section disagree, one of them is a
bug. the **decision log** below it is history, kept verbatim, with a marker wherever a later decision
superseded it.

## current design (2026-09-23)

### domain and semantics

* values: the affine extended reals `[-inf, inf]`, both infinities as points. **one zero.** `-0.0` is
  normalized to `0` at construction and the sign bit is never consulted (as v1 already does)
* `[inf]` and `[-inf]` are legal degenerate intervals; `[a, inf]` and `[a, inf)` are different sets
* number types: int and Fraction (exact, never rounded), float (endpoint arithmetic goes through the
  rounding hook, identity by default — see arithmetic). datetime/timedelta: the time layer is
  deferred; v1's `time_interval.py` is archived with the rest of v1 and returns once it runs on
  the v2 class (D4)
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
* consequence: `1/x` is an involution only on sets with no degenerate piece at `0`, `inf` or `-inf`.
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
  `A.end == B.start`, plus disjoint / overlaps / contains / within / equals and certainly_/possibly_
  variants. (`sup A < inf B` is wrong for `[1,2)` before `[2,3]`). relations return plain
  `bool` — they are set-level facts; only the pointwise `< <= > >=` return a `TruthSet`. so
  `before([1,2), [2,3])` is True and `before([1,2], [2,3])` is False, while `[1,2) < [2,3]` is
  `{T}` and `[1,2] < [2,3]` is `{T, F}`. for non-empty operands `before(A, B)` is exactly
  `(A < B).certainly`
* `allen(a, b)` on contiguous pieces only (raise otherwise). cuts make it finer than classical Allen:
  tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing one point (`[1,2] ∩ [2,3] = {2}`)

### arithmetic

* one generic applicator, **shape-then-attainment** — the modulo v3 lesson. v1's epsilon propagation
  is unsound, not just inelegant: `[0,1] * (2,3)` gives `(0,3)` but 0 is attained by `0 * 2.5`; true
  result `[0,3)`
    1. split at the domain points the op descriptor names (zero for reciprocal/division **and
       mul**, sign-pure quadrants for modulo). a piece containing 0 splits into `[lo, 0]` and
       `[0, hi]` with the zero in both halves (keeping its flag); a degenerate `[0]` stays one piece
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
       injective in each argument there. every other endpoint uses the op's attainment predicate,
       tested against the **full** multi-interval operands, not per piece pair:
        * **every infinite result endpoint.** add/sub/mul are flat at ±inf (`inf + y = inf` for
          every finite `y`), so the corner-flag rule is wrong there: `[inf] + (1, 2)` would give
          `(inf, inf)` = `∅` instead of `[inf]`, and `[1, inf] + [0, 1)` would give `[1, inf)`
          instead of `[1, inf]`. the rule: an infinite result endpoint is attained iff a closed
          infinite operand endpoint combines with some non-indeterminate partner, or a pole at a
          closed zero produces it (reciprocal)
        * flat spots at finite values: mul at 0, pow, min/max-like
    4. union the piece results, normalize
* op descriptor: monotonicity directions (2-corner fast path for add/sub for free), split points,
  flat-spot / attainment predicate, rounded-eval pair
* **division is computed as its own op** (`a / b` at the corners), not as `a * (1/b)`: the
  reciprocal identity defines the semantics (splits, poles, D2 corners), but evaluating through it
  would round twice on floats
* division: split the denominator at zero, direction rule as above. `A / ∅ = ∅`. exact operands
  divide as Fraction (see number types)
* floordiv: enumerate integer points below a size cap, else hull + warning — never silently drop
  openness (v1's `[1,2) // 1` = `[1,2]` is wrong; should be `[1]`)
* modulo: the v3 far-edge algorithm, `references/modulo-derivations/claude-fable/`, for every sign
  combination before release. Q1 (dividend ≥ 0, divisor > 0) is proven; Q3/Q4 follow from Q1/Q2 by
  the antipodal identity; Q2 needs its own primitive derivation (design notes §2, Thm B); operands
  crossing zero split into sign-pure pieces; a divisor touching zero drops 0 with
  `DomainClippedWarning`
    * infinite operands (D8, owner-confirmed 2026-09-24): a dividend of ±inf has no value (python gives `nan`) and is dropped
      with `DomainClippedWarning`; a finite dividend mod a divisor of ±inf follows python's scalar
      result, which is also the limit along the box (`3 % [inf]` = `[3]`, `-3 % [inf]` = `[inf]`,
      `0 % [inf]` = `[0]`)
* **rounding**: the hook exists from day one in the descriptor (retrofitting touches every op twice)
  but is **identity by default**. int and Fraction are exact and never rounded. outward
  `math.nextafter` (later gmpy2/mpfr as tight mode) is enabled by a subclass or factory, never by a
  flag or context manager: wrapper types add meanings, flags change what existing objects mean. once
  an endpoint is rounded it is attained by nothing, so the open/closed flag on a rounded float
  endpoint is conservative, not a promise; attainment tests run on exact types only (±inf counts
  as exact, see number types). libm is not
  correctly rounded, so ±1 ulp around trig is pragmatic, not rigorous — document

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
* exceptions only for malformed construction (`[2,1]`, bad types)
* **no ambient modes**: no zero mode (there is no signed zero), no rounding flag (a type property), no
  `config.py`, no context managers. v1's mutable global `INFINITY_IS_NOT_FINITE` is deleted

### ieee 1788

* never a runtime mode (closed intervals only, connected only, infinity never attained, decorations
  everywhere — a flag would be `INFINITY_IS_NOT_FINITE` ×10 and multiply the test matrix). instead a
  conformance adapter in the test suite over itf1788 vectors:
    * **input rule**: a 1788 unbounded bound maps to our open-at-inf, because 1788 never attains
      infinity. without this every vector touching infinity mismatches
    * **output rule**: closed-hull **both** our result and the expected value before comparing
      (the input rule would otherwise read 1788's unbounded expected bound as open at inf, while
      our hulled result is closed there); absorbs multi-interval vs connected (`[1,2]/[-1,1]`:
      1788 entire, ours `[-inf,-1] ∪ [1,inf]`, hull = entire → match)
    * residual divergence table: degenerate infinities, domain-clipped functions, decoration
      expectations. (`1/[0]` is not a row: both give empty)
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = the
  community test framework and its `itl` vector DSL. check its licence before vendoring `.itl` files
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
                           size; Builder (collect, sort once, sweep)
        relations.py       TruthSet, pointwise compare, relation predicates, allen()
        applicator.py      op descriptor, corner evaluation, closure pass, rounding hook
        ops.py             add sub mul div reciprocal (pow, functions later) as descriptors
        modulo.py          v3 far-edge mod / divmod / floordiv
        fmt.py             format and parse cut tuples; regexes compiled at module level
        multi_interval.py  the class: immutable cut tuple; _coerce (numbers and intervals only —
                           strings go through an explicit parse()); one-line dunders. arithmetic
                           owns `+ - * / // % **` and unary `- + abs`; set algebra is `| & ^ ~`
                           plus named methods (`difference` has no operator, because `-` is
                           arithmetic subtraction, as in v1)
        time_interval.py   the same kernel over datetime/timedelta values (deferred, D4)
        __init__.py        public API, constants (EMPTY, REALS, ...)
    tests/
        oracles.py         sampling + attainment oracles, promoted out of modulo_v3_prototype
        itf1788/           vector runner, input/output rules, divergence table
        test_<module>.py

* only the two class files know the class; everything below takes and returns tuples. this removes
  the mixin return-type problem, keeps fmt below the class, makes every kernel function
  oracle-testable with tuples, and makes the time-interval port a thin second wrapper
* `_consistency_check` under `if __debug__:` (v1 runs an O(n) scan at the top of nearly every public
  method; that is the real hot cost). correctness lives in the tests
* immutability retires the `inplace=` dual API. `interval.py` (the alternative debug implementation)
  is not ported; the sampling oracle does that job better
* v1 is **archived, never deleted**: all v1 modules move unchanged to `archive/v1/` once v2 is in
  place, and stay as the reference until v2 works

### testing

* soundness fuzz for every op: `op(x, y) ∈ op(A, B)` for sampled `x ∈ A, y ∈ B`
* attainment checks for closure, on int/Fraction operands only
* algebraic properties that pin the cut encoding cheaply: `~~A == A`, De Morgan, the size tiling
  invariants, and for arithmetic:
    * isotonicity for every op: `A ⊆ B ⇒ f(A) ⊆ f(B)`
    * `f(A ∪ B) == f(A) ∪ f(B)` only for ops without split points on finite operands (add, sub, neg,
      abs, mul). for reciprocal/div only `f(A ∪ B) ⊇ f(A) ∪ f(B)` (i.e. isotonicity): under the
      direction-from-the-piece rule equality fails with *any* value of `1/[0]`. counterexample
      `A = [-1, 0)`, `B = [0]`: `1/(A ∪ B)` = `[-inf, -1]`, but `1/A ∪ 1/B` = `(-inf, -1]`
    * `1/(1/A) == A` for every A with no degenerate piece at `0`, `inf` or `-inf`
* the sampler must draw a closed ±inf endpoint with positive probability, and the attainment
  oracle must decide ±inf symbolically; otherwise the infinite-endpoint closure rule is untested
* **interior sharpness**, not just endpoints: soundness fuzz passes on any superset, and endpoint
  attainment checks cannot see a spurious interior point. so check that every gap of the true
  result is a gap of ours — on exact operands, sample points of `op(A, B)` and confirm each is
  attained (pins e.g. `[-1, 1] * [inf]` = `[-inf] ∪ [inf]`, not `[-inf, inf]`)
* sabotage each check once (flip one merge comparison) and watch it go red before trusting it
* itf1788 conformance through the adapter above

### later (not in v2.0)

* a **direction tag on a degenerate zero piece** if a solver ever needs `1/(1/[inf]) == [inf]`:
  metadata that `==` and hash ignore, created only by limits (`1/[±inf]`, `exp([-inf])`), consumed
  only by branch-at-zero functions. never a position in the order — that is what the signed-zero seam
  was
* a decorated wrapper type, with the solver
* `functions.py` (sqrt/log/exp/trig through the applicator), forward-mode autodiff, newton's method as
  a test, numpy compat (array API / `__array_ufunc__`), tight rounding via gmpy2/mpfr

## decision log

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
