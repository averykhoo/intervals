# `MultiInterval` v2 plan

* the range will be the affine extended real numbers, meaning support for ±inf along with negative zero, i.e.:
  `[-inf] + (-inf, 0) + [-0, 0] + (0, inf) + [inf]`
* divide by zero is supported, and there will be warnings for indeterminism
* newton's method solver as a test
* generalized function applicator as long as its continuous and differentiable
* forward mode autodiff
* numpy compat via https://data-apis.org/array-api/latest/ or `__array_ufunc__` or https://numpy.org/doc/stable/user/basics.interoperability.html
* consider ieee 1788 compatible decorations, although multi intervals actually support a bit more so not sure if it matters

## negative zero

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

## update

found that there's an ieee spec that does something similar

## v2 consolidated decisions (2026-08-16)

supersedes the enum discussion above (kept for history). derivation: chat sessions 2026-08-15/16.

### representation: cuts (boundaries), not endpoints

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

### signed zero: keep it

* -0 is direction info on a zero boundary, not a user-facing number. IEEE hands us the sign bit anyway (`-1.0 * 0.0 == -0.0`), and it's exactly what keeps reciprocal sharp under our closure-of-limits semantics (`1/[-5,-0] = [-inf, -0.2]`)
* zero has *three* boundaries instead of two: `(0,-1)` below -0, `(0,0)` between -0 and +0, `(0,+1)` above +0. side 0 is only legal at value 0 (one constructor check)
* the merge rules from the section above now fall out of plain comparison, nothing written down:
    * `(..., 0) | [-0]` -> merges -> `(..., -0]`
    * `[0] | (-0, ...)` -> merges -> `[0, ...)` (in cuts `(-0, ...)` *is* `[+0, ...)` — same boundary, the "feels wrong" state never exists)
    * `[-0] | (0, ...)` -> does NOT merge (+0 genuinely missing)
    * `[-0] | [+0]` -> merges -> the full zero
    * deviation from the old plan: `(..., -0) | [0]` does NOT merge (-0 genuinely missing). this was the case marked "somewhat questionable" above — merging it is what manufactured the spurious -0 and the phantom -inf under reciprocal
* the seam at zero is irreducible: -0 and +0 are two *adjacent* points in an otherwise dense order — an order-theoretic fact no encoding can hide. its entire footprint is one extra token at value 0. dropping -0 entirely is the only way to a perfectly homogeneous line (and would cost sharp division)

### zero gluing: float semantics by default

model the UX on how python/IEEE already treat -0.0 (equal, same hash, sign preserved through arithmetic, visible only via copysign / division):

* `zero_mode='glued'` (default): `==`/`hash`/membership/merging treat -0 == +0; sign tags on zero endpoints are preserved until an actual cross-zero merge consumes them, so `1/[-5,-0]` stays sharp even in glued mode. gluing only ever *widens* by the other zero point -> sound, never wrong
* `zero_mode='strict'`: no gluing; `[-0] | (0,...)` stays two pieces; for solvers and branch cuts (1/x, log, sqrt). toggle via context manager (same pattern as the warnings design)
* repr always faithful (prints -0 when present, like python floats); str likewise

### comparisons

* `< <= > >=` return a tri-state truth set (TRUE / FALSE / BOTH, since "every a op every b" can be both); `__bool__` raises on BOTH (numpy precedent), so `if a < b:` either works or fails loudly — never guesses
* `==` and `__hash__` are structural set equality (required for dicts/tests); pointwise equality is a method, and is BOTH for any non-degenerate `a == a` (document this, it surprises everyone)
* explicit `sort_key` (lex on cuts) for structural ordering; `sorted()` raising on ambiguous intervals is a feature
* relation vocabulary (the adjacent/adjoining/intersecting/overlapping todo): named set-level predicates — disjoint, adjoins (end cut == start cut), overlaps, contains, within, equals, before (sup A < inf B), after — plus certainly_/possibly_ modal variants
* allen's 13 relations are only JEPD for *contiguous* intervals; expose `allen(a, b)` restricted to contiguous pieces (raise otherwise, or caller passes hulls explicitly). optional: per-piece relation matrix, or the set of relations holding between any piece pair (allen's algebra natively reasons over relation sets). note cuts make the taxonomy *finer* than classical allen: tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing-one-point (`[1,2] ∩ [2,3] = {2}`) are distinguishable

### division semantics vs ieee 1788 (they are not wrong, just different)

* ours: closure over attainable values/limits (cset-flavored, cf. hickey/van emden paper in README todo) -> zero denominators contribute their limit infinities, signed zeros pick the branch
* 1788: division is the *inverse relation of multiplication* over the reals ({z : x = z·y}); y=0 contributes no z because z·0=1 has no solution -> `1/[-5,0] = [-inf,-0.2]` with decoration dropped to `trv` (partiality is recorded, not ignored). solvers get the split result via `mulRevToPair` (two-interval extended division)
* consequence: itf1788 vectors will disagree on division-by-zero cases *by design*; keep a documented divergence table rather than chasing exact matches
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = community test framework for it (test files in a small DSL, "itl"; used by IntervalArithmetic.jl et al)

### housekeeping

* rename `cardinality` -> `measure`: it's a lex-graded size (rays, open length, closed endpoint count), not cardinality. the ω/ε gloss is fine intuition; we only compare these tuples, never do arithmetic on them
* delete `INFINITY_IS_NOT_FINITE`: affine extended reals, `[inf]` degenerate allowed; cuts encode `[a, inf]` vs `[a, inf)` naturally, and a mutable global that changes set semantics is a footgun
* v2 core type immutable + hashable; incremental building via a small builder (bisect-insert is 82x faster than re-sort per compare.py)
* directed rounding moves FIRST, before newton/autodiff: `math.nextafter` outward rounding behind a rounding-policy hook, gmpy2/mpfr later as tight mode. every arithmetic op touches endpoint computation — retrofitting means touching everything twice. (libm is not correctly rounded, so ±1 ulp around trig is pragmatic-not-rigorous; document)
* floordiv: enumerate integer points below a size cap, else return hull + warning — never silently drop openness
* testing: random fuzz vs sampling oracle for soundness (`op(x,y) ∈ op(A,B)` for sampled x,y) + attainment checks for closure — the pattern already validated by the modulo v3 work; itf1788 as conformance suite with the divergence table above

### ieee 1788 conformance: test adapter, not a runtime flag

* 1788 is further away than a flag: closed intervals only (no open bounds exist in the standard), connected only (no multi-intervals — everything hulls), ±inf never attained, decorations everywhere. a semantics flag would be `INFINITY_IS_NOT_FINITE` again ×10, and every flag multiplies the test matrix (`zero_mode × 1788_mode × rounding`)
* instead: a conformance adapter in the *test suite* — parse itf1788 vectors, run our ops, closed-hull the result, compare, consult a small divergence table. hulling absorbs most divergence automatically (`[1,2]/[-1,1]`: 1788 says entire, we say `[-inf,-1] ∪ [1,inf]`, hull = entire → match). residual table: empty-vs-degenerate-infinity division cases, domain-clipped functions, decoration expectations
* principle: flags change what existing objects mean; wrapper types add meanings. if real conformance is ever needed, it's a thin wrapper class in `ieee1788.py`, never a mode on MultiInterval

### package architecture

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

### generic applicator: keep it, but shape-then-attainment

* keep the generic endpoint applicator — not to save code, but because it's the ONE place where directed rounding, closure decisions, and domain splitting get woven into endpoint computation. per-op code would reimplement all three per op, forever. 4-vs-2 evals is noise next to interpreter overhead
* v1's epsilon propagation through corners is UNSOUND, not just inelegant: `[0,1] * (2,3)` — min corner `0*2=0` gets eps `0 or 1` = open, v1 returns `(0,3)`, but 0 is attained (`0 * 2.5`); true result `[0,3)`. the flat spot at zero makes the result independent of the open operand. same disease the modulo v3 work diagnosed (604 closure faults from epsilon propagation), here it excludes attainable values → containment violation
* restructure around the modulo lesson:
    1. locations first, all endpoints treated closed (corner min/max — correct for coordinatewise-monotone and bilinear ops)
    2. closure per endpoint separately: strictly-monotone ops → corner-flag rule is provably fine (fast path, zero cost); flat-spot ops (mul at 0, pow, min/max-like) → attainment check
    3. driven by a small op descriptor per operation: monotonicity directions (gives add/sub a 2-corner fast path for free), flat-spot predicate, rounded-eval pair from `rounding.py`

### more v1 retirements

* empty operands in arithmetic propagate: `A / ∅ = ∅` (vacuous union over no divisors), same for `∅ / B`, `A + ∅`, etc. — this matches ieee 1788 (empty in → empty out) and the other interval libraries, and mirrors nan propagation in floats, so silent propagation is the standard-aligned default. **note (owner):** silent empty propagation is effectively implicit nan/null propagation and can be annoying to debug later — so emit a dedicated `EmptySetPropagationWarning` on division (maybe all arithmetic) with a default `'ignore'` filter installed at import; solver code opts into `warnings.simplefilter('error', EmptySetPropagationWarning)` as a tripwire. exceptions stay reserved for malformed construction (`[2,1]`)
* parsing moves out of `merge()` into `fmt.py`, regexes compiled at module level (hot-path compile in v1)
* one `_coerce(other)` in core instead of per-method isinstance ladders
* the `inplace=` dual API disappears with immutability — large chunk of v1 surface gone for free
* retire `interval.py` (the alternative debug implementation) — the sampling oracle does that job better, and a second implementation is a maintenance tax
* keep `__getitem__` slicing (`x[0:5]` as restriction); `in` = scalar membership + documented subset alias; subset stays a named method since `<=` is the tri-state comparator
* `__bool__` = non-empty, explicitly (set precedent) — distinct from tri-state comparisons, which raise on ambiguity
