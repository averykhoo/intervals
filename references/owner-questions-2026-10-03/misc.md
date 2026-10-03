# owner questions: misc (open items 3b, 9, 5) — 2026-10-03

Agent report, read-only on the repo. Sections appended as written. "verified" = read in the tree or measured by a
probe under `.scratch/owner-questions/probes-misc/` today; "inferred" = my reasoning.

## 1. open item 3 `vectors-ext` (b): worst cases for rootn / pown / fma

### the question and what is built today

HANDOFF row 3 (b): rootn, pown and fma have no CORE-MATH `.wc` file; glibc's `auto-libm-test-out` rows are the
next source, but they are LGPL test data. Built today: `tools/coremath.py` (fetch at a pinned commit, sha256
manifest, `sample`, `check`) and `tests/test_coremath.py` (the gate sample, 24 functions, `DEEP` floors on calls
past 64 bits). The owner's 2026-10-02 principle (HANDOFF row 3): outputs computed by MPFR are not worth vendoring,
the depth is in the CHOICE of inputs.

### facts gathered (2026-10-03, verified unless marked inferred)

* licence: the repo has NO licence of its own (no `LICENSE` at root, no `license` key in `pyproject.toml`;
  plan D15 says so in words: "this repo has no licence of its own"). LGPL-2.1 test data is ALREADY vendored:
  `tests/itf1788/{mpfi,fi_lib,c-xsc}.itl` with `tests/itf1788/COPYING.LESSER` beside them (D15, M13a). the
  wheel ships `intervals/` only (`pyproject.toml [tool.setuptools] packages`), so no test file is distributed.
  So "LGPL test data in this repo" is not a new question: the precedent and the mechanism (licence file beside the
  data, a README naming each file's licence) exist.
* glibc's rows (fetched 2026-10-03 from sourceware HEAD, `math/auto-libm-test-in`, 10,874 lines, LGPL-2.1+): the
  INPUT file has `rootn` 206, `pown` 330, `fma` 263 lines (each line becomes one vector per format x rounding mode in
  the `-out` files; the survey's 434/494/318 count binary64 output rows). the rootn inputs are special values only:
  x in {0, -0, min, -min, min_subnorm, max, 1, 2, -1, -2, 0x1.234p50, 0x1.234p500, 0x9.8765p5000} with degrees
  1..5, 63..16383, 0x7fffffffffffffff, negatives. NOT ONE rootn input has a long random mantissa (grep for
  `0x1.[0-9a-f]{6,}`: 0 hits). pown inputs: +-min, +-max, 1+-ulp, 0x1.0000xp1 ... with small n. fma: 103 of 263
  rows have long mantissas, mostly binary32-shaped (`p-126`, `p-149`, `p+127`) overflow/underflow/double-rounding
  traps for a HARDWARE fma. **these are conformance edge cases, not hard-to-round cases**: glibc's file is hand-written
  and the `-out` values are MPFR's; nothing in it comes from a worst-case search (BaCSeL / Lefevre).
* what each op does in v2 (code read):
  - `fma`: `ops.py::fma` = `add(mul(exact_cuts(a), exact_cuts(b)), exact_cuts(c))` exactly, then
    `float_cuts(result, outward)` rounds ONCE. no libm, no ziv loop. a hard case for fma tests a hardware fma's
    double rounding, which cannot arise here. `tests/test_minmax_fma.py::test_fma_with_floats_rounds_the_exact_result_once`
    already pins the one-rounding property.
  - `pown`: outward (`ops.py::_power_descriptor`), `elementary.exact_pow` exactly then `round_rational`, or past
    `EXACT_POWER_LIMIT` bits `elementary.rounded_pow` (ziv over exp(n ln|x|)), which CORE-MATH's `pow.wc` sample
    already exercises (`tests/coremath/pow.tsv`, `DEEP['pow'] = 865`). **to nearest, a float corner is python's
    `float ** int`, i.e. libm's `pow`, "to nearest but not promised correctly rounded"** (`ops.py::_exact_power_descriptor`
    docstring, a recorded design choice). so an MPFR-exact pown vector run against the NEAREST class would report
    libm's last-bit misses as "failures" that are, by the recorded design, not bugs.
  - `rootn`: `elementary.py::_enclose` sends `cbrt` to `_root_fractions(x, 3, p)` and `rootn` to
    `_root_fractions(x, base, p)`: THE SAME ziv path (verified, lines 397-400). CORE-MATH's `cbrt.wc` is vendored
    (`tests/coremath/cbrt.tsv`, 622 lines, `DEEP['cbrt'] = 150` calls past 64 bits) and was full-checked 2026-10-02
    (106,248 inputs, 0 mismatches), so the enclosure's past-64-bit path that rootn uses is already driven by worst
    cases; what differs per degree is `_div_int(ln, n)`'s n and the n < 0 negation (`_exact_rootn` handles perfect
    powers). `tests/test_functions.py::test_cbrt_is_rootn_3` pins the identity at the set level.
* making our own hard cases by random filtering is infeasible: probe `probes-misc/rootn_hard_probe.py`
  (wraps `elementary._ziv` as §3g did), 2026-10-03, pure path: degrees 2, 3, 5, 7, -3 x 120,000 inputs x DOWN/UP =
  1,200,000 calls, final p = 64 in EVERY call (0.15-0.25 ms/call). `_root_fractions` works at p + 16 bits, so a
  random double is undecided at the first precision with probability well under 1e-6. real worst cases need a
  lattice search (Lefevre/BaCSeL), a project of its own.
* other inputs-only sources for rootn that the survey found: Lefevre's `hrcases-powint` (x^(1/n) for small n, 37,399
  lines, §3c) has NO licence anywhere (§3f, searched 2026-10-02); the IA.jl-derived `intervalarithmeticjl.itl` has 29
  rootn vectors (MIT, §1b), set-level conformance rows, not hard cases.

### options

**A. vendor glibc's rows** (inputs, or inputs + MPFR outputs) into `tests/glibc/` with a `COPYING.LESSER`.
* means: ~800 binary64 lines for the three ops, a parser for `= rootn downward binary64 <x> <n> : <y> : flags`
  (one regex), a test like `test_coremath.py`, a licence file and a README row per the D15 pattern.
* pros: zero network at test time; LGPL-2.1 test data already sits in `tests/itf1788/`, so no new licence class;
  covers 1788 edge inputs (min, max, subnormal, huge degree) in three rounding modes.
* cons: it buys conformance at special values, which `tests/test_functions.py::test_rootn_domain_and_poles`,
  `test_extreme_floats_functions.py` (rootn is in `CASES`) and the itf1788 rootn vectors already give; it contradicts
  the 2026-10-02 principle (nothing here is a hard-to-round input); for pown to nearest it would surface libm's
  non-correct rounding as red rows needing a divergence category; for fma it tests a property our implementation
  cannot violate. a repo with no licence of its own accumulating a third licence class for ~800 rows is a cost
  with nothing on the other side.
* when better: if the owner wanted an INDEPENDENT third-party expected value at IEEE edge inputs (MPFR's, not
  ours) for the three ops in directed modes, or if the repo later adopts an LGPL-compatible licence anyway and the
  concern evaporates.

**B. fetch at test time** (the `tools/coremath.py fetch` shape: pinned glibc commit, sha256, cache under
`.scratch/`), nothing vendored.
* means: a second fetch tool or a `--source glibc` branch in `tools/coremath.py`, a manual `check` only.
* pros: no licence text in the tree at all (nothing is distributed); the same manual-only workflow as CORE-MATH.
* cons: all of A's "what does it catch" problems, plus the tool work; the data is ~800 rows, so the fetch
  machinery is heavier than the data. a cached LGPL file in `.scratch/` is no cleaner legally than a vendored one
  (it is not distributed either way: the wheel ships `intervals/` only).
* when better: only if the owner decides LGPL text must never enter the tree, and still wants A's rows.

**C. generate our own hard cases** (an MPFR/arb search).
* means: for rootn, a lattice-based worst-case search per degree (Lefevre's method) or BaCSeL in WSL; for pown
  and fma, nothing to search for (exact paths).
* pros: our own data, no licence; would be the only true worst cases for rootn n != 3.
* cons: the random version is measured infeasible (0 of 1.2M calls past 64 bits); the lattice version is a
  multi-day project; and the payoff is low because `cbrt.wc` already drives the shared `_root_fractions` path
  past 64 bits (what §3h showed worst cases are for).
* when better: if a degree-specific bug in `_root_fractions` is ever suspected (n large, n < 0), or if the
  library grows a rootn path that is not `_root_fractions`.

**D. skip** (close (b) as "no file, none needed", recorded with the reason).
* means: a line in the plan's vectors-ext record: fma exact-then-rounded (no hard case can exist), pown exact or
  `rounded_pow` (covered by `pow.wc`; nearest is libm by design), rootn shares `cbrt.wc`'s path.
* pros: nothing to maintain, no third licence class, consistent with the 2026-10-02 principle.
* cons: rootn's n != 3 degrees are only random- and property-tested past the shared path; the per-degree
  `_div_int(ln, n)` step is covered by `tests/test_elementary.py::test_rootn_correctly_rounded` (decimal oracle,
  9 degrees) and `tests/test_oracle_flint.py::test_rootn_against_arb`, not by hard cases.
* when better: the default, given what the code does.

**E. (not in the plan) a small structural rootn set, no data file**: a parametrized test that walks
`tests/coremath/cbrt.tsv`'s hard inputs through `rooted` degrees 3 and -3 (the negation path) and checks DOWN/UP
against MPFR via `tools/coremath._oracle()`, plus the IA.jl 29 rootn vectors (MIT) under (c)'s pull.
* pros: an hour's work, no licence, exercises exactly the two rootn-specific branches (`n < 0`, `x < 0`) with
  inputs known to be hard for the shared path.
* cons: still not hard cases for n other than 3 (cube-root hardness does not transfer to n = 5).
* when better: if the owner wants (b) closed with a positive artifact rather than a reason.

### recommendation

**D, with E's first half if a positive artifact is wanted** (confidence: high for fma and pown, medium-high for
rootn). The glibc file is the wrong kind of data for this item: it is conformance input for a C libm, not a
hard-to-round set, and two of the three ops have no rounding path a hard case could stress. Rootn's one such path
is `cbrt`'s, already covered. Record the closure in the vectors-ext row with the three one-line reasons above, and
add the IA.jl 29 rootn rows under (c) if (c) is done.

What would change my mind: (1) the owner deciding the NEAREST class's pown should be correctly rounded (then the
libm `float ** int` in `_exact_power_descriptor` is the change, and CORE-MATH's `pow.wc` through `rounded_pow`
to nearest is the test, still not glibc); (2) a rootn bug found for some n != 3 that `cbrt.wc` could not have
caught, which would make C's lattice search worth a session.

Cost of changing later: nil either way, this is test data, not API; A/B/C/E can be added after 2.0 with no
user-visible effect. The only irreversible-ish move is adding a licence file to the tree, and even that is a
`git rm`.

## 2. open item 9 `Q6-rest`: random_multi_interval and public apply()

### the question and what is built today (verified)

* v1 `archive/v1/multi_interval.py::random_multi_interval(start, end, n, decimals=2, prob_neg_inf, prob_pos_inf)`:
  a test-data generator on the module-global `random` (not seedable per call), 20 % degenerate pieces, the four
  openness mixes. its only callers were v1's own `__main__` demo loop (line 2005) and a manual stress loop in
  `archive/v1/interval.py` (lines 716-723). it was never a documented user feature (v1 README: no mention).
* v2 has, in the test tree only: `tests/strategies.py::cut_tuples` / `::piece_pairs` (hypothesis, over cut tuples,
  with a shared value pool so coinciding ends are common), `tests/test_backend.py::multi_intervals(cls, values,
  max_pieces)` (hypothesis, builds objects of either class), and ad hoc `random.Random(seed)` loops in the exhaustive
  harnesses. plan §4 row: "the tests use hypothesis strategies instead".
* v1 `apply_monotonic_unary_function(func)` / `apply_monotonic_binary_function(func, other)`: apply ANY python callable
  endpoint-wise, assuming monotone, no rounding control; v1 built every arithmetic op on them (`+`, `*`, `<<`, `exp`,
  `round`...). v2 replaced them with `applicator.apply_unary(desc, cuts)` / `apply_binary` driven by an
  `applicator.OpDescriptor(name, fn, monotone, split_points, attained, rounded, pole)` (shape-then-attainment, the
  rounding hook for float corners, the pole rule), and `functions.apply(name, cuts, outward, base)` for the named
  elementary functions (`functions.NAMES` + rootn). none is exported: `tests/test_applicator.py::test_package_exports_unchanged`
  asserts `intervals` has no `apply`, `apply_binary`, `OpDescriptor` attribute; plan §4 row: "`applicator` and
  `OpDescriptor` are not exported". python has no privacy, so `from intervals.applicator import apply_unary` works
  today; what is withheld is the stability promise.

### random_multi_interval: options

**R1. do not port; record as gone** (plan §4 row becomes "gone; the tests' hypothesis strategies replaced it").
* pros: nothing in the public surface that the library itself does not use; hypothesis shrinking and replay are
  what made the v2 test suite find its bugs (the fuzz rows in plan §2), a `random.Random` generator has neither.
* cons: a user writing their own property tests has to write a strategy (ten lines, see `test_backend.py::multi_intervals`).
* when better: the default for a 2.0 with "zero users" (D17) — nothing added that would have to be kept.

**R2. port as a plain function** (`intervals.testing.random_multi_interval(rng, lo, hi, pieces, ...)`, stdlib `random`).
* pros: no optional dependency; a quick way to make examples in a REPL or a notebook.
* cons: a second generator to keep in step with the type (end types int/Fraction/float, both classes, mixed-type
  points); its parameters (`decimals`, `prob_neg_inf`) are v1's, shaped by v1's float-only ends; v1's version is not
  seedable, so a faithful port would need redesign anyway.
* when better: if the owner wants `MultiInterval` demos/benchmarks without hypothesis, e.g. for a README example or
  `tools/backend_speed.py`-style timing (today that tool makes its own operands).

**R3. ship a hypothesis strategy** (`intervals.testing.multi_intervals(cls=MultiInterval, values=..., max_pieces=...)`,
importing `hypothesis` lazily so the package has no new runtime dependency).
* pros: the shape users of a property-tested library actually want (what `hypothesis.extra.numpy` is to numpy); the
  test tree's own strategy would move into the package and be tested by the suite for free; shrinking, replay,
  `@example` pins come with it.
* cons: an optional-dependency module in the wheel and a public strategy whose VALUE DISTRIBUTION becomes something
  users depend on (changing the pool changes what their tests find); the test tree's strategies are tuned for the
  library's own bug-hunting (small pool, coinciding ends), which is also what a user wants, so the tuning would be
  frozen.
* when better: once there is an external user writing property tests against the library — an additive change
  then, with a real request shaping the signature.

**R4. a classmethod `MultiInterval.random(rng, ...)`.** same as R2 but on the type; cons: a testing concern on the
core class, and the class's docstring surface is already large. not recommended.

**recommendation: R1 now, R3 later if asked** (confidence: high). Adding is cheap after 2.0 (purely additive);
removing after 2.0 is a break. Nothing in the repo calls for it (the tests have better), and v1 never documented it.
What would change my mind: a concrete non-test caller inside the repo (a benchmark, a README demo) — then R2 as a
`tools/` helper, still not public.

### public `apply()`: options

**P1. none in 2.0** (status quo; the plan §4 row records "not public; `functions.apply` and the applicator are
internal, importable by path").
* pros: the library's one promise — "the set of values attained", exactly on exact ends or rounded in a declared
  direction (the `OutwardMultiInterval` docstring: "every result holds the exact result of its operands") — cannot
  be kept for an arbitrary user callable: a user's `fn` gives neither exact arithmetic nor directed rounding, nor
  the pole/limit rules. a public apply would be the one operation whose result is unsound by construction, in a
  class whose other 80 operations are sound.
* cons: a user with a function the library lacks (erf, gamma, a tariff curve, a lookup table) has no supported
  entry point and must compose from existing ops or go to the internal module.
* when better: for 2.0 as it stands; nothing in `README.md`, the plan or the solver stack needs it.

**P2. export the applicator as a power-user API** (`apply_unary`, `apply_binary`, `OpDescriptor` in `__all__`).
* pros: the most general thing; a user can give `monotone`, `split_points`, `rounded=(down, up)` and `pole` and
  get the same shape-then-attainment treatment the library's own ops get, outward class included.
* cons: `OpDescriptor`'s seven fields and the applicator's conventions (`_ends` reading a mixed point by its low cut,
  Q20's crossed-piece rule, the face rule for attainment, `rounded` seeing only finite float corners) become API
  and freeze the applicator's internals right when the fuzz is still changing them (plan §2 fuzz-mixed-points,
  fuzz-rootn-crossed, both this week); the soundness burden lands on the user with no check.
* when better: never before the applicator stops moving; after 2.0, only if a second library wants to build on it.

**P3. a narrow public method** (`MultiInterval.apply(fn, *, split_points=(), rounded=None)`: `fn` continuous on
each piece between the split points, no monotonicity needed, the shape found by the applicator; without `rounded`
the outward class refuses with `TypeError`, the nearest class takes `fn`'s floats as they are).
* pros: the useful 80 % of P2 with one callable and two keywords; the outward refusal keeps that class honest;
  the nearest class already accepts libm's last bit elsewhere (`ops.py::_exact_power_descriptor`: pown to nearest
  is python's `float ** int`), so "fn's float is the value" is consistent with it.
* cons: still a second kind of promise ("the set `fn` attains if `fn` is what you say it is"), to document and
  to defend; shape-then-attainment needs the extrema, which for a non-monotone `fn` means the user also supplies
  the critical points — at which point the user is writing an `OpDescriptor` anyway.
* when better: after 2.0, if users ask for user-defined functions; a `Dual`-based variant (monotonicity checked by
  the derivative's sign over the piece) could make it sound for C¹ functions, which is a design of its own.

**P4. v1's endpoint-wise map** (`map_ends(fn)`, monotone assumed, no soundness).
* pros: trivial; what v1 had.
* cons: silently wrong on any non-monotone `fn` and on every float rounding; the exact failure mode the v2
  rewrite exists to remove (plan "shape-then-attainment — the modulo v3 lesson").
* when better: never in this library.

**recommendation: P1 for 2.0** (confidence: medium-high). Keep `functions.apply` and the applicator importable by
path, say so in the plan row, and revisit as P3 when a caller exists. The asymmetry decides it: adding P3 after
2.0 costs nothing; exporting `OpDescriptor` now and changing it later is a break of exactly the kind the fuzz has
been forcing weekly. What would change my mind: a named user function wanted in a demo or the solver (then build it
as a named function in `functions.py`, which stays sound, rather than open the generic door).

### cost of changing later (both)

Before 2.0: free. After 2.0: adding either is additive (no cost); removing or reshaping either is a break.
That favours not shipping them now.

## 3. open item 5 `Q6-shift`: semantics of << and >>

### the question and what is built today (verified)

Decided: port `<<` and `>>` (owner 2026-09-26, "for sure"). Open: the meaning on real sets. v1
(`archive/v1/multi_interval.py::__lshift__`, 1772-1782) applied `operator.lshift`/`rshift` endpoint-wise through
`apply_monotonic_binary_function`, so: int ends only (a float end raised TypeError), the shift count could be a
`MultiInterval` (Cartesian corners), `>>` floored like python's int, and `__rlshift__`/`__rrshift__` existed.
v2 has no shift today; `tests/test_numpy_compat.py` derives the reflected dunders of our classes (lines 142-146),
so a new `__rlshift__` joins that test the day it is written and is red until `numpy_compat._OPERATORS` gains
`left_shift`/`right_shift` (HANDOFF row 5).

Python's own rules (probe `probes-misc/shift_probe.py`, 2026-10-03): `3 >> 1 == 1`, `-3 >> 1 == -2` (floor),
`1 << -1` raises `ValueError: negative shift count`, `1 << True == 2`, and `Fraction(3) >> 1`, `3.0 << 1`,
`1 << 2.0` all raise `TypeError`; numpy: `np.int64(3) << 1` works, `np.float64(3.0) << 1` and
`np.left_shift(3.0, 1)` raise `TypeError`. So python defines shifts for INTEGERS only; for reals it defines nothing,
it refuses. There is no python precedent to be consistent with on Fraction or float ends.

What the existing multiplication already does (same probe), which is what `A * 2**n` would inherit:
`M(3) * Fraction(1, 2)` = `[3/2]`; `M(3.0) * Fraction(1, 2)` = `[1.5]`; subnormal `M(5e-324) * Fraction(1, 2)` =
`[0.0]` to nearest and `(0.0, 5e-324)` outward; overflow `M(1e308) * 2 ** 10` = `[inf]` to nearest and
`(1.7976931348623157e+308, inf)` outward; `M(-inf, 3) * 4` = `[-inf, 12]`; `M(-3, 7, end_closed=False) *
Fraction(1, 4)` = `[-3/4, 7/4)` (openness kept). And `M(3) // 2` = `[1]`, `M(-3) // 2` = `[-2]`: the floor reading
of `>>` already has a spelling. `exp2` exists (`M(1, 3).exp2()` = `[2, 8]`), so `A * 2 ** B` for an interval B is
one expression today (`M(1, 2) * 2 ** M(1, 3)` = `[2, 16]`).

### options for `A << n`

**L1. `A << n` = `A * 2**n`, n an int** (any `Integral` but bool, as `functions._check_degree` and `ops.power`
take degrees: numpy ints included via `int(n)`).
* means: `ops.mul(a, 2 ** n)` (a `Fraction(1, 2 ** -n)` for n < 0), so shape-then-attainment, the rounding hook,
  the outward class, the warnings and numpy dispatch are all reused; exact on int and Fraction ends, exact on float
  ends except at overflow (→ inf / `(MAX, inf)`) and in the subnormal range (bits lost, rounded per class).
* pros: the natural real-set meaning (binary scaling: `math.ldexp`, MPFR `mul_2si`, arb `mul_2exp_si`, decimal
  `scaleb` are all this); no new semantics to prove; the one op in the library guaranteed exact on float ends
  almost everywhere, which is what a numerics user shifts for (bisection by powers of two, scaling a problem).
* cons: on int ends it is just `* 2**n`; a user expecting python's bit semantics (ValueError on a negative count)
  gets a value instead.
* when better: for a library of real sets, always; this is also the reading the owner wrote down (`v2-plan.md`
  "2026-09-26 revision: owner answers", Q6).

**L2. v1's reading: python's int shift endpoint-wise, floats raise.**
* pros: exact python mirror on integer sets.
* cons: the only operator in the class that refuses float ends; endpoint-wise application is v1's unsound pattern
  (plan "shape-then-attainment"); for Fraction ends python has no shift at all, so v1 raised there too.
* when better: only if the shifts are meant as bit manipulation on integer sets, which nothing in the plan suggests.

Sub-choices under L1:
* **negative n**: allow (`A << -n == A >> n`, the ldexp reading) or raise `ValueError` as python does. recommend
  **allow**: with a real-set meaning there is nothing undefined about it, and the identity makes `>>` and `<<` one
  op with a sign, which is how every arbitrary-precision library spells it. (python raises because an int shift by a
  negative count would be a floor division, a different op; here it is the same op.) confidence medium: pure taste,
  low stakes, decide before release.
* **an interval shift count `A << B`**: refuse (`_coerce` returns NotImplemented for a `MultiInterval` count →
  TypeError), because it has two readings (`A * 2**B` over the reals, or `A * {2**k : k in B ∩ Z}`), and the first
  is spelled `A * 2 ** B` already. v1 allowed it; nothing used it.
* **reflected `n << A`**: refuse for the same reason (the shift count would be a set). then `np.int64(3) << A` and
  `3 << A` both raise TypeError, so `test_numpy_scalar_operators_are_python_numbers` sees the same outcome either
  way (inferred from the test's `assert_same_outcome` shape; the implementing session checks it).
* **huge n**: `2 ** n` is built as an int; `A << 10 ** 7` is a 10-million-bit Fraction scalar, slow but finite, and
  python's own `1 << 10 ** 7` is the same object. no cap needed (pown's cap exists because `x ** n` of a float base
  grows in n times the base's bits; here the base is 2).
* **bool count**: refuse (python allows `1 << True`; the class refuses bool everywhere else: `_coerce`,
  `__pow__`, `_check_degree`). consistency with the class over consistency with python here.
* **the other types**: `DecoratedInterval` (shift is `com`-preserving: defined, continuous, bounded iff the input
  is) and `Dual` (d(2^n x) = 2^n dx) should get the same dunders so the numpy test's class loop stays uniform;
  `ieee1788.Interval` follows its own rule (row 6).

### options for `A >> n`

**R1. exact: `A >> n` = `A * 2**-n` = `A / 2**n`**, the inverse of `<<` (`A >> n == A << -n`).
* means: the same `ops.mul` call with `Fraction(1, 2 ** n)`; `[3] >> 1` = `[3/2]`; `[3.0] >> 1` = `[1.5]`;
  `[5e-324] >> 1` = `[0.0]` to nearest, `(0.0, 5e-324)` outward; `[-3, 7) >> 2` = `[-3/4, 7/4)`.
* pros: one operation with a sign (ldexp), an involution with `<<` on exact ends, exact on float ends outside the
  subnormal range; consistent with the class's own precedent that `/` is exact (`MultiInterval(1) / 3` = `[1/3]`,
  README line 16 "int and Fraction stay exact") while `//` is the floor; `Fraction` and `float` have no `>>`, so
  there is no python floor to be inconsistent with on real ends.
* cons: on int ends it departs from python's `3 >> 1 == 1`; a user porting int code would get `[3/2]`. the same
  departure the class already makes for `/` on ints (python's int `/` gives a float; the class gives a Fraction),
  so the precedent is "real-set semantics win", and the result is visibly a Fraction, not a silently different int.
* when better: for a real-set library, always: it is the only reading that adds an operation (exact binary scaling).

**R2. floor: `A >> n` = `floor(A / 2**n)` = `A // 2**n`.**
* means: `modulo.floordiv(a, 2 ** n)`: an integer set, enumerated, with `modulo.FLOOR_ENUMERATION_CAP` and its
  `HullWarning` past the cap; `[3] >> 1` = `[1]`, `[-3] >> 1` = `[-2]`, `[0, 10) >> 2` = `{[0], [1], [2]}`.
* pros: python's int `>>` exactly, on int ends.
* cons: it duplicates an operator that exists (`A // 2**n`), so it adds no operation; it breaks `A >> n == A << -n`
  and `(A << n) >> n == A`; it turns a float piece into an enumerated integer set, which for a numerics user is a
  surprise (`[0.1, 0.9] >> 1` would be `[0]`, not `[0.05, 0.45]`); a step function dressed as a scaling, next to
  `<<` which is a scaling: the two operators would not be inverses even on exact ends, where python's are "almost"
  (`(x << n) >> n == x` holds for ints in python; with R2 it holds too, but `(x >> n) << n != x`, as in python).
* when better: if the shifts are bit operations on integer sets (L2's world); then also raise on floats and on a
  negative count, i.e. port v1 as is.

**R3. type-dependent** (floor on int ends, exact on Fraction/float ends). rejected: one operator, two meanings,
and a mixed-type piece (`[2, 2.0]`, the fuzz-mixed-points row) would have to pick per end.

### recommendation

**L1 + R1** (confidence: medium-high): `A << n` is `A * 2**n` and `A >> n` is `A * 2**-n`, exactly, for an int n
of either sign (not bool; numpy ints accepted; a `MultiInterval` count and the reflected forms refuse with
TypeError); float ends round only where a double cannot hold the exact value (subnormal range, overflow), to nearest
in `MultiInterval` and outward in `OutwardMultiInterval`, through the existing `ops.mul`; `A // 2**n` stays the
floor spelling. Document the one departure from python (`[3] >> 1` is `[3/2]`, as `[1] / 3` is `[1/3]`) in the
dunder's docstring and the README's "int and Fraction stay exact" line.

What would change my mind: the owner's intended use being integer bit arithmetic on int sets (then port v1: L2 +
R2, floats raise, negative counts raise). Nothing in the plan or the README suggests it; the owner's own note
(`v2-plan.md` Q6) already frames `<<` as `A * 2**n`.

### cost of changing later

Before 2.0: free, nothing exists yet. After 2.0: an operator's semantics is the costliest kind of change (results
differ silently on every odd int end for R1 vs R2, and on every float end for L1 vs L2), so this must be fixed
before the release; adding the refused forms (interval count, reflected) later is additive and cheap, so refusing
them now keeps options open.

## probes (2026-10-03)

* `.scratch/owner-questions/probes-misc/shift_probe.py`: python/numpy shift rules and the class's `*`, `/`, `//`
  on the cases above (run with `PYTHONPATH=.`).
* `.scratch/owner-questions/probes-misc/rootn_hard_probe.py <N>`: wraps `elementary._ziv`; N = 2000 and 120000
  per degree, degrees 2, 3, 5, 7, -3: 0 of 20,000 and 0 of 1,200,000 calls past 64 bits.
* `.scratch/owner-questions/probes-misc/auto-libm-test-in.txt`: glibc HEAD's input file as fetched (LGPL-2.1+,
  283 KB, kept only as the evidence for the counts above; delete with the run's directory).
