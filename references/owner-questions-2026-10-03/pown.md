# Q17 / m14b-open (4300 digits) / Q18: pown of exact operands, huge-int repr, correctly rounded nearest pown

advisory report, 2026-10-03, read-only on the tree at `679bb2f`. probes under
`.scratch/owner-questions/probes-pown/` (each run with a hard subprocess timeout). "verified" = run
by probe today; "inferred" = read from the code/plan, not run.

## 0. what is built today (common to all three)

tree at `679bb2f`, python 3.13.15 (`sys.get_int_max_str_digits()` = 4300). probe logs: `probes-pown/p1.log`
(what runs), `p2.log` (exact power growth), `p3.log` (int-str limit), `p4*.log`, `p7.log` (Q18), `p5.log` (cost),
`p6.log` (warning messages).

**pown's three routes** (`intervals/ops.py::_power_descriptor`, `::_exact_power_descriptor`; read, and the
plan record §2 "pown-huge"):

| corner | outward class | nearest class |
|---|---|---|
| float, `|n| * bits(x) <= EXACT_POWER_LIMIT` (100000) | `elementary.exact_pow` once (`float_exact`, cached), `round_rational` down/up | python's `float ** int` = **libm `pow`** while `|n| <= 2 ** 53` |
| float, past the limit | `elementary.rounded_pow` each way (1788's pow route), attainment against `ops._NOT_A_DOUBLE` | `rounded_pow(.., NEAREST)` (past `2 ** 53` only) |
| int / Fraction, any n | `base.fn`: python's `x ** k`, **unbounded** | same, unbounded |

**pow (`functions.pow_`, `::_power_corner`) already does Q17's (c) for exact operands**: `exact_pow` while the
bound `|n| * max(bits(num), bits(den)) <= EXACT_POWER_LIMIT`, else `rounded_pow` in the directed
direction (`direction = want` when no operand is a float, so the tightest open float enclosure).
verified (p1, 2026-10-03): `M(2) ** M(50000)` is the exact 50001-bit int, `M(2) ** M(50001)` is
`(1.7976931348623157e+308, inf)`; `M(3) ** 70000` is the exact 110948-bit int in 2 ms while `M(3) ** M(70000)` is
`(MAX, inf)`; `M(Fraction(1, 2)) ** M(2 ** 60)` is `(0.0, 5e-324)`; `O(2) ** O(2 ** 60)` is `(MAX, inf)`.

**a third limit, inconsistent with both**: `exp2`/`exp10` of an exact int build exactly while the *exponent*
is at most `EXACT_POWER_LIMIT` (`elementary._exact`, line 550: `abs(x) > EXACT_POWER_LIMIT`), not the result's
bits. verified: `M(100000).exp10()` builds a 332193-bit int (8 ms) while `M(10) ** M(25001)` (bound 100004
bits, true size 83052) is `(MAX, inf)`. so today there are three thresholds for "an exact power is built":
pown none, pow 100000 bits (as a bound), exp2/exp10 100000 as an exponent.

**the proof floor**: `ops._check_marker_premises` refuses at import an `EXACT_POWER_LIMIT` below 36550 bits.
it is a floor only; raising the limit costs the proof nothing (read). the proof in `ops._NotADouble`'s
docstring is stated for "|x| = num/den in lowest terms" and uses only that the denominator is a power
of two *or not*; its three cases (x = ±1; a power of two; an odd factor >= 3) cover every rational, so
it extends as written to int and Fraction corners (inferred from the docstring, not re-derived
formally).

**hangs verified today (8 s timeout each, p1)**: `O(0.5, 2) ** 2 ** 40` (the 2 an int), `M(2) ** 2 ** 60`,
`Fraction(1, 3) ** 2 ** 40`, and plain python's own `2 ** 2 ** 60` and `2 ** 10 ** 300` (CPython raises
nothing; it squares until memory runs out). the float forms answer at once: `M(2.0) ** 2 ** 60` = `[inf]`,
`O(0.5, 2.0) ** 2 ** 40` = `(0.0, inf)`, `ieee1788.pown(Interval(2, 3), 2 ** 40)` = `[MAX, inf]` in 1 ms
(the 1788 layer stores floats).

**the exact result past 4300 digits**: `O(0.5, 3) ** 2 ** 22` finishes in 1.7 s with a 6647815-bit sup
whose `repr`/`str` raise ValueError and whose `float()` raises OverflowError. `M(3) ** -70000` has a
Fraction sup with a 110948-bit denominator: the same for Fractions, and the stdlib's own
`repr(Fraction(1, 10 ** 4300))` raises too (p3).

**cost of building an exact power, by result size** (p2, 2026-10-03, this laptop, loaded by other sessions;
`M(3) ** n` goes through the applicator, which evaluates the corner about three times):

| result bits | python `3 ** n` | `M(3) ** n` | `O(0.5, 3) ** n` | `M(Fraction(2, 3)) ** n` |
|---|---|---|---|---|
| 2**18 | 0.002 s | 0.008 s | 0.007 s | 0.010 s |
| 2**20 | 0.024 s | 0.073 s | 0.073 s | 0.077 s |
| 2**21 | 0.074 s | 0.22 s | 0.22 s | 0.23 s |
| 2**22 | 0.22 s | 0.66 s | 0.66 s | 0.69 s |
| 2**23 | 0.66 s | 1.95 s | 3.3 s | 3.7 s |
| 2**24 | 1.9 s | 8.8 s | 10.6 s | 12.4 s |
| 2**25 | 10.4 s | 36.8 s | 35.5 s | 35.9 s |

so "builds quickly" ends between 2**22 and 2**23 result bits (under a second vs seconds), the
designer's 2**20-2**24 band. the cost is in CPython's multiplication (Karatsuba): about x3 per doubling.

## 1. Q17 pown of exact operands

**the question**: `A ** n` with an int or Fraction corner builds the exact power whatever n is, so
`M(2) ** 2 ** 60`, `M(2) ** 1e300` (an integral float exponent is the int, D11) and `O(0.5, 2) ** 2 ** 40`
(the 2 stored as an int, as `O.parse('[0.5, 2]')` stores it) never finish; `pow_` of the same exact
operands already rounds past `EXACT_POWER_LIMIT`. the two routes disagree on exact points today
(`M(3) ** 70000` exact, `M(3) ** M(70000)` = `(MAX, inf)`).

**what the contract says now** (read): README "numbers" row: "int and Fraction exact, never rounded"
(README line 256); `v2-plan.md` "arithmetic": "int and Fraction are exact and never rounded" (line 314) but
also "an irrational value of an exact operand is its tightest float enclosure, open at both ends, in both
classes (`sqrt([2])`): an exact operand never loses its true value" (lines 323-325) and "`exp2`/`exp10` of
an int past `EXACT_POWER_LIMIT` are rounded rather than built exactly, and so is a rational power longer
than that many bits" (lines 428-430). so the invariant as actually built is: *an exact operand's result is
exact when it is rational AND short enough to build; otherwise it is the tightest float enclosure, which
still holds the true value.* pown is the one op that never applies the second clause.

### options

**(a) keep exact, document the limit (there is none)**
* meaning: no code change. `M(2) ** 2 ** 60` hangs as python's own `2 ** 2 ** 60` hangs (verified: both
  past 8 s; CPython raises nothing).
* pros: zero cost; the literal "never rounded" stays literally true for pown; matches python's own
  int semantics, which the exact class otherwise mirrors.
* cons: a hang is the worst failure mode a library can have inside a solver loop or a fuzz run (the
  M16b review found it by accident; `tests/test_pown_huge.py::test_reproductions_finish` had to be
  written as a subprocess with a timeout). the trap is about literal *types*, not mathematics: `O(0.5, 2.0)`
  answers in 0.1 ms and `O(0.5, 2)` never, with identical sets. pown stays inconsistent with pow and
  exp2/exp10 on exact operands. and the results that do finish past about 2**22 bits are unprintable (§2).
* when better: if the owner reads "never rounded" as the class's defining promise and accepts that a
  huge exact power is the user's own doing, as it is in plain python. for users doing exact rational
  arithmetic who would rather wait a minute than get floats.

**(b) raise `OverflowError` up front past a bit budget**
* meaning: before building, if `|n| * max(bits(num), bits(den)) > BUDGET`, raise (the check is O(1); the
  sketch said about 2**26 bits). python's precedent: `float(10 ** 400)` raises `OverflowError: int too large
  to convert to float`, but `int ** int` itself never does.
* pros: loud and immediate instead of a hang; the exact class stays exact-or-error, never a silent type
  change; cheap to build in `ops._exact_power_descriptor.fn`.
* cons: a new exception in the middle of arithmetic, which `DecoratedInterval` (NaI? raise?), `Dual`,
  `solve`/`newton` and numpy object loops each need a rule for; the result's *kind* depends on the
  operand's magnitude, which is the thing (c) avoids; pown then raises where `pow_` returns `(MAX, inf)` for
  the same exact point, a worse inconsistency than today's; and the budget must sit high enough not to
  refuse things that build in milliseconds (`M(3) ** 70000`), so results between 4300 digits and the budget
  are still unprintable (§2). the HullWarning pattern already shows the library prefers a coarser answer
  plus a warning over an error (`steps.py::ENUMERATION_CAP`, `errors.HullWarning`).
* when better: for a user base that pins types in downstream code (an int end must stay an int) and
  prefers a crash to a float. also as the *opt-in* form of (c), via a warning filter: see (e).

**(c) past a limit shared with `pow_`, the tightest open float enclosure (both designers)**
* meaning: an exact corner whose power's bound exceeds the limit is treated like a float corner past it:
  outward, `rounded_pow` down and up with attainment against the marker (an open piece, e.g. `(MAX, inf)`
  or `(0.0, 5e-324)`); to nearest, `rounded_pow(.., NEAREST)` (so `M(2) ** 2 ** 60` = `[inf]`, the exact
  class's nearest rule for overflow, as `M(2.0) ** 2 ** 60` already is). the marker proof already covers
  rationals (§0). the applicator change: today `desc.rounded` is called only "for a corner whose operands
  are all finite and at least one is a float" (`applicator.py` line 59-60, `evaluate_box` line 171
  `_rounds(args)`), so exact corners never reach the hooks; the descriptor needs a way to say "round this
  exact corner too" (e.g. `fn` returning the marker makes `evaluate_box` call the hooks). a moderate change
  in `ops.py` plus a small one in `applicator.py`, with the exact-corner rows added to
  `test_pown_huge.py::test_marker_boundary` and `::test_pown_matches_pow` (pinning `A ** n` == `A ** M(n)` on
  exact points is the pin that makes the two routes one).
* pros: never hangs, never raises, always sound (the true value is inside, exactly the `sqrt([2])` clause);
  one rule for pown, pow and (if its limit is restated in bits) exp2/exp10; the result's *type* tells the
  user rounding happened (float ends), the same signal `sqrt([2])` gives.
* cons: breaks the literal reading of "never rounded"; at the *current* limit it turns cheap exact results
  into floats (`M(3) ** 70000`, 2 ms today, would become `(MAX, inf)`) so the limit must rise; past the
  float range the answer collapses to `(MAX, inf)` and the magnitude is lost (an exact 3**70000 says
  more than `(MAX, inf)` does); and it does not by itself close §2 (a 2**22-bit result is 1.26M digits).
* the limit: one *exact-result* limit shared by pown, pow and exp2/exp10, in result bits, at **2**22**
  (4194304 bits, about 1.26M digits): the largest size that still builds under a second on this laptop
  (0.66 s, 2026-10-03, table in §0; 2**23 is 2-4 s, 2**24 about 10 s). the designer's "about 2**20 to
  2**24". a sharper argument for 2**22 than 2**24: a set op evaluates a corner several times and a `Dual`
  evaluates `n * u ** (n - 1)` too, so the per-op cost is a few builds; 2**22 keeps that under a few seconds.
* **do not move the float-corner threshold with it** (my addition to the designers' "one shared limit"):
  for a float corner the exact build is a speed choice, not a semantics one (the doubles are the same by
  `rounded_pow`), and `rounded_pow`'s ziv route is O(1) in n while the exact build is not: at 2**22 bits a
  float corner such as `O(1.5) ** 7000000` would cost 0.7 s where today it costs milliseconds. so: two
  named constants, `EXACT_RESULT_LIMIT` (exact operands; 2**22; the one pown/pow/exp2/exp10 share) and the
  existing `EXACT_POWER_LIMIT` as the float-corner build threshold (100000, floor 36550 from the marker
  proof). the pow/pown agreement on *floats* is already pinned by `::test_pown_matches_pow` independent of
  the thresholds, since both round the same exact value. (the exact-result limit for exp2/exp10 should be
  restated in bits — `|x| * 4 > limit` for exp10 — or the raised constant makes `M(4000000).exp10()` build
  a 13M-bit int.)
* what changes for pow at the new limit: exact results between 100000 and 2**22 bits that are `(MAX, inf)`
  today become exact ints/Fractions (e.g. `M(2) ** M(50001)`). more exact, never less sound; a value change
  in both classes for exact operands only.
* when better: for everyone who uses the library as an interval library (enclosures that always answer),
  and for consistency across pown/pow/exp. this is the library's own style elsewhere.

**(d) (c) in the outward class only, (a) or (b) in the exact class**
* meaning: the minimum that makes `O(0.5, 2) ** 2 ** 40` finish (the outward class rounds anyway, so an int
  end is incidental there); the nearest class keeps the exact build.
* pros: smallest surface; no change to `MultiInterval`'s results.
* cons: the two classes today differ only in the *direction* float results round; (d) makes them differ
  on *int* operands, breaking "exact operands give the same set" in both classes (the property Q19 is
  trying to protect from the other side); `M(2) ** 2 ** 60` still hangs; `Dual` and `DecoratedInterval`
  are built on `MultiInterval` so they keep the hang.
* when better: only if the owner wants (a) for `MultiInterval` on principle and still wants the outward
  trap closed.

**(e) (c) plus a warning, not in the plan** (my recommendation's shape)
* meaning: (c), and the rounding of an exact operand's result emits an `IntervalWarning` subclass (name
  it beside `HullWarning`; e.g. "the exact power has more than 2**22 bits, so its float enclosure was
  returned"), ignored by default like `EmptySetPropagationWarning` and `DomainClippedWarning` are
  (`errors.py` lines 55-56), so a user who wants (b) gets it with one line:
  `warnings.simplefilter('error', ThatWarning)`, the "tripwire" idiom `errors.py`'s docstring already
  documents. the same warning belongs on `pow_`'s and exp2/exp10's existing silent rounding of exact
  operands (today `M(2) ** M(2 ** 60)` = `(MAX, inf)` with no signal at all).
* pros: (c)'s never-hang default with (b)'s loudness on demand; it also makes today's silent `pow_`
  rounding visible; it follows an existing pattern exactly (`HullWarning`: "its hull was returned: a
  superset, with no pieces missing").
* cons: one more warning class to document; the warning fires inside hot loops only in the rare case,
  so no cost concern.

### recommendation

**(e): (c) with one shared exact-result limit of 2**22 bits, the float-corner threshold left where it
is, and a default-ignored warning as the opt-in error. confidence: high for (c) over (a)/(b)/(d) (about
85%); medium for 2**22 as the number (70%: 2**21 or 2**23 are defensible; any of them beats 100000, which
would downgrade results that build in milliseconds); medium for the warning (65%: it is extra surface, and
the owner may prefer the type change alone as the signal, as `sqrt([2])` has no warning).**

why: the library already has the (c) rule in three places (`sqrt` of a non-square, `pow_` past the limit,
exp2/exp10 past theirs) and documents it as "an exact operand never loses its true value"; pown is the
odd one out, and the odd one out hangs. the exact-class invariant worth keeping is *soundness plus
exactness where exactness is finite*, not exactness at any cost — the plan's own words (lines 323-325).
a hang is not an answer, and (b)'s error is a worse answer than (c)'s enclosure in a library whose other
"too big" case (`HullWarning`) chose the enclosure.

what would change my mind: (1) if the owner's intended audience is exact-rational users for whom a float
end is a wrong type (then (b) with the warning promoted to an error by default, i.e. (e) with the filter
flipped); (2) if the marker proof does not in fact hold for a general Fraction corner on re-derivation
(I read the docstring's three cases as covering every rational; a reviewer should re-derive it before the
build, as the pown-huge soundness lens did for floats).

cost of changing later: **before 2.0, cheap** — exact results past 2**22 bits change type in both classes
(no user has them, since today they are unprintable or hang) and pow results between 100000 and 2**22 bits
become exact (a value change only on exact operands; `tests/test_pow*`'s rows at the old boundary move).
**after 2.0, expensive the other way round**: going from (a) to (c) later changes the *type* of results
users could have relied on (an int end turning float is the kind of change the release note has to
shout), while going from (c) to a *higher* limit later is backward compatible (more exact results, never
fewer). so decide (c) now and the number can still move up later; (a) now and (c) later is the costly
order.

## 2. m14b-open: repr/format past python's 4300-digit int-str limit

**the question**: `repr(MultiInterval(10 ** 4300))` raises `ValueError: Exceeds the limit (4300 digits) for
integer string conversion` (python's CVE-2020-10735 mitigation, 3.11+; `fmt.format_value` is `str(value)`
for an int or Fraction). `tests/test_fmt.py` holds `BIG = 10 ** 4300 - 1` out of its round trip and calls
it an open question.

**what is affected, verified (p3, p6, 2026-10-03)**:
* raises: `repr`, `str` of `MultiInterval`/`OutwardMultiInterval`/`DecoratedInterval` with an int or
  Fraction end past 4300 digits (the stdlib's own `repr(Fraction(1, 10 ** 4300))` raises too); and
  **`M.parse('1' * 4301)` raises the same error** (`int()` of a long literal is limited too), so the round
  trip is blocked by python at both ends, not by `fmt`.
* works: `==`, `hash`, arithmetic (`M(big) + 1`), `pickle`, `mid()` (returns an int that then cannot be
  printed), `hex(big)` (power-of-two bases are not limited), `decimal.Decimal(big)` and its `'.17e'` format
  (`'1.00000000000000000e+4300'`). `float(M(big))` raises OverflowError as python's `float(big)` does
  (unrelated; by design). `np.asarray(M(big), dtype=float)` fails with numpy's unhelpful "setting an array
  element with a sequence" wrapping that OverflowError (a cosmetic finding, not for this question).
* no *value* operation was found to fail through a message: the `DomainClippedWarning` text formats the
  function's *domain*, not the clipped operand (`functions.py` line 160), and the indeterminate-box
  messages (`applicator.py` line 122, `functions.py` lines 481 and 674) format boxes made of 0, 1 and
  ±inf. `ops._power_name` already caps the descriptor name (pown-huge C3). so the limit bites only in
  `repr`/`str`/`parse` — and `pytest`'s assertion rewriting, logging and debuggers, all of which call
  `repr`.
* how big is it: an int of 2**22 bits (Q17's proposed limit) is 1.26M digits; `str()` with the limit
  lifted takes 0.6 s (CPython 3.12+'s divide-and-conquer `_pylong`), 2**24 bits (5M digits) 8.4 s;
  `hex()` is 2 ms and 20 ms; `Decimal(v)` is quadratic (38 s at 2**22 bits) and not a route.

### options

**(1) let it raise (today)**
* meaning: python's own choice for `repr(10 ** 4300)`; document it as python's limit and point at
  `sys.set_int_max_str_digits`.
* pros: zero cost; honest (no lossy spelling); the user who built a 10000-digit end knows it.
* cons: a `repr` that raises is unusual and breaks tools that assume it does not (pytest's failure
  output, `logging`, `%r` in a user's message, REPL echo); under Q17 (c) at 2**22 bits the library itself
  *produces* such values in 0.7 s (`O(0.5, 3) ** 2 ** 22` today, 1.7 s), so "the user built it" is not the
  whole story.
* when better: if 2.0 ships with Q17 (a) or (b) unchanged and the owner wants no new text forms; cheap to
  revisit (see cost below).

**(2) lift the limit locally around the conversion**
* meaning: in `fmt.format_value` and `fmt.parse_value`, on the ValueError, retry with
  `sys.set_int_max_str_digits(0)` and restore in `finally`. decimal output as today, round trip restored.
* pros: smallest change; the output stays decimal, which `parse` (also lifted) reads back; 0.6 s for 1.26M
  digits is acceptable for something the user asked to print.
* cons: the setting is process-global and not a context manager: a concurrent thread parsing untrusted
  input loses the DoS mitigation during the window; a library silently flipping an interpreter security
  knob is a smell reviewers flag; and the output is still a 1.26 MB line nobody reads.
* when better: if the owner wants decimal round trip above all and the library is single-threaded in
  practice (it is pure python with no threads of its own; the hazard is the host's).

**(3) an abbreviated or scientific spelling for ints past the limit**
* meaning: `format_value` writes e.g. `<1.4657482936901607e+1262581>` or `<a 4194204-bit int>` for an int
  (and each part of a Fraction) past 4300 digits; `repr` still evaluates? no — `parse` would have to refuse
  or read the abbreviation, so `repr` stops being evaluable for these values (python's `repr` convention
  tolerates `<...>` forms that do not evaluate).
* pros: readable, bounded output; no global state; the leading digits and the decimal exponent come
  cheaply from `bit_length` and one shift (not from `Decimal`, which is quadratic) — an exact decimal
  exponent needs a `10 ** k` comparison (0.2 s at 2**22 bits) or can be stated as "about".
* cons: breaks `fmt`'s stated contract "`parse(format(x)) == x`" (`fmt.py` docstring line 13) and
  `test_fmt`'s round-trip property for these values; two spellings of the same type (exact below, lossy
  above) is a seam users hit without warning; the `MultiInterval.parse(...)` repr form would wrap a string
  `parse` rejects.
* when better: for human-facing output only (`__str__`), paired with an exact `__repr__` from (4) or (2).

**(4) hex spelling past the limit, and `parse` reads `0x`** (my addition)
* meaning: `format_value` writes an int past 4300 digits as `hex(v)` (`-0x...`), a Fraction as
  `0x.../0x...`; `fmt._NUMBER` gains `0x[0-9a-f]+` (optionally over `/0x...`), and `parse_value` reads it
  with `int(s, 16)`, which python does not limit. round trip exact, O(n) at both ends (2 ms for 2**22 bits),
  no global state. below the limit nothing changes.
* pros: `repr` stays evaluable and the `fmt` contract holds for every value; python's own escape hatch
  for huge ints is `hex()`; `parse` accepting hex is harmless and useful on its own (`0x1p-3`-style float
  hex could come later, separately). grammar growth is backward compatible.
* cons: two spellings for one type (decimal below, hex above) — the same seam as (3) but lossless; hex
  is unreadable to humans, though so is a 4300-digit decimal; a small grammar extension to test (the
  fmt spelling strategies in `test_fmt.py` already generate every spelling, so the test cost is a few
  lines).
* when better: if the owner wants `repr` never to raise and always to evaluate back, and accepts hex as
  the "you asked for something enormous" spelling.

### recommendation

**decide the *principle* with Q17 and the *spelling* later; the principle: `repr`/`str` must not raise
once Q17 (c) lets the library itself produce such values in under a second. spelling: (4) hex with `parse`
reading `0x`, falling back to (1) if the owner wants no grammar growth in 2.0. confidence: medium (65%)
that (4) is best; high (85%) that this can safely wait past the Q17 build, see the cost paragraph.**

why hex over lifting the limit: no interpreter-global knob, O(n) instead of 0.6-8 s, exact round trip;
why over abbreviation: `fmt`'s contract is round trip, and `repr` is used by the tests' own oracles.

what would change my mind: if the owner would rather keep `fmt`'s grammar frozen for 2.0, (1) is the
right call, documented in the "numbers" row, because of the cost argument below.

cost of changing later: **near zero either way, which is why it can wait.** everything the limit touches
*raises* today, so any spelling chosen after 2.0 only turns errors into output — no existing user output
changes. `parse` growing to read `0x` is backward compatible at any time. the only thing 2.0 must not do
is *promise* that the decimal form is the sole spelling. one hazard to record now: `tests/test_fmt.py`'s
round-trip strategies stop at 4300 digits (`BIG`), so whichever spelling lands needs the strategies
widened past it in the same change, or the new path is unpinned.

## 3. Q18 correctly rounded pown to nearest

**the question**: to nearest, a float corner of `A ** n` with `|n| <= 2 ** 53` is python's `float ** int`,
i.e. the C library's `pow` (`ops._exact_power_descriptor.fn`: `x ** k`, and `x ** n + 0.0` for n < 0),
"not promised correctly rounded"; past `2 ** 53` it is `elementary.rounded_pow(.., NEAREST)`, which is.
is correctly rounded pown a promise of the nearest class?

**what the nearest class promises elsewhere** (read):
* README "functions" bullet (line 126-127): "values are correctly rounded by a pure-python evaluator (no
  libm), so they are the same on every platform"; README "rounding" (line 196): "`MultiInterval` rounds a
  float result to nearest"; the comparison table (line 260): "`MultiInterval` rounds to nearest as python's
  float ... Q18 open".
* `v2-plan.md` line 422: "`elementary.py` computes every value in pure python, correctly rounded in all
  three directions ... so results are the same on every platform; libm is not correctly rounded (this
  laptop's UCRT `acosh` is 2 ulp off near 1, measured 2026-09-25)"; line 2081: "**no libm**: every value
  comes from a pure-python, correctly rounded evaluator".
* `tests/test_extreme_floats.py::test_nearest_class_rounds_the_exact_result_to_nearest` (docstring, lines
  464-471): the nearest class's float results are "python's own float ops (correctly rounded) or the exact
  value rounded once ... pow is left out: a float corner is libm's `pow`, not promised correctly rounded".
  this is the one explicit carve-out in the tree, and it is a carve-out in a *test oracle*.
* grep of `math\.` over `intervals/`: the only libm calls are `math.log(2)` as a precision estimate
  (`elementary.py` line 190) and `math.frexp` in `solver.py`; `+ - * /` are IEEE (correctly rounded by
  the hardware). so **pown with `|n| <= 2 ** 53` to nearest is the only value in the library that comes
  from libm**, and the only one that can differ between platforms.

**is libm's pow correctly rounded here? no — measured, not inferred** (2026-10-03, this laptop's UCRT,
python 3.13.15; `p4.log`, `p7.log`):
* 20000 random `(x, n)` (bases over the whole double range, near 1, in [0.5, 2]; n in 2..10**6 and
  -300..-2): **6 of 15712** finite results differ from `rounded_pow(.., NEAREST)` by 1 ulp, at n = 6, 10, 139,
  -175, -243, -285; MPFR agrees with ours on all 6. (the 2026-09-29 check of 3000 near-1 bases found none:
  a narrower sample.)
* CORE-MATH's `pow` worst cases (`.scratch/coremath-cache/.../pow.wc`, 994438 rows; 872081 unique with an
  integral exponent `|n| <= 2 ** 53`, 270639 with a finite nonzero libm result): **libm differs from the
  correctly rounded double on 63169 (23%)**; against `gmpy2.ieee(64)` RoundToNearest (the oracle
  `tests/test_pown_huge.py::test_nearest_huge_exponent_against_mpfr` uses) **libm is wrong on all 63169
  and `rounded_pow` on 0**. the wrong ones span every |n| from 2 to past 100 and normal as well as
  subnormal results; e.g. `1.0287349703833546 ** 10` is `1.32750153854842` from libm, correctly
  `1.3275015385484197`; `1.0026606152364441 ** 13` is `1.0351455727723202`, correctly `1.03514557277232`;
  `2.1095375758280437e-154 ** 2` is `4.45014878383046e-308`, correctly `4.450148783830459e-308`. these are
  hard cases by construction, but they are plain inputs: `M(1.0026606152364441) ** 13` is wrong by an ulp
  today.
* so the nearest class's `A ** n` is today (i) not correctly rounded, (ii) platform-dependent (glibc's
  `pow` since 2.28 is a different implementation with its own misses; CI on ubuntu and this laptop can
  disagree on a pinned value — `::test_nearest_past_2_53`'s row at exactly `2 ** 53`, commented "python's
  own, still", pins a libm value; it passes on both today, inferred from CI being green), and (iii)
  inconsistent with itself across the `2 ** 53` seam, where it becomes correctly rounded.

**cost of the alternatives per float corner** (p5, 2026-10-03; `x = 1.2345678901234567`):

| n | libm `x ** n` | exact `Fraction(x) ** n` then `round_rational(.., NEAREST)` | `rounded_pow(.., NEAREST)` (tries exact first, then ziv) |
|---|---|---|---|
| 2 | 0.10 us | 1.35 us | 3.4 us |
| 7 | 0.10 us | 1.75 us | 4.0 us |
| 20 | 0.10 us | 2.5 us | 5.2 us |
| 100 | 0.09 us | 13.6 us | 14.3 us |
| 1000 | 0.09 us | 354 us | 359 us |

set level: `M(1.2, 2.5) ** 7` is 20 us today and `O(1.2, 2.5) ** 7` (which already does exact-then-round
at both ends, with flags) 44 us; `M * M` 31 us, `M.sqrt()` 57 us, `M.exp()` 126 us. so a correctly rounded
nearest pown costs about +2 us a corner for ordinary n (a 20 us op becoming perhaps 25 us), and for
n in the thousands what the outward class pays today.

### options

**(i) keep libm, document the carve-out**
* meaning: no change; README's "no libm, same on every platform" gets an exception for pown to nearest;
  the carve-out in `test_nearest_class_rounds_the_exact_result_to_nearest` stays.
* pros: fastest (0.1 us); `M(x) ** n` equals python's own `x ** n` bit for bit, which is one reading of
  "rounds to nearest as python's float".
* cons: the one platform-dependent value in a library whose stated selling point is that there is none;
  wrong by an ulp on 23% of the hard cases and on about 1 in 2600 random ones; the class is correctly
  rounded for `|n| > 2 ** 53` and not below, so the promise cannot even be stated per class; pinned libm
  values in tests are a latent CI/laptop disagreement.
* when better: if the owner reads "as python's float" as *bit-compatibility with python's `**`* and wants
  the nearest class to be exactly "what python would have computed", speed first.

**(ii) exact power rounded once, within the float-corner threshold; `rounded_pow` to nearest past it**
(my recommendation)
* meaning: the nearest form of `_power_descriptor` reuses the outward form's `float_exact` (the cached
  `exact_pow` while `|n| * bits(x) <= EXACT_POWER_LIMIT`) and returns `round_rational(v, NEAREST)`; past the
  threshold, and already for `|n| > 2 ** 53`, `s * rounded_pow(|x|, n, NEAREST) + 0.0` as today. the
  `2 ** 53` branch disappears: one rule for every n. an overflowing exact value rounds to `inf` through
  `round_rational` as the nearest rule wants (`round_rational(2 ** 1024, NEAREST)` is `inf`, its doctest).
* pros: correctly rounded by construction (int/int to float is correctly rounded in CPython; `rounded_pow`
  is pinned against MPFR and has the gmpy2 backend); platform-independent, which lets the README's claim
  stand without an exception; no new numerics — the exact path is the outward class's and the ziv path
  is pow's; the test carve-out can be *removed*, so an existing oracle gains coverage, and
  `::test_nearest_huge_exponent_against_mpfr` can drop its `PAST_2_53` restriction and run over every n;
  the CORE-MATH integer-exponent rows become a free regression set (`tests/test_coremath.py` could gain
  pown rows derived from `pow.tsv`, closing part of vectors-ext (b) "pown has no CORE-MATH file" — the
  pow file's integral exponents are exactly pown's hard cases).
* cons: +1.3 to 2.5 us per float corner for small n (a 10-25% slowdown of a 20 us op, still under
  `M.sqrt()`); for n near the threshold hundreds of us, as outward pays today; `M(x) ** n` can differ from
  python's `x ** n` by an ulp (now in the library's favour); value changes in the nearest class on about
  1 in 2600 random inputs (pinned nearest values in tests that came from libm would move: a review of
  `tests/test_outward.py` line 263's rows and `::test_nearest_past_2_53`'s `2 ** 53` row is part of the
  change).
* when better: for every reader of the README who took "no libm" at its word; for reproducibility across
  CI and laptops; for the `Dual`/`solve` users who expect the nearest class to be "the exact value, rounded
  once", as every other op is.

**(iii) `rounded_pow(.., NEAREST)` for every n, no exact shortcut**
* meaning: the ziv route always. same doubles as (ii).
* pros/cons: strictly slower than (ii) for small n (3.4 vs 1.35 us) because `rounded_pow` calls
  `exact_pow` first anyway, and `_ln_bracket`'s shortcuts only pay off past the float range. no reason to
  prefer it over (ii); listed for completeness.

**(iv) libm, then verify and correct**
* meaning: compute libm's value, then check it against the exact value and fix it when off.
* cons: checking needs the exact value, which is (ii)'s whole cost; a strict superset of (ii)'s work.
  not a real option.

### recommendation

**(ii). confidence: high (85%).** the nearest class's promise, as the README and plan state it, is
"the exact value rounded once to nearest, with no libm, the same on every platform"; today pown below
`2 ** 53` is the single exception, it is measurably wrong (23% of CORE-MATH's integer-exponent hard cases,
1 in 2600 random inputs, on this laptop), and fixing it reuses two paths already pinned against MPFR at a
cost of a couple of microseconds. the only honest alternative is (i) with the README's "no libm" claim
amended, which trades the library's distinguishing property for 2 us.

what would change my mind: if the owner's reading of "rounds to nearest as python's float" is a
bit-compatibility promise with python's `**` (then (i), and the README must say "except pown, which is
libm's `pow` and may differ by platform"); or a measured hot loop where pown dominates and 2 us a corner
matters (none in the tree: the gate's own timing is dominated by hypothesis and exact arithmetic).

cost of changing later: **before 2.0: a value change of at most 1 ulp on rare inputs in the nearest
class, with the pinned libm values in tests moved in the same commit — cheap.** **after 2.0: the same
code change, but now a documented-behaviour change** ("results may differ in the last bit from 2.0.x")
that a user comparing against stored outputs would see; and the longer libm is in, the more pinned values
and downstream goldens depend on it. the reverse move (from (ii) back to libm) would never be wanted.
decide before 2.0.

## 4. summary of recommendations

| question | recommendation | confidence | decide before 2.0? |
|---|---|---|---|
| Q17 pown of exact operands | (c) the tightest open float enclosure past a limit, as (e): one exact-result limit of 2**22 bits shared by pown, `pow_` and exp2/exp10 (restated in bits), the float-corner threshold `EXACT_POWER_LIMIT` left at 100000, plus a default-ignored warning (the `HullWarning` pattern) as the opt-in error | 85% for (c); 70% for 2**22; 65% for the warning | yes: (a) now and (c) later changes result *types* under users; (c) now and a higher limit later is backward compatible |
| m14b-open, 4300 digits | the principle now (`repr`/`str` must not raise once the library produces such values in under a second); the spelling after the Q17 build: hex past the limit with `parse` reading `0x` (lossless, O(n), no global knob); else leave it raising, documented | 65% for hex; 85% that it can wait | no: everything it touches raises today, so any later spelling only turns errors into output; just do not promise decimal as the sole form |
| Q18 correctly rounded nearest pown | (ii) exact power rounded once within the float-corner threshold, `rounded_pow` to nearest past it, for every n; drop the `2 ** 53` branch and the test carve-out; add CORE-MATH's integral-exponent pow rows as pown rows | 85% | yes: it is a last-bit value change, cheap now, a documented behaviour change after |

**one build, not three**: (c)/(e) and (ii) both live in `ops._power_descriptor` and both reuse
`float_exact` + `rounded_pow`; built together, the descriptor has one rule: *a corner's power is built
exactly while short, else rounded by `rounded_pow`; outward in two directions with the marker for
attainment, to nearest once.* the 4300-digit spelling is a separate, later `fmt` change.

**what a reviewer should re-derive first-hand before that build** (my inferences, not probed): that
`ops._NotADouble`'s proof covers a general Fraction corner (its three cases read as covering every
rational in lowest terms); and that `applicator.evaluate_box` can be told to run the hooks for an exact
corner without disturbing Q19's "exact operands stay exact" for every *other* op (a per-descriptor
signal, e.g. `fn` returning the marker, keeps it local to pown).

**findings outside the three questions, recorded here so they are not lost** (2026-10-03):
* exp2/exp10's exact limit is on the exponent, not the result's bits (`elementary._exact` line 550):
  `M(100000).exp10()` builds 332193 bits while `M(10) ** M(25001)` (83052 bits) is rounded. a third
  threshold, to fold into the shared one.
* `np.asarray(M(10 ** 4300), dtype=float)` fails with numpy's "setting an array element with a sequence"
  rather than the OverflowError underneath (cosmetic).
* `M(3) ** n` costs about three times python's `3 ** n` (table in §0): the applicator evaluates an exact
  corner more than once (the evaluate-box row in `HANDOFF.md` notes the same for float corners under
  OUTWARD). an exact-power cache like `float_exact`'s would make the exact route pay one build.
* `tests/test_pown_huge.py::test_nearest_past_2_53` pins one libm value (`2 ** 53`, "python's own,
  still"), a latent platform disagreement while libm stays.

probe scripts: `.scratch/owner-questions/probes-pown/` (`run.py` runs each snippet in a subprocess with a
hard timeout and kills its own child; `p1`..`p7`, logs beside them). nothing in the tracked tree was
changed.
