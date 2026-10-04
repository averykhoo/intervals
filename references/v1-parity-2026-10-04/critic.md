# completeness critic (2026-10-04)

working dir: .scratch/v1-parity/critic/

## part 1: v1 surface vs auditors rows

method: `grep -nE '^\s*(def|class) |^[A-Z_]+ *[:=]' archive/v1/*.py` (every def/class/constant), read of the
signatures and branches of __init__, merge, apply_monotonic_*, __pow__ (incl. modulo branch), reciprocal, the step
functions, DateTimeInterval/TimeDeltaInterval __init__, compare.py strategies; each matched against the 10 row lists
and a keyword grep of the reports (run_unsorted_process, run_incremental_sort, run_timsort_no_key, pickle, hash,
truthiness, dataclasses, replace(, sort=False ...).

### auditors' not_covered items already covered by ANOTHER slice's rows (no task)
* construct: _intervals_intersect / _mod_attained -> setops row "_intervals_intersect", floordivmod row "_mod_attained".
* setops: __getitem__ -> construct rows; merge n_overlaps -> construct rows (DROPPED_DOCUMENTED); time-layer set ops -> datetime/timedelta rows.
* arith: __floordiv__ -> floordivmod rows.
* accessors: time accessors -> datetime/timedelta rows.
* floordivmod: time //, %, divmod -> v1 TimeDeltaInterval/DateTimeInterval define none of them (refused; timedelta row "refused pairings ... T//2 [EQUAL]").
* readme: v1 merge non-string inputs -> construct rows; time-layer full parity -> datetime/timedelta.
* OutwardMultiInterval / decorated / -0.0 in outward (setops, arith, powfn, intervalpy, readme): not v1 capabilities, nothing to compare.

### v1 surface no row covers (new, found by this critic)
* DateTimeInterval truthiness: v1 has no __bool__ on DateTimeInterval (object default), v2 `__bool__` = non-empty.
  RAN `.scratch/v1-parity/critic/probe_dti_bool.py`: `empty bool v1 True v2 False`; `point bool v1 True v2 True`;
  pickle/deepcopy round trips equal on both sides. same case as timedelta's row "truthiness bool(T())
  [UNDOCUMENTED_DIFFERENCE]"; the only doc line is the core's v2-plan.md:2688 "`__bool__` = non-empty, explicitly
  (set precedent)", which does not mention the time layer. the datetime slice has no row for it.
* v1 MultiInterval iteration: no __iter__/__len__, but __getitem__ takes a number, so `iter(A)` uses the sequence
  protocol A[0], A[1], ... and never stops. RAN `critic/probe_1982_and_iter.py`: `v1 first 5 from iter(): ['{}',
  '[1]', '[2]', '{}', '{}']` (list(A) would hang); v2 `list(A)` = pieces `['[1, 2]']`. a v1 bug, not a capability;
  no row records it (construct).
* compare.py `run_unsorted_process` (sort an UNSORTED raw record list, then sweep), `run_incremental_sort`,
  `run_timsort_no_key`, `generate_sorted_chunk`: no row; the readme rows cover fast_sweep/optimized_sweep on sorted
  input, timsort/heapq bulk union and bisect insertion only.
* local-zone history: this laptop's Windows zone ('Malay Peninsula Standard Time') has no 1982 +7:30 -> +8:00 shift:
  RAN `critic/probe_1982_and_iter.py`: across 1981-12-31 23:00 -> 1982-01-01 01:00 `v1 total_seconds 7200.0`,
  `v2 total_seconds 7200`. so neither DST (Q21(g)) nor historical-offset differences can be probed on this machine.

## part 2: EQUAL / EQUIVALENT_RENAMED claims whose probe could not have failed (re-runs)

static scan: `grep -n 'or True\|except.*pass' .scratch/v1-parity/*/probe*.py` plus a read of every slice's common.py
and the evidence column of every EQUAL/EQUIVALENT_RENAMED row.

### W1. datetime "`__contains__` of a date (whole day a subset) [EQUAL]" -- over-claimed
* probe_expand_contains_overlap.py:54 `check((dd in a1) == (dd in a2) or has_date or True, 'date in')` can never fail;
  the second loop ("date in A disagreements: 0") draws ends with random microseconds (gen.py `rand_dt`), so no set ever
  ends at a day's last microsecond, the one place v1's closed day `[00:00, 23:59:59.999999]` and v2's half-open day differ.
* RAN `timeout 120 .../python.exe .scratch/v1-parity/critic/probe_date_in.py`:
  `[d 00:00, d 23:59:59.999999] from datetimes: v1 True  v2 False <-- DIFFER`; four other edge shapes agree.
  sabotage built in (asserts >= 1 difference; passes, so the probe can fail).
* verdict for that edge: DIFFERS_DOCUMENTED (v2-implementation-plan.md D30 (c): "a `date` is the half-open day
  `[d 00:00, d+1 00:00)`"). the row should be DIFFERS_DOCUMENTED at the day edge, not a blanket EQUAL.

### W2. arith "float rounding of + - * / (to nearest) [EQUAL]" -- false for `/`
* evidence was hand cases only (arith not_covered: "The v1 set-set sweep never put float endpoints in random sets").
* RAN `.scratch/v1-parity/critic/probe_float_sweep.py` (seed 4242, 700 random float-ended set pairs per op, exact
  structure: end values bit-for-bit and closedness): `agree {'+': 700, '-': 700, '*': 693, '/': 457}`;
  `* differ 6` (all zero-holding open corners, e.g. `[0, 934.29) * (1.85, 3.3)`: v1 `(0.0, False ...)`, v2 `(0.0, True
  ...)`, the known V1_BUG_FIXED row); `/ differ 243`, one ulp, e.g. v1 `-1261.3983923837163` v2 `-1261.3983923837166`.
  sabotage run (`... probe_float_sweep.py sabotage`, one v2 end nudged by an ulp) is caught.
* RAN `.scratch/v1-parity/critic/probe_float_div_exact.py`: for every differing end, `{'v2 nearest, v1 not': 304,
  'v1 nearest, v2 not': 0}`: v2's end is the correctly rounded exact quotient of a corner, v1's is not (v1's
  `__truediv__` = `self * other.reciprocal()` rounds twice: `1/b`, then `a * (1/b)`).
* verdict: `+`, `-` EQUAL (now with a float sweep); `*` EQUAL except the known zero-corner fix; `/` with float ends
  V1_BUG_FIXED (double rounding), not EQUAL. same applies to the "A / B with 0 not in B [EQUAL]" row, whose sweep used
  Fraction ends only, and to `x / A`.

### W3. setops "merge_adjacent() (distance 0, sort=) [EQUIVALENT_RENAMED]" -- probe vacuous, conclusion holds
* probe_interval_ops.py:27 compares `a1.copy().merge_adjacent()` with `a2` on sets v1 built publicly, which are already
  merged: identity vs the v2 set itself; it could only fail if `build` were inconsistent.
* RAN `.scratch/v1-parity/critic/probe_merge_adjacent_raw.py` (seed 77, 500 hand-built UNMERGED, SHUFFLED raw
  endpoint lists of 1-5 pieces incl. points and touching open/closed ends; v1 `merge_adjacent()` vs v2
  `MultiInterval.from_pieces(pieces)`, exact piece structure and membership on a 1/100 grid, and v2 vs brute force):
  `cases 500 v1 merge_adjacent vs v2 from_pieces mismatches 0  v2 vs brute 0`. sabotage arg (`... sab`, one v2 set
  given an extra point) raises AssertionError. verdict stands: EQUIVALENT_RENAMED, spelling `MultiInterval.from_pieces`.

### W4. timedelta "__str__ (non-negative) [EQUAL]" -- over-claimed
* probe_props.py:43 compares `str(a).replace(', ', ' ') == str(b).replace(', ', ' ')...`, which erases the difference.
* RAN `.scratch/v1-parity/critic/probe_td_str.py`: `'[1 day, 2:00:00, 3 days, 0:00:00]' | '[1 day 2:00:00, 3 days
  0:00:00]' <-- DIFFER`, `'[2 days, 0:00:00]' | '[2 days 0:00:00]'`, multi-piece likewise; `identical texts 4 of 7`
  (all sub-day durations identical).
* verdict: DIFFERS_DOCUMENTED for any duration >= 1 day: v2-implementation-plan.md:457 "`str` of a duration is python's
  text signed as a whole without the comma" (and v2-plan.md:1281 "without the comma (`-0:20:00`, `2 days 1:00:00`)").
  EQUAL only below one day.

### W5. datetime ".interval attribute / building from raw seconds [EQUIVALENT_RENAMED]" -- probe hard-codes the zone
* probe_misc.py checks `v1 .interval + 28800 == v2 .seconds` (28800 = this laptop's UTC+8): true here only, and the
  two numbers are different quantities (v1 POSIX seconds via `timestamp()`, v2 naive wall-clock seconds, D30 (a)).
* RAN `.scratch/v1-parity/critic/probe_interval_attr.py` (seed 5, 300 naive instants 1990-2031):
  `v2 naive .seconds == v1 .interval: 0`, `v2 DTI(t.astimezone()).seconds == v1 .interval: 300`.
* verdict: EQUIVALENT_RENAMED holds with the zone-independent spelling `DateTimeInterval(t.astimezone()).seconds`;
  the naive `.seconds` is DIFFERS_DOCUMENTED (D30 (a) "a naive datetime is exact wall-clock seconds ..., never
  `timestamp()`"). (DST caveat: `astimezone()` on a naive time inside a DST gap/fold follows python's fold rule, as
  v1's `timestamp()` did; not probeable on this machine, see part 1.)

### W6. arith "operand types ... gmpy2 mpq/mpz/mpfr ... [EQUAL]" -- evidence was printed text; set-compare crashed
* arith/operand_types.out ends in a traceback: the file's exact set-compare loop died at the gmpy2.mpq case
  (`SystemError: Object does not appear to be Fraction` inside v1 `__contains__` via common.py::finite_diff), so the
  mpq and float32 set-compares never ran; the EQUAL rests on `str()` lines like `MultiInterval [4/3, 7/3)` both.
* RAN `.scratch/v1-parity/critic/probe_gmpy2_operands.py` (exact end values as Fractions, `[1,2)` op scalar, 4 ops):
  mpq(1,3), mpz(2), mpfr('0.1') at 53 bits, np.float32(0.1): `same exact ends True` on all 16 (types differ: v1 keeps
  `mpq`/`mpfr` ends, v2 `Fraction`/`int`/`float`). mpfr('0.1', 200 bits): `same exact ends False` on all 4, e.g. `+`
  diff `8.88e-17`: v1 rounds in gmpy2's 53-bit context, v2 is exact.
* verdict: EQUAL for mpq/mpz/53-bit mpfr/float32; a WIDE mpfr is DIFFERS_DOCUMENTED (v2-plan.md ~1118-1123: "a
  foreign real ... any other real is the float it equals where it is a double ..., else its exact
  `as_integer_ratio()` ... so ... a wide `mpfr` are exact"; v2-plan.md:1789 "a foreign real is exact (Q15(b))").
  v1 Fraction/float/mpq queries on an mpq-ended set did not crash in `.scratch/v1-parity/critic/probe_mpq_contains.py`
  (9 queries), so the auditor's crash input is unidentified (a v1 quirk, not a capability).

### other static observations (not re-run)
* datetime/probe_accessors.py:77 `check(len(p1) == len(p2) or True, 'n pieces')` never fails; the piece comparison
  survives only as a printed count `pieces identical as sets` (the row is V1_BUG_FIXED anyway).
* timedelta: the auditor's own note that probe_misc.py and probe_arith_extra.py have no sabotage assertion; they back
  the EQUAL rows "pickling, copy.copy, copy.deepcopy" and part of "T * real" (np.float64). the pickle/deepcopy part is
  independently confirmed for DateTimeInterval by critic/probe_dti_bool.py (round trips equal on both sides).

## part 1 (cont.): uncovered items as audit tasks (incl. the auditors' own not_covered lists)
* datetime: add a row for bool(DateTimeInterval()) (v1 True, v2 False; critic/probe_dti_bool.py) and decide it with the timedelta truthiness row.
* datetime: DST and historical-offset behaviour (Q21(g)): run v1 vs v2 naive and aware arithmetic across a DST change in WSL with TZ=Europe/London (and Australia/Lord_Howe), since this Windows zone has no DST and no 1982 shift.
* arith: v1 arithmetic, reciprocal, division by a set holding 0, and pow under INFINITY_IS_NOT_FINITE=False (closed ±inf ends, [inf] points) vs v2's closed-inf ends.
* accessors: inf/sup/closed flags, closed_hull, is_finite, cardinality of sets with closed ±inf ends and [±inf] points, with v1's flag set False (the only way v1 can hold them).
* accessors: gmpy2 mpq/mpz/mpfr (incl. a wide mpfr) and np.longdouble ends through every accessor.
* arith: apply_monotonic_unary/binary_function with a user function mixing Fraction and float within one call.
* arith: extend binary_sweep.py with float ends (critic/probe_float_sweep.py did + - * / once; also x / A, and reciprocal of float sets, which double-rounds the same way in v1).
* floordivmod: compare the TYPE (int vs float) of // results with a float operand, v1 math.floor vs v2.
* powfn: seeded sweep of multi-piece bases mixing [0, b] pieces with interval exponents holding negatives.
* intervalpy: dataclasses.replace/asdict/fields, keyword construction Interval(start=, start_open=, end=, end_closed=), pickling of Interval/MultipleInterval, frozen-ness.
* intervalpy: MultipleInterval hashability (v1 __eq__ without __hash__) vs v2.
* intervalpy: numpy nan/inf scalar endpoints (np.float64('nan'), np.inf) in Interval.__post_init__.
* construct: merge called on an instance; keyword-only enforcement of start_closed/end_closed (positional third arg).
* construct: iteration/len protocol: v1 iter(A) never ends (sequence protocol over __getitem__(int)); v2 iterates pieces. record it.
* setops: merge_adjacent(sort=False) on a hand-built UNSORTED list (critic did sort=True only).
* timedelta: __str__ of non-whole-microsecond ends (text, not read-outs); v1 td / TimeDeltaInterval and -T (v1 has no __rtruediv__/__neg__) vs v2.
* readme: compare.py run_unsorted_process / run_incremental_sort / run_timsort_no_key on unsorted record lists vs v2 from_pieces; full-scale (50000 x 2) bench if wanted.
