# handoff

what is true now: ranked open items, open questions for the owner, loose ends, and a dated session
log. volatile by design. the spec of anything not yet built stays in `v2-implementation-plan.md`
(milestones, D rows, exits) and the design in `v2-plan.md` ("current design"); the rows below only
point at them. a task the owner assigns overrides the ranking. at the end of a session: add a
session-log entry, refresh the banner, edit rows in place, sweep closed ids (a finished item leaves
the open-items table: its record goes in the plan's milestone section, with a one-line entry in the
session log below; nothing is listed as open and done at once), and list anything skipped as
"Still owed:".

## banner (2026-09-28)

* **M16 (H3's second part) built on `v2` (merged in `h3-merge`, then fast-forwarded), committed, not pushed**: the owner, 2026-09-27: "get
  the rest of h3 done". five streams, each on its own branch off `v2` at `04946af`, merged into
  `h3-merge` without conflicts: M16a the solver in several variables (`gradient`, `jacobian`,
  `solve`, `RootBox`, exported from `intervals`), M16b the 1788 layer (`intervals/ieee1788.py`, not
  imported by `intervals`), M16c the per-piece allen matrix (`allen_matrix`, `allen_relations`),
  M16d numpy interop (`intervals/numpy_compat.py`; numpy optional), M16e the gmpy2/mpfr backend
  (`intervals/backend.py`, `intervals/_gmpy2.py`; opt-in, the pure path by default). the choices
  the build made are D20-D24, owner questions Q12-Q16. gate on the merged tree green: 27795 passed in 84 s (`tests/itf1788`) + 5535 in 747 s (the rest) = 33330, 2026-09-28 at `511056d` (the merged code; the doc commit after it changes docs and one comment only). records:
  plan §2 M16; design: `v2-plan.md` "current design" and its five 2026-09-28 revisions
* **M15 (H3's first part) built on `v2` (`04946af`), committed, not pushed**: forward-mode autodiff
  (`intervals/autodiff.py`, `Dual`, `derivative`) and interval newton (`intervals/solver.py`,
  `newton`, `Root`), exported from `intervals`. the choices the build made are D19, owner question
  Q11. record: plan §2 M15; design: `v2-plan.md` "the solver stack". M16 is on top of it

* branch `v2`: M13 finished and merged 2026-09-27 (branches `m13e`, `m13g`, merged in `m13-merge`,
  then fast-forwarded into `v2`); pushed. `origin/v2` is at `a1d45a9`, whose CI run 36305984327
  is green (all 8 jobs, 22166 passed on each of python 3.11-3.14, 2026-09-27): M13e and M13g have
  now run on every supported python
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root; on
  this shared laptop it runs past the 10-min tool limit, so run it as two calls (`tests/itf1788` and
  `--ignore=tests/itf1788`; M16's streams split the second in three by file). last recorded
  2026-09-27 at M15: 18246 passed in 67 s + 4088 in 482 s (22334) (22166 at the H2' fix); on the
  merged M16 tree green: 27795 passed in 84 s (`tests/itf1788`) + 5535 in 747 s (the rest) = 33330, 2026-09-28 at `511056d` (the merged code; the doc commit after it changes docs and one comment only). M16b's exact pass puts 9549 more items in `tests/itf1788` (27795
  there on its branch, 2026-09-28); the whole merged tree collects 33330 items in one process
  (`--collect-only`, 2026-09-28)
* M13 done: every statement of the 19 itf1788 files runs (9542 vectors of 111 ops, 0 skipped, pinned
  by `tests/itf1788/test_itf1788.py::test_nothing_is_skipped`); 185 divergence keys, 0 unknown
  failures (2026-09-27; regenerate with `tools/itf1788_census.py`). the 1788 layer's exact pass
  (M16b): all match but 104 vectors under 94 rows (2026-09-28; the census's last line). M14: fuzz
  job and flint oracle built, never run on GitHub

## open items (ranked)

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | M14-run | the fuzz workflow's first green run on GitHub; record its example count and time, dated. **the default is ×10 since 2026-09-27** (the owner's choice, to keep runs cheap; `fuzz.yml` and `tests/conftest.py`, `timeout-minutes: 180`), because ×100, the old default, does not fit the 350-min timeout. measured locally 2026-09-27 at `dbec908` (the workflow's command, `FUZZ_MULTIPLIER=10`, python 3.13, shared laptop under another session's load): `22166 passed in 5036.97s` (1 h 24 min), no failures; it did not find the H2' example. linear fit through the ×1 gate (691 s non-vector, same day) and ×10 (4971 s), itf1788's 66 s held fixed: ×100 ≈ 47800 s (13.3 h); ×1 fuzz of `test_reverse.py`/`test_orders.py` scaled 12.0× to ×10 (per test 9.1-23.6×), so slightly superlinear, ≈ 16 h. with 25% headroom (≤ 262 min) ×25 fits (≈ 204 min), ×30 is the edge. laptop timings, not the runner's; the old 2.2-3.4 h estimate is withdrawn. since M16e the run also fuzzes the backend differential: `tests/test_backend.py` alone 735 passed in 387.5 s at ×10 locally (2026-09-28, loaded), about 5400-5600 s for the whole run with the last ×10's 5037 s, against the 180-min timeout | blocked: GitHub has only `ci` registered (checked 2026-09-27, `gh workflow list --all`); `fuzz.yml` must reach `master` (the default branch) before `schedule` or `workflow_dispatch` can run it — the owner's call | plan §2 M14 "exit" and "**the fuzz job, built 2026-09-26**" |
| 2 | pown-huge | the library's `pown` of a huge integral exponent never finishes: it computes the exact `Fraction(u) ** n`. `OutwardMultiInterval(0.5, 1) ** (2 ** 31 - 1)` and `ieee1788.pown(Interval(0.5, 1), 2 ** 31 - 1)` ran past a 30 s timeout where `2 ** 20` took 0.2 s (2026-09-28, loaded laptop), and `O(u) ** 2 ** 60` does not finish for u in {2, 2.0, 1e300} (a plain `M(2) ** 2 ** 60` too). inherited from `04946af`; the layer's `pown` and `**` are new public paths to it. fix: stop at the exponent where the result saturates (the overflow or underflow without the exact power); then M16d's B1 pin for r = `2.0 ** 60` (`tests/test_autodiff.py::test_pow_integral_exponent_derivative_is_exact`, u = -1 today) can take u in {2, 1e300, 1e-300} | ready (found by the M16b review, F4, and by M16d) | plan §2 M16b review (F4), M16d record ("what the build found") |
| 3 | M14-breadth | fuzz where it is thin: `tests/test_extreme_floats.py` extended to the functions, `minimum`/`maximum`/`fma`, `%`, `//` and `OutwardMultiInterval`; more `@given` in `test_outward`, `test_steps`, `test_fmt`, `test_applicator` | ready | plan §2 M14 "**breadth where fuzz is thin**" |
| 4 | newton-width | `newton`'s `width <= piece.wid() / 2` (`intervals/solver.py`, M15) is int true division, so an exact piece wider than the doubles would raise `OverflowError` once a step narrows without proving; not observed (a linear `f` is proved at the first step; `x ** 2 - 9 * 10 ** 800` over `[10 ** 400, 10 ** 401]` with `max_steps=10` did not finish in 2 minutes, 2026-09-28: the exact fractions grow). `solve`'s copy is already `2 * width <= _width(box)` (M16a review F2) | ready, small | plan §2 M16a review (soundness F2) |
| 5 | Q6-shift | port v1's `<<` and `>>` (owner 2026-09-26: "for sure"); choose the meaning on real sets when built (`A * 2**n`; `>>` exact or floored). when they land, `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers` derives them and goes red until `numpy_compat.py::_OPERATORS` gains `left_shift`/`right_shift` (M16d) | ready | plan §4 (the `<<`, `>>` row); `v2-plan.md` "2026-09-26 revision: owner answers" |
| 6 | layer-numpy | the 1788 layer's numpy rule: `ieee1788.Interval` keeps `__array_ufunc__ = None`, commented "the package's rule for every type" (`intervals/ieee1788.py`), which M16d made stale by giving `MultiInterval`, `DecoratedInterval` and `Dual` numpy's hook. the comment is fixed at the merge (2026-09-28: "the layer has no ufunc hook"); what stays is the rule: keep refusing (the default, as built) or give the layer the hook | owner's call, small (the two streams were built in parallel) | `v2-plan.md` "the 1788 layer" (the operators bullet) and "numpy"; plan §2 M16d |
| 7 | T1 | a reusable sabotage engine in `tools/sabotage.py`: M13's sub-tasks wrote the same ~30-line loop nine times (copy the file, apply one replacement that must match exactly once, clear `.hypothesis`, run pytest with a timeout for hangs, restore, `filecmp`, log a line), each with its own table of breaks. the break tables are per task and not worth keeping; the engine is. M16c found a hazard the engine must avoid: a same-size break restored within the same second as the broken write left python running the broken `.pyc`; clear `__pycache__` before and after each break, run with `PYTHONDONTWRITEBYTECODE=1`, restore with `copy2`, start with a control row on the intact code (H3's template `.scratch/h3/sabotage.py` has the hazard; M15's table was not re-checked for it) | idea, not scheduled (from the `.scratch/m13` audit, 2026-09-27) | plan §2 intro (sabotage rule); plan §2 M16c ("the sabotage harness ran stale bytecode") |
| 8 | evaluate-box | a pure speed change noted by M16e's design: `applicator.evaluate_box` evaluates a float corner's exact value three times under `OUTWARD`; passing `fn`'s value into the hook would cut it to one, maybe worth as much for arithmetic as the backend, with no dependency | idea, not scheduled (M16e, 2026-09-28) | plan §2 M16e; `v2-plan.md` "elementary and step functions" (the backend) |
| 9 | Q6-rest | `random_multi_interval`, a public `apply()`: to-do, undecided whether to port | owner's call, later | plan §4 (their rows) |
| 10 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | when everything is fully done (owner 2026-09-26); M16e's backend is opt-in and can ship in 2.0 as is (Q16(f)) | plan §2 M11; D5, D17 |
| 11 | M8 | the time layer on the v2 class | on hold, no rush (owner 2026-09-26); D4 recommends (a), Fraction seconds under a thin wrapper | plan §2 "M8 `time_interval.py`"; D4 |
| 12 | H4 | delete `archive/v1/` | after v2 is stable (owner 2026-09-26) | plan §2 M10 (last bullet before "done") |

## open questions for the owner

* **Q9 `mulRevToPair`'s decoration.** 1788 decorates the pair's first interval as the decorated
  division `c / b` where `0 ∉ b` (6 com, 41 dac, 5 def in `libieeep1788_mul_rev.itl`), but its
  `mulRev`, the hull of the same set, trv. ours is one op, `mul_rev`, always trv (sound: trv claims
  nothing), so 52 vectors are rows under "decoration expectations" on the decoration alone (the set
  must match, `tests/itf1788/test_itf1788.py::DECORATION_ONLY`). add a pair op with 1788's
  decoration, or keep the rows? built as the conservative reading (plan §2 M13 "exit for M13").
  **default built in the 1788 layer, pending Q13 (c)**: `ieee1788.mul_rev_to_pair` (M16b) is 1788's
  pair with its decoration; the library's `mul_rev` stays one op, trv, and the rows stay
* **Q10 the constructors' outward pass.** the 201 interval-valued vectors of the four 1788
  constructors (`b-`/`d-textToInterval` 91 each, `b-numsToInterval` 10, `d-numsToInterval` 9) run in
  the plain pass only: they have no interval operand, so an outward item would repeat the plain
  call. give the constructors a class argument (`OutwardMultiInterval`, 1788's binary64 hull; new
  API), or is the plain pass enough? kept as built (plan §2 M13g "review"). **default built in the
  1788 layer, pending Q13 (d)**: the layer's pass runs the constructors in binary64 (M16b); no class
  argument on the library's constructors

* **Q11 M15's choices (D19)**, built as the session's defaults when the owner said "do h3 first":
  (a) `Dual`, `derivative`, `newton` and `Root` are public and exported from `intervals` (the
  2025-12 sketch named `autodiff.py` and `solver.py`), not newton "as a test" only; (b) newton's
  step runs only where decorations prove `f` C¹, else the piece is pruned and bisected; (c) the
  step is `mul_rev`, never `/`; (d) one variable only; (e) `tol=1e-10` absolute, `max_steps=10_000`.
  keep, rename, or narrow the public surface? (plan §0 D19; `v2-plan.md` "2026-09-27 revision: M15")

* **Q12 M16a's choices (D20)**, built as the session's defaults when the owner said "get the rest
  of h3 done" (plan §0 D20, §2 M16a; `v2-plan.md` "2026-09-28 revision: M16a"):
  * **Q12(a)** a jacobian as n passes with `Dual` untouched (default), or vector mode (a tangent tuple
    inside `Dual`, one pass, M15's chain rules edited)? measured only indirectly: a box costs n + 2
    calls of `F` either way and the library's ops dominate
  * **Q12(b)** names `gradient`, `jacobian`, `solve`, `RootBox` (default), or `newton_system` /
    `krawczyk` and a widened `Root`? folds into Q11 (keep, rename or narrow the public surface)
  * **Q12(c)** zeros on split faces that are not simple rationals stay unproved, with unproved slivers
    beside proved ones (default); or bisect off-centre (measured on the prototype: 1 to 3 of 4 face
    zeros proved before the simplest point existed). noise, not error
  * **Q12(d)** the simplest-point rule and the inflated krawczyk test are additions beyond H3's wording
    (a gradient, a jacobian, krawczyk), each measured to be needed (`v2-plan.md` "2026-09-28 revision: M16a"): keep
    (default)?
  * not a question, recorded: hansen and sengupta's uniqueness test not used; smear, the mean value
    prune and "any component halved" measured on the prototype and left out

* **Q13 the 1788 layer's choices (D21)**, built as the session's defaults when the owner said "get
  the rest of h3 done":
  * (a) **shape**: `intervals/ieee1788.py`, one class `Interval` for both flavours, snake_case names
    with 1788's camelCase in `NAMES`, builtins with a trailing underscore; not exported from
    `intervals` (`from intervals import ieee1788`). keep, rename, or export (`intervals.ieee1788`
    imported in `__init__`, one line)?
  * (b) **numbers of the empty set**: `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad` of `[empty]`
    raise `ValueError`, as the library's (D9) and the reductions (Q2), and the pass reads the raise
    as `NaN`; or return `nan`, 1788's own answer, in the layer only? (`inf`/`sup` of it return `±inf`
    either way)
  * (c) **Q9**: default built, pending Q13 (c): 1788's pair with its decoration is
    `ieee1788.mul_rev_to_pair`; the library's `mul_rev` stays one op, trv on decorated operands; the
    adapter's 52 `DECORATION_ONLY` rows stay (true of the library). close Q9 so?
  * (d) **Q10**: default built, pending Q13 (d): the constructors' binary64 run is the layer's pass
    (their vectors match through it but the 9 that are rows, 7 exact parsing and 2 no NaI); no class
    argument on the library's constructors. close Q10 so?
  * (e) **two answers to one 1788 name**: where 1788 and the library disagree the layer answers
    1788's way (cancellation's "no answer", `meets`, attained infinities dropped) and the library
    keeps its own. keep?
  (plan §0 D21, §2 M16b; `v2-plan.md` "the 1788 layer" and "2026-09-28 revision: M16b")

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

* **Q15 M16d's choices (D23)**, built as the session's defaults when the owner said "get the rest
  of h3 done" (plan §0 D23, §2 M16d; `v2-plan.md` "numpy" and "2026-09-28 revision: M16d"):
  * **Q15(a) the array API or the hook.** default **`__array_ufunc__` on the three classes**
    (a multi-interval is an element, not an array); alternative: an interval-array type exposing the
    array API standard's namespace, whose dtypes, elementwise bool `==` and float special cases all
    collide with the library's choices (a new type, not interop)
  * **Q15(b) foreign reals** (`np.longdouble` on linux, gmpy2's `mpq`/`mpfr`): default **the exact
    value**, a `Rational` exact by type (`mpq(1, 2)` is `Fraction(1, 2)`, as `Fraction(1, 2)` is),
    any other real exact where `float()` would round (a double stays the float); alternatives: refuse
    a foreign real that is not a double (`TypeError`), or keep `float()` (unsound for
    `OutwardMultiInterval`), or the value rule for rationals too (`mpq(1, 2)` a float)
  * **Q15(c) an ndarray meeting ours**: default **elementwise into an object array**, `==`/`!=`
    identity as before (`f == A` False; `np.array([A]) == A` False and `A in np.array([A])` False
    although the array holds `A`); alternatives: `TypeError` as before; elementwise `==` (numpy's
    convention, which changes what `f == A` and `A in f` mean today)
  * **Q15(d) numpy's names as methods**: default **no aliases** (the 1788 names are the library's),
    so `np.arcsin(object_array)`, `np.round(A)` and `np.around(A)` are TypeErrors; alternative:
    eight aliases (`arcsin arccos arctan arcsinh arccosh arctanh rint arctan2`) on the three classes,
    after which numpy's loops and the table agree except `square`
  * **Q15(e) `np.invert(A)`**: default **the complement `~A`** (numpy's `invert` is the `~` ufunc,
    and `np.bitwise_and/or/xor` must mean `& | ^` for `np.int64(1) | A`); alternative: TypeError for
    the explicit unary call only
  * **Q15(f) `fmin`/`fmax`**: default **not mapped** (TypeError): their point is a nan operand, which
    the library refuses; alternative: `minimum`/`maximum` with `fmax(A, nan) is A`
  * **Q15(g) numpy in the `[test]` extra**: default **no** (CI installs numpy beside it; the numpy
    tests skip without it; the README section is prose); alternative: add it, and write the README
    section as doctests
  * **Q15(h) both operands ours in a method ufunc** (`hypot minimum maximum arctan2`): default **the
    subclass decides, as for the operators** (`np.hypot(M, O)` is `O.hypot(M)`, outward in either
    order; `np.arctan2(M, O)` takes y as an `OutwardMultiInterval` first), found by the M16d review;
    alternative: the first operand's method and class (as `M.hypot(O)` called directly is, which
    rounds to nearest and can miss the true value). with no subclass between them (`M` and
    `DecoratedInterval`) the first operand's method either way

* **Q16 M16e's choices (D24)**, built as the session's defaults when the owner said "get the rest
  of h3 done" (plan §0 D24, §2 M16e; `v2-plan.md` "elementary and step functions" (the backend)
  and "2026-09-28 revision: M16e"). none blocks H1; only (a) and (f) would change what a user
  without `INTERVALS_BACKEND` sees:
  * **Q16(a) automatic or opt-in.** built: opt-in. unset is the pure path; `INTERVALS_BACKEND=auto`
    takes gmpy2 when it imports (`2.3 <= version < 3`), `gmpy2` forces it. alternative: `auto` as the
    default, which gives every user who has gmpy2 (sympy's and mpmath's often do) the speed, and puts
    their MPFR build on the path unasked
  * **Q16(b) public surface.** built: the variable and the `[fast]` extra (a new name) only;
    `intervals.backend.name()` importable, not exported from `intervals`; no setter (it would read like
    the ambient rounding mode the plan rules out). alternative: export `backend_name()` for bug reports
  * **Q16(c) the non-dyadic points.** built: pure (a user's `Fraction(1, 3)`, `log` to a base,
    `pow_rev2`'s `log_t v`, `acoth`, `rootn` with n < 0, the periodic reverse ops' `k pi + f(v)`),
    except atan, acot, atan2's angles and the hook's mixed operands (native, one rounding).
    alternative: a second part, an mpfr ziv loop for a monotone f at a bracketed x; not measured
  * **Q16(d) gmpy2 in `[test]`.** built: yes, so the differential never skips, pinned `<3` (the
    review, 2026-09-28): `auto` takes only `2.3 <= version < 3`, so an unpinned gmpy2 3 on PyPI would
    turn `test_env_var`'s auto row red in every CI job; `::test_the_test_extra_installs_what_auto_takes`
    keeps the pin equal to `backend.FLOOR` and `backend.CEILING`. `[fast]` stays unpinned (a user's
    environment is not narrowed; forced takes gmpy2 3, `auto` does not). alternatives: gmpy2 only in CI
    jobs that ask for it, with `tests/test_backend.py` skipping locally (a test that passes by
    skipping); or `[fast]` pinned `<3` as well
  * **Q16(e) CI and fuzz.** built: no workflow change; every gate job runs the whole suite on the pure
    path and `tests/test_backend.py`'s differential in-process, `fuzz.yml` likewise (so its ×10 run
    fuzzes the differential too: `tests/test_backend.py` alone takes 387.51 s at x10, 2026-09-28,
    loaded, which with the last whole x10 run's 5037 s still fits the 180-min timeout).
    alternative: a gate job with `INTERVALS_BACKEND=gmpy2` (the whole suite on MPFR in CI, as this
    build ran it once locally), and a selection assert in each job
  * **Q16(f) 2.0 or later.** the item sat under `v2-plan.md` "later (not in v2.0)". built as an opt-in
    that changes nothing unless selected, so it can ship in 2.0 (H1) as is; alternative: keep it out
    of the 2.0 release notes until Q16(a) is answered
  * not questions, recorded: `round_rational` has no gmpy2 path (the design measured 1.0-2.1x on a call
    of a few µs, 2026-09-27); `floor_over_pi` and `compare` stay pure (they decide integers and signs),
    which is why `.sin()` gains nothing at set level

Q1-Q8 answered 2026-09-26 (`v2-plan.md` "2026-09-26 revision: owner answers to the open
questions"); D18 (M13's two proposed categories, the exact-com rows, `set_dec`) answered 2026-09-27
(`v2-plan.md` "2026-09-27 revision: owner answers on M13's proposed categories (D18)").

## still owed

* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (plan §2 M13d sabotage). recorded, not scheduled
* M16 is not pushed: none of its new tests (`tests/test_gradient.py`, `tests/test_solve.py`,
  `tests/test_ieee1788_layer.py`, `tests/itf1788/test_ieee1788.py`, `tests/test_numpy_compat.py`,
  `tests/test_backend.py`, and M16c's and M16d's additions to `tests/test_relations.py` and
  `tests/test_autodiff.py`) has run on CI (python 3.11-3.14) or under the fuzz profile; the local
  gates are the only evidence. the long double half of
  `tests/test_numpy_compat.py::test_longdouble_is_exact` discriminates only where
  `np.finfo(np.longdouble).nmant > 52` (CI's linux, never this laptop): its first CI run is its
  first real run
* M16a: a constructed n = 3 system with two zeros costs more than 120 s, so n = 3 is covered by one
  constructed zero and the sphere only, with no random n = 3 test; whether a faster jacobian
  (Q12(a)) or a tighter form of `F` would change that is not measured. the natural path to a split
  of a box already proved unique was not found (pinned by a monkeypatched `_krawczyk` only). a
  continuum costs up to 2n + 1 output boxes per box of width `tol` (review F3); whether to output it
  differently (one box per connected unproved region) is not asked, Q12 has no item for it
* sabotage rows red in their first run were not re-run after the closing tests were added (M16a:
  the four closing tests and the review's, which only add red paths); M15's table was not re-checked
  for the stale-bytecode hazard M16c found (T1)
* M16b: the adapter's docstring (`tests/itf1788/test_itf1788.py`) does not yet name the third pass,
  and `_PAIR_DECORATED_AS_DIVISION`'s reason does not name `ieee1788.mul_rev_to_pair` (left alone so
  the build does not pre-empt Q13 (c); one clause each once Q13 is answered). the layer's per-call
  `warnings.catch_warnings` is not thread-safe on python 3.11-3.13 (as `decorated.py::_quietly`);
  recorded, not addressed. the pass imports the adapter a second time as
  `tests.itf1788.test_itf1788` (the vectors parsed twice, 7.5 s cold, 2026-09-27); accepted. one
  leftover gate call of M16b's verifier ended rc=1 with its output lost (its log overwritten
  mid-run), on the tree less one test edit; both later runs of that call green; recorded in plan §2
  M16b, not explained
* M16c: the other relations over cut tuples in `relations.py` (`before`, `adjoins`, ...) also read
  normalized operands and do not assert it; only `allen_relations`, whose wrong answer would be
  silent and partial, does. the methods are unaffected
* M16e: the whole suite under `INTERVALS_BACKEND=gmpy2` exists only as the build's one local run
  (2026-09-28); no CI job runs it (Q16(e)). the backend is verified only with gmpy2 2.3.1 / MPFR
  4.2.2 on windows (python 3.13); CI's linux jobs run `tests/test_backend.py` with the PyPI wheel, no
  other MPFR build has been run. free-threaded builds untested (the three contexts are shared module
  objects; `backend._use` is a global). the speed numbers were taken beside four other streams' runs:
  a quiet-machine re-measure (`tools/backend_speed.py`) is owed before any number goes into the
  README. gmpy2 3: `auto` and `[test]` stop below it; when it ships, run the differential against it
  before `backend.CEILING` and the `[test]` pin move together
  (`tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` keeps them equal)
* `.scratch/h3b/` in the main checkout holds M16's designs and critiques (`design/`), review probes
  and harness logs, cited by the plan's M16 records as gitignored sources; audit it before recycling
  (the records took what the builds used from it)

## session log (newest first)

* **2026-09-28** M16, H3's second part: the owner said 2026-09-27 "get the rest of h3 done",
  superseding 2026-09-26's "numpy and gmpy2/mpfr recorded, not now". a design workflow first (five
  designers, one per stream, and five adversarial critics), then a build workflow per stream in five
  worktrees (a builder, three read-only reviewers with the lenses soundness, sabotage audit and
  spec/regression, a fixer that reproduced each finding first, and a verifier), each stream on its
  own branch off `v2` at `04946af`; merged into `h3-merge` without conflicts. built: M16a
  `gradient`, `jacobian` (n passes, `Dual` untouched) and `solve`, `RootBox` (krawczyk proves,
  gauss-seidel with `mul_rev` narrows; the direction tag argued not needed in n variables); M16b
  `intervals/ieee1788.py`, the 1788 layer, with `mul_rev_to_pair` and a third, exact conformance
  pass (all match but 104 vectors under 94 rows), Q9's and Q10's defaults built in it; M16c
  `allen_matrix` and `allen_relations`; M16d numpy interop (`intervals/numpy_compat.py`, numpy
  optional), which also fixed a soundness hole M15 shipped: `Dual ** r` computed `r - 1` in r's own
  float arithmetic (`Dual.variable(O(1e300)) ** 0.1` missed its derivative, bare and decorated;
  `Dual.variable(O(-1)) ** 2.0 ** 60` had its sign flipped); M16e the gmpy2/mpfr backend
  (`backend.py`, `_gmpy2.py`), opt-in and the pure path by default, the same doubles by differential.
  the reviewers marked 4 findings blocking, all M16a's (a false C¹ claim in `gradient`'s docstring
  and the design; three unpinned rules in the solver, S1-S3), each fixed and pinned; every other
  finding was minor, fixed and pinned or deferred with evidence (pown-huge). the reviews found no
  wrong answer in the solver, the layer or the backend, none through `MultiInterval` in the allen
  matrix (one in `relations.allen_relations` on out-of-order cut tuples, now asserted), and no
  unsound result in numpy's (a method ufunc on two of ours took the first operand's class, now the
  subclass's, Q15(h)). two incidents: a builder killed another stream's sabotage harness by
  matching its command line, and that stream's file was restored from its `.orig`; a duplicate
  builder was spawned by an orchestrator message and stood down, no damage. gate on the merged tree
  green: 27795 passed in 84 s (`tests/itf1788`) + 5535 in 747 s (the rest) = 33330, 2026-09-28 at `511056d` (the merged code; the doc commit after it changes docs and one comment only). not pushed. the streams' records (`h3-records/`) folded into the docs and removed.
  records: plan §0 D20-D24, §2 M16; `v2-plan.md` "current design" and five 2026-09-28 revisions;
  README; new questions Q12-Q16, new items pown-huge, newton-width, layer-numpy, evaluate-box

* **2026-09-27** M15, H3's first part: the owner asked for H3 first; built its suggested first pick,
  forward-mode autodiff over sets (`Dual`, chain rules over every elementary method, arb's taylor
  series as the oracle) and interval newton over multi-intervals (`newton`: the step is `mul_rev`,
  so a derivative set holding 0 splits a piece in one step; C¹ proved by decorations; uniqueness
  proofs; exponent splits for wide pieces). 30 breaks sabotaged, 23 red at once, the 7 green ones
  (3 gaps, 4 cost rules) closed with tests and re-run red. the gate's randomized run found an old
  test-oracle bug in `tests/test_kernel.py::test_normalize_is_canonical` (a midpoint underflowing
  onto an end), fixed and pinned. gate 18246 passed in 67 s + 4088 in 482 s (22334). not pushed. records: plan
  §0 D19, §2 M15; `v2-plan.md` "the solver stack" and its 2026-09-27 revision; new question Q11

* **2026-09-27** (`69a667a`..`a1d45a9`, pushed) H2' done: `v2` pushed at `dbec908`, CI red on one
  test-oracle gap, fixed in `d7e46c2`, CI green at `a1d45a9` (plan §1). fuzz ×10 measured locally,
  ×100 does not fit, the default is now ×10 (M14-run). `.scratch/m13` still not recycled (the
  VisualBasic recycle call refused it; `.scratch/fuzz` went through)
* **2026-09-27** M13 finished, as an orchestrated build: M13e (reverse ops, `intervals/reverse.py`:
  sqr, abs, pown, cosh, mul, sin, cos, tan, pow_rev1, pow_rev2) and M13g (`DecoratedInterval`,
  `intervals/literals.py`'s 1788 text syntax, the four constructors, `set_dec`, the signals
  `UndefinedOperationError` and `PossiblyUndefinedOperationWarning`, propagation through every op)
  were built in parallel worktrees, each by sequential builders with properties and sabotage, then
  three independent reviewers each (math differential, sabotage audit, spec) and a fixer that
  reproduced each finding first. one library bug found and fixed: M13e's rational `log` rewrite was
  cubic in the operand's size (pinned). then merged, the decorated reverse vectors wired (481, 174
  pairs), the exit test added (`::test_nothing_is_skipped`; dropping an op from `OPS` turns it red).
  the owner approved both proposed categories and kept the exact-com rows and `set_dec`'s demotion
  (D18). the adapter's `::is_decorated` missed a decoration on the result alone (0 vectors run
  differently; the census undercounted 1521 for 1624), fixed and pinned. `.scratch/m13` audited
  (174 files): 2 notes' unrecorded review results transcribed into the M13e record, then the
  directory left in place: neither Recycle Bin route worked from the session, and the audit found
  nothing else in it untracked, so `.scratch/m13/` can go to the Recycle Bin by hand. records: plan §2 M13e, M13g, "exit for M13"; `v2-plan.md` 2026-09-27 revisions

* **2026-09-26** the owner answered Q3-Q7 and H1-H5 (`v2-plan.md` "2026-09-26 revision: owner
  answers to the open questions"); then, after 1788's context, Q1 (`UndefinedOperation` and `IntvlPartOfNaI` raise,
  `PossiblyUndefinedOperation` warns), Q2 (keep `ValueError`) and a new Q8 (no NaI at all), so
  M13g is unblocked. Q6's `<<`/`>>`
  became an open item; the v1 README's leftovers moved to `references/todo-from-v1-readme.md`
  (H5 done). gate green (17346 in 380 s), then `v2` pushed (H2 done); CI run 36219282601 at
  `2f3a895` all 8 jobs green, the first CI on M12 and M13
* **2026-09-26** M13d: 1788 `pow` through `**` (D11), `__rpow__`, and expm1, log1p, cbrt,
  rootn, hypot, cot, sec, csc, acot, coth, csch, sech, acoth, correctly rounded in pure python, with
  the decimal and arb oracles and set-level properties. the 1939 vectors of those 14 ops, already
  vendored at M13a, now run: all match in both passes, no new row. a sabotage harness
  (throwaway, results in the plan's record) turned 23 of 25 breaks red; the two green ones only save ziv iterations.
  three first-thin catches were thickened with an example or a property. records: plan §2 M13d,
  `v2-plan.md` "2026-09-26 revision: M13d"

* **2026-09-26** HANDOFF.md created; open items and owner questions moved here from both plans.
  the census script behind the itf1788 counts moved from `.scratch/` to `tools/itf1788_census.py`
* **2026-09-26** (`814b302`..`bcee889`) M13a (vendored all 19 files of oheim/ITF1788, new
  parser and adapter), M14's fuzz job and flint oracle, then M13h, M13b, M13c, M13f; a review pinned
  cancel's vacuous `-inf`. fuzz found a test-oracle bug (mixed pairs rounded twice), and the gate an
  `exp10` test bug; the library had none. records: plan §2 M13a, b, c, f, h and M14
* **2026-09-25** (`62cc80c`..`c8d843d`) M13 and M14 planned; the owner settled D9-D17. before that
  (`53400d4`..`4e98bdc`) M12: the elementary and step functions, min/max/fma, outward rounding, 5
  more itf1788 files; exhaustive modulo re-run, 0 mismatches. records: plan §2 M12, M13, M14
* **2026-09-25** (`c6cfe14`..`d232b78`, pushed) M10 (v1 archived, new README, M11 backlog),
  M11's restored harnesses, CI; `d897d77` (not pushed) records the first CI run. records: plan §1,
  §2 M10, M11. M1-M9: 2026-09-23/24, see plan §2
