# session log (older entries)

`HANDOFF.md`'s session log keeps its newest 8 entries; older ones move here, unchanged, newest first
(the rule is in `HANDOFF.md`'s preamble). pointers in them to `v2-plan.md` and `v2-implementation-plan.md` ("plan §2")
mean `docs/archive/v2/`; decisions they mention are in `docs/decisions.md`.

## entries (newest first)

* **2026-10-04** the v1 parity audit (the owner: "send out a bunch of agents to read and run code, to verify that
  everything in v1 is possible in v2"). one workflow, 44 agents: ten auditors, one per slice of `archive/v1/`, ran v1
  and v2 side by side (839,971 cases, compared as sets); two skeptics per claimed gap (reproduce; records); a coverage
  critic (17 uncovered items, 8 probes that could not fail) and eleven second-round auditors. 558 rows, no v2
  regression, every v1 capability possible in v2. the session reproduced each surviving item itself (the to-do diff;
  the flag, truthiness, `DateTimeInterval(None, t)` and no-bound `start_closed=False` cases). restored the three lines
  `references/todo-from-v1-readme.md` had dropped; recorded the rest in plan §4 ("the parity audit") and Q22; the
  reports, probes and skeptics' notes moved to `references/v1-parity-2026-10-04/` and `.scratch/v1-parity/` deleted.
  docs only, the gate's code unchanged. Still owed: Q22; the push (x10 prepush) of `c0e6777..` HEAD
* **2026-10-04** M8 built (the owner: "build the time intervals"; subagents on opus, the owner: "stick to opus
  unless you really need fable"). a builder in the worktree `../intervals-m8` (`fd79070`); three read-only reviews
  on private snapshots (spec: nothing blocking; soundness: numpy `timedelta64` read as an int (pre-existing in the
  numeric class), pandas-left `%` too wide, a DST fold's point lost, aware read-outs overflowing near the range ends;
  sabotage: 19 of 26 new breaks green); the session reproduced the blocking ones; a fixer (`f617971`..`5996c2b`)
  fixed or documented each and closed every gap: 77 breaks red. the session re-checked D30 point by point, the
  numeric diff and the order rule, then fast-forwarded `master` to `5996c2b`; worktree, branch and `.scratch/m8`
  removed. gate on `master`: 27795 + 6477 passed (2026-10-04). Still owed: Q21; the push (x10 prepush) of `c0e6777..5996c2b`; the
  CORE-MATH full check is the owner's call (`cuts.py` changed: a refusal of numpy durations only)
* **2026-10-04** M8's choices (the owner: "send a fable agent to noodle over it", then "I'm okay with the
  recommendations"). a read-only agent wrote one recommendation per choice and seven smaller decisions
  (`references/m8-choices-2026-10-04/`); the session re-ran every claim they rest on with its own probe (each held,
  listed in that README), then recorded them: D30, plan §2 M8, `v2-plan.md` decision log. meanwhile the prepush of
  `0dc80eb` ran green under a haiku babysitter (fuzz-x10:itf 27795 passed in 92 s, fuzz-x10:rest 6184 in 4358 s;
  the docs commit after it kept the src id) and `c55ce20..5888c6e` was pushed (the owner's go); a babysitter on
  `tools/ci_watch.sh 5888c6e`: CI run 37187049250 green; fuzz run 37187049245 red, `1 failed, 33978 passed in
  3429.58s`, `test_mul_rev_by_a_point` on a subnormal y: a test oracle that predated D26 (the library's `[-inf]` is
  D26's), fixed and pinned (plan §2 "fuzz-mulrev-point"), not pushed. then M8's build began in the worktree
  `../intervals-m8` (branch `m8`)
* **2026-10-04** the owner asked how certain the library's correctness is (answered from the records: independent
  oracles, ITF1788, CORE-MATH, the exhaustive grids, fuzz and sabotage; the weak spots are the still-owed list and
  the discovery rate), then what v1 has that v2 lacks. a census of v1's class members against v2's found nothing
  but renames, deliberate drops and three narrow gaps; the owner: `merge`'s k-overlap mode and parsing were
  artifacts, `random_multi_interval` a test helper, `apply()` not public for now, and the shifts dropped (no use
  case; the plausible one, fixed-point code over sets, wants python's floor; no integer ranges, none in 1788
  either). `0513109` reverted, the pin `test_no_shifts` red on the old library (checked in a `git archive` copy
  with its own `pytest.ini`: the repo's `pythonpath` otherwise imports the live package and the check passes
  vacuously). then the owner asked whether v1 can go: yes, but for the time layer; what still reads v1 is in
  row H4. the owner: port the time layer first (M8 now row 1, H4 row 2). Still owed: the push (prepush is an x10
  fuzz); `kernel.overlap_count` is now used by its test only
* **2026-10-04** m14b-open's quadratic parse (the owner: "optimize the regex so it's not quadratic"). a first
  agent with `isolation: worktree` vanished with no worktree, notes or commit; the session made the worktree
  itself (`../intervals-regex-linear`, branch `regex-linear`) and a second agent measured, fixed, pinned and
  sabotaged (plan §2 "M14-breadth", "the parse fixed"); the session re-checked the diff, the timing and the new
  test, and fast-forwarded `master` to `6450502` (the owner's go). worktree, branch and `.scratch/regex-linear/`
  removed. the owner also asked what porting the time layer (M8) takes: not a copy-paste, about half of
  `archive/v1/time_interval.py` ports as is; the rest is new (exact Fraction seconds for v1's float
  `timestamp()`, which reads naive datetimes in local time; v1's `.interval.endpoints` (20 uses) and in-place
  mutation have no v2 counterpart; infinite ends; no tests in v1). before building: the owner's call on
  timezones (naive as wall clock? mixing aware and naive raises?), what an infinite end reads as, and whether the
  end-of-day snap (23:59:59.999999) stays or becomes a half-open next midnight. Still owed: the push (prepush is
  an x10 fuzz)
* **2026-10-03/04** the open owner questions (the owner: "what questions are open for me", then "call in fable subagents to think about each", then "i'll accept everything fable said, update accordingly", then "when all the agents are done ... run the full gate and fuzz and then push"). nine read-only agents wrote one report per group of questions (`references/owner-questions-2026-10-03/`, `09435ca`; the session re-ran the key claims, listed in its README). decisions recorded (`a1a5716`: decision log, D19-D24 marked, D27-D29). four build streams in worktrees, merged (`87c9319`, `1db1b15` CLAUDE.md's push clause for `gate:gmpy2`, `9896284`, `87e6ea3`); their records in `streams/`. at the merge the session changed one thing: the pown stream had built the nearest class's over-limit pown as rounded to nearest, from a summary line that misread the report; the report's (c) is the enclosure in both classes, now built and pinned (4 tests red on the stream's `ops.py`). gate 27795 + 6234; prepush of `c55ce20` green (x10: 27795 in 59 s + 6234 in 6096 s; `gate:gmpy2` 34029 in 754 s); pushed `912558b..c55ce20`; the babysitter, checked by the session: CI run 37147322181 green (gate 34029 on python 3.12, 3.13, 3.14 in 464, 490, 456 s; `gate-gmpy2` 34029 in 337 s; the exhaustive jobs) and fuzz run 37147322232 green (`34029 passed in 3458 s`, x10). the CORE-MATH full check is the owner's call (the scalar evaluator changed)
* **2026-10-03** gate and push (the owner: "Gate and push"). the prepush of `e5cb396` went red: `fuzz-x10:rest` 30 failed, 24 errors, read as MOVED. 27 failures and the 24 errors were the laptop (child processes and `git` exiting `0xC0000142` under another session's concurrent hypothesis run); three were test oracles that assumed a float stays a float, the mixed one-point piece `[2, 2.0]` read as its exact 2 by the rounding-hook check, and python's double-rounded `float - Fraction` past an exact end in the identity-rounding check. both oracles fixed and pinned, the library unchanged (plan §2 "fuzz-mixed-points"). prepush of `8e33d7e` green (x10: 27795 in 53 s + 6033 in 5594 s); pushed `233fdd4..8e33d7e`. the babysitter: CI run 37098878518 green; fuzz run 37098878528 red, `1 failed, 33827 passed in 3235.17s`: a library bug, `rootn((10 ** -30, 1.0000000000000003e-30], 5)` raised in the exact class (an exact end beside a float end rounded to nearest past it), and `pown_rev` the same; fixed by the applicator's rule (the piece between the two values, each keeping its flag) and pinned (plan §2 "fuzz-rootn-crossed", Q20). gate 27795 + 6035, prepush of `912558b` green (x10: 27795 in 53 s + 6035 in 5459 s); pushed `8e33d7e..912558b`; the babysitter, checked by the session: CI run 37107899644 green (`33830 passed` on python 3.12, 3.13, 3.14 in 495, 430, 361 s) and fuzz run 37107899636 green (`33830 passed in 2051.77s`, x10)
* **2026-10-03** overnight fuzz (the owner: "run the fuzzer ... maybe 50x", missing fuzz tests first). a census agent listed public behaviour no @given test randomised and found a soundness bug: `OutwardMultiInterval.expand` rounded a moved float end to nearest (`O(0.1, 0.2).expand(1)` was `[-0.9, 1.2]` with 0.2 + 1 > 1.2), fixed in `4447f6f` (expand is the set plus [-d, d], the class's outward sum). new or widened @given tests, each sabotaged red: the domain guard at random points, rootn vs arb to degree 10**4, decorated set ops, 1788 pown/rootn/pownRev exponents to +-40, `overlap`, `is_member`, Dual powers and rootn, `gradient` vs `jacobian`, the predicates, `positive`/`negative`/`finite`, `A[a:b]`. fuzz x50 on all of it: itf 27795 passed in 72 s, rest 2 failed, 6021 passed in 27006 s (7 h 30 min): `test_steps.py::test_isotone[round]` and `[round_ties_away]`, a test-oracle bug (Q19; the session first misread it as a library bug, an agent showed it was the oracle and the session re-checked). not pushed
* **2026-10-02** CORE-MATH worst cases (the owner: outputs MPFR computed are not worth vendoring, the choice of inputs is; Lefevre's data has no licence, CORE-MATH is MIT and carries blocks of it). built: `008fa4a` the scalar evaluator refuses a point outside a domain (six functions hung, `acos(-2)` was 0.0; no public result changes, the set layer clips first); `3044197` `tools/coremath.py`, a vendored gate sample of 29,054 rows (`tests/test_coremath.py`, 12 s, sabotaged 4 ways) and a manual full check, never in CI or prepush (`CLAUDE.md`); its first run at `3044197`: 17,077,691 inputs, 51,233,073 calls, 0 mismatches, 2547 s at 4 jobs. gate green on both commits (27795 + 5929, then 27795 + 5978 passed). a review agent's second opinion and its sabotage table: `references/test-vector-sources.md` §3h; the build: §3i. not pushed
* **2026-10-02** pushed `93d9e3e..233fdd4` (the ledger, M14-breadth and its fixes) after `tools/prepush.sh` read
  the x10 verdict from the ledger and ran nothing (its first use as designed: a fuzz run earned before the commit
  counted after it). a babysitter agent on `tools/ci_watch.sh`: CI run 36989262364 green in 8 min 14 s (all 7
  jobs; the gate `33710 passed` on python 3.12, 3.13 and 3.14 in 351, 457 and 469 s; both exhaustive jobs) and
  fuzz run 36989262372 green, `33710 passed in 3637.44s` (x10), the third green fuzz run in a row
* **2026-10-02** the prepush of `6f9fd09` went red (`fuzz-x10:rest`: 4 failed; the ledger recorded FAILED while
  the background task reported exit 0, the trailing-`echo` trap): a mixed-type point lost a type in `repr`
  (library, fixed), the shared oracle's float `x ** -n` rounded twice as the library had (fixed), and the new
  nearest-ends oracle assumed rounded ends keep their order (fixed); each pinned red-on-old (plan §2
  "M14-breadth"). the fuzz x10 then ran on the fixed tree before its commit (the ledger keys by content):
  red again twice, on test oracles only (a too-narrow fix of the nearest-ends oracle; an overflow in a steps helper;
  the applicator hook at a split point, the m14b-open type quirk), each fixed and pinned; the fourth run green:
  fuzz-x10:itf 27795 passed in 56 s + fuzz-x10:rest 5915 in 5959 s = 33710 on `s:8f11144ba309` (2026-10-02)
* **2026-10-02** M14-breadth (the owner paused the push's fuzz run for it): six builders in worktrees, one
  per thin file, each property sabotaged (122 breaks, all red); merged by cherry-pick. the session verified and
  fixed five library bugs they reported, each pinned red-on-old: outward `floor`/`ceil`/`trunc` rounded their
  values to nearest (unsound past 2 ** 53), the step cap counted a shared value twice, nearest `x ** -n` rounded
  twice, and two fmt parse bugs (`[1/0]` raised ZeroDivisionError, `[] , [1]` refused); and one stale D26
  oracle in `tests/test_pow_rev.py` a randomized gate run can draw. what is left is row m14b-open. the owner
  asked whether ITF1788 is complete and what else exists: answered from the census (19 files, 9542 vectors,
  0 skipped, 185 rows in 7 categories) and `references/test-vector-sources.md`; vectors-ext unchanged. gate:
  gate:itf 27795 passed in 61 s + gate:rest 5914 in 687 s = 33709, green on `c:52ebf09347e6`
  (2026-10-02; gate:rest was 596 s before: the new properties cost about 90 s)
* **2026-10-01** the run ledger (the owner: "machinery to know whats run and not on the current code",
  adapted from the zanzibar repo's): `tools/gate.py` records each gate, docs and fuzz run against two
  content ids of the code (`src`, `code`), reports what is green here, and plans a push;
  `tools/prepush.sh` now runs only what it plans and exits with its verdict. `tests/test_gate_ledger.py`
  (45 tests) pins it; 13 sabotage breaks each red, after the first round found a bug in the tool itself
  (the log header read as the run's summary), fixed and pinned. record: plan §2 "run ledger". the
  owner answered the banner's question: docs-only pushes skip the fuzz (built in `ccc6c3f`). gate
  through the ledger: gate:itf 27795 passed in 63 s + gate:rest 5656 in 596 s = 33451, green on
  `c:b80c4e5d48d7` (2026-10-01). Still owed: the push (needs the owner's go; prepush is a full x10 fuzz)
* **2026-10-01** docs-only pushes skip the fuzz (the owner, 2026-09-30): `tools/prepush.sh` compares with
  `origin/master` and, when every changed file is `*.md` or under `references/`, runs only the changed
  READMEs' doctests (each path checked in a throwaway worktree; a broken README doctest turns it red);
  `fuzz.yml` skips such a push (`paths-ignore`). a repo skill, `testing` (`.claude/skills/testing/`),
  holds how to run each kind of test; `CLAUDE.md` keeps the rules and points at it. `.git/info/exclude`
  narrowed from `.claude/` to `.claude/worktrees/` and `settings.local.json`, so the skill is tracked.
  `9b58329` and `93d9e3e` pushed by the docs path; their runs green, checked by the session: CI run
  36800145512 (33406 passed on python 3.12-3.14 in 299-414 s) and fuzz run 36800145572 (`33406 passed in
  3123.63s`, x10, restored from the first green run's database), the second green fuzz run in a row. `ccc6c3f` (the
  tooling and the skill): `tools/prepush.sh` green, x10, 27795 in 53 s + 5611 in 4906 s = 33406,
  2026-10-01; not pushed, awaiting the owner's go
* **2026-09-30** M14-run closed, M14's exit met: fuzz run 36654816589 at `697abdd` green, x10, `33406 passed
  in 1765.10s`, the job 29 min 41 s, having restored the red run's saved database; CI run 36654816564
  green (33406 passed on python 3.12-3.14 in 345-427 s). a babysitter agent watched both
  (`tools/ci_watch.sh`); the session checked the conclusions and the cache restore. record: plan §2
  M14 "the exit's green GitHub run"
* **2026-09-30** fuzz-rev-inf closed (plan §2 "fuzz-rev-inf", D26): the owner weighed 1788 (intersect with
  `x`, then enclose) against python's rounding and chose (a): to nearest, a reverse op meets `x` before the
  rounding, so a part of the answer inside `x` that rounds wholly onto one double is that double, as a point
  (`reverse._keep_squeezed`): an end `x` excludes (the fuzz case, `[-inf]`), or a point of `x` at an open
  rounded end (`mul_rev(10, (1, 2), [0.1])`, a known loss M13e's tests worked around, now `[0.1]`). forward
  ops unchanged, documented (README "rounding", `v2-plan.md` decision log). four pins, each red with the fix a
  no-op; the float tests of mul_rev and the trig ops, which pinned the old order, now check D26 and that no
  piece of the outward result in `x` vanishes. there is no single list of where the library departs from
  1788 until the owner asked for one: README "departures from ieee 1788" (a census agent, every
  pointer checked). `tools/prepush.sh` at `697abdd` green, x10: 27795 in 53 s + 5611 in 4975 s = 33406,
  2026-09-30; pushed (`97d9824..697abdd`), a babysitter on `tools/ci_watch.sh`
* **2026-09-29** the new push procedure, first use: `tools/prepush.sh` at `97d9824` green, x10: 27795 passed
  in 53 s + 5611 in 4899 s (1 h 22 min) = 33406, 2026-09-29; `master` pushed (`8a4cc2f..97d9824`, the
  owner's go), a babysitter agent on `tools/ci_watch.sh`. the owner asked whether the 1788 set is
  complete: a research agent surveyed ITF1788's forks and other test data (kept as
  `references/test-vector-sources.md`; new row vectors-ext); the upstream errata are noted in
  `tests/itf1788/README.md`, whose stale `SKIPPED` paragraph is fixed
  the babysitter: CI run 36580954215 green (33406 passed on python 3.12-3.14 in 353-405 s); fuzz
  run 36580954134 red, 49 min, new row fuzz-rev-inf (the library's nearest class; the babysitter's
  root cause and fix were wrong, its reproduction right). the CI example database now saves and
  replays locally (checked by the session; a first replay attempt nested the copy and wrongly
  passed: CLAUDE.md now warns)
* **2026-09-29** cleanup and `master` (the owner: clean up, "make sure everything is on the master
  branch", run hypothesis again). a read-only audit of `.scratch/pown-huge/` and `.scratch/m14-run/`
  found three things tracked nowhere, re-checked and transcribed (`8a506f3`: plan "pown-huge",
  Q17, T1); the fuzz failure was already pinned. then `.scratch/` emptied to the Recycle Bin
  (pown-huge, m14-run, h3, h3b, m13, the audit's own report). why the Recycle Bin "did not work"
  before: sixteen orphaned `tail -f` watchers from 2026-09-26..28 sessions (monitors on sabotage and
  review logs) still held files and working directories in `.scratch/h3`, `h3b`, `m13` and two gone
  worktrees; stopped by PID (each checked to be a `tail.exe` on this repo's paths), after which the
  VisualBasic recycle call worked. a monitor's `tail -f` outlives its session: stop it by PID when
  done. worktree `../intervals-pown-huge` removed; gate green at `4f51e86` (27795 in 70 s + 5608 in
  829 s = 33403); `master` fast-forwarded to `v2` and pushed; CI run 36540572649 at `8a4cc2f` green
  (all 7 jobs; the gate 33403 passed on python 3.12-3.14 in 228-358 s, 2026-09-29); `v2` and
  `pown-huge` deleted. the fuzz rerun (run 36540588320) restored the first run's cache, passed
  `test_symmetry`, and found one test-oracle bug, fuzz-floordiv-overflow (`float()` of an exact
  floor past MAX in `tests/test_modulo.py`), fixed and pinned (red on the old oracle), not pushed.
  then (the owner: fuzz "fully autonomously or not at all"): no schedule; `fuzz.yml` on push to
  `master`; `tools/prepush.sh` (the fuzz run locally, before a push) and `tools/ci_watch.sh` (the
  babysitter's watch, tested on runs 36540572649 and 36540588320); a new `CLAUDE.md` with the push
  procedure. replaying run 36540588320's artifact locally replayed nothing: the fuzz profile had no
  database on CI (inherited from hypothesis's `ci` profile), so no GitHub run had saved an example;
  fixed and pinned (`tests/test_fuzz_profile.py`, red without the fix under a simulated CI). the
  `@reproduce_failure` blob was refused locally (hypothesis 6.167.1 here, 6.168.3 on CI). `fuzz-run`
  and its worktree deleted
* **2026-09-29** merged `pown-huge` into `v2` (`git merge --ff-only`, the owner's go; not pushed).
  fuzz-symmetry diagnosed and fixed: the library's `-`, which read a mixed exact/float point by its
  low cut, not the test; `-`/`+` now keep each cut's type. a first fix in the reverse ops (read a
  point at its low cut) was tried and dropped: it broke `test_trig_rev_symmetry`'s pinned example.
  pins red on the old `ops.py`; gate green: 27795 + 5608 = 33403 passed, 2026-09-29. item fuzz-symmetry closed (plan §2
  "fuzz-symmetry")
* **2026-09-29** M14-run: the owner chose a throwaway branch over `fuzz.yml` on `master`; branch
  `fuzz-run` (`8a8abf8`) pushed with a temporary `push` trigger, watched by a read-only haiku agent.
  run 36507253782: `1 failed, 33331 passed in 3216.59s`, `test_symmetry` on a mixed exact/float
  point under `pown_rev(., -7)`, reproduced locally at `7288e81`; new item fuzz-symmetry
* **2026-09-29** pown-huge, as an orchestrated build (the owner: "start work on pown huge"): two
  designers (binary powering with directed rounding; `exp(n log u)` through `elementary.rounded_pow`),
  an adversarial critic (chose the second: the first's cost grows with bitlen(n), 38 s at `2 ** 30000`;
  grafted a zero-base fix and a bounded descriptor name, both found by the critique), a builder on
  branch `pown-huge` in the worktree `../intervals-pown-huge`, three read-only reviewers (soundness,
  sabotage audit, spec/regression: 0 blocking, 11 minor), a fixer that reproduced all 11 first (9
  fixed and pinned, SND-1 deferred as Q17, F3 this HANDOFF edit), and a verifier (gate 33402 green at
  `3ed888f`). two nearest-class bugs found beside the hang: `M(-1.0) ** (2 ** 60 + 1)` was `[1.0]`,
  `M(0.5) ** 10 ** 400` was `[inf]`. the session re-probed infinities, poles and 3000 small-n outward
  powers against the exact Fraction (0 misses). not merged into `v2` (the fast-forward was held for
  the owner), not pushed. records: plan §2 "pown-huge"; `v2-plan.md`; new questions Q17, Q18

* **2026-09-28** pushed `v2` at `3aaf8f4` after a full local gate (27795 + 5535 = 33330 passed); CI run 36402681261 at `3aaf8f4` (M15 and M16's first, 2026-09-28): 7 of 8 jobs green, the gate 33330 passed on python 3.12-3.14 in 365-429 s; on python 3.11 `1 failed, 33329 passed in 393 s`, `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[longdouble-pow]`: the oracle's `Fraction(2 ** 60 + 1, 2 ** 60) ** (-inf, -2)` gave `[1.0]`. first misread as numpy 2.4 comparing a long double as its double (`42234f0` made the oracle decide exactly, and run 36406179185 failed the same way); the cause is CPython 3.11's `Fraction.__pow__`, which answers any non-rational exponent with `float(a) ** b`, so a Fraction base is rounded before the library's `__rpow__` sees it: on 3.11 `Fraction(1, 3) ** O(2)` is `(0.11111111111111109, 0.1111111111111111)`, missing 1/9, through `Dual` too (checked with a local 3.11.15; `+ - * / // % divmod` stay exact; 3.12 returns NotImplemented). nothing in the library can see it (its `__rpow__` gets a float), so the owner set python >= 3.12 (D25); `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11. the exhaustive jobs 19 s to 8 min 54 s. a CI babysitter agent (read-only) watched the run and extracted the failure
* **2026-09-28** M16, H3's second part: the owner said 2026-09-27 "get the rest of h3 done",
  superseding 2026-09-26's "numpy and gmpy2/mpfr recorded, not now". a design workflow first (five
  designers, one per stream, and five adversarial critics), then a build workflow per stream in five
  worktrees (a builder, three read-only reviewers with the lenses soundness, sabotage audit and
  spec/regression, a fixer that reproduced each finding first, and a verifier), each stream on its
  own branch off `v2` at `04946af`; merged into `h3-merge` without conflicts. built: M16a
  `gradient`, `jacobian` (n passes, `Dual` untouched) and `solve`, `RootBox` (krawczyk proves,
  gauss-seidel with `mul_rev` narrows; the direction tag argued not needed in n variables); M16b
  `multiinterval/ieee1788.py`, the 1788 layer, with `mul_rev_to_pair` and a third, exact conformance
  pass (all match but 104 vectors under 94 rows), Q9's and Q10's defaults built in it; M16c
  `allen_matrix` and `allen_relations`; M16d numpy interop (`multiinterval/numpy_compat.py`, numpy
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
  subclass's, Q15(h)). three incidents: a builder killed another stream's sabotage harness by
  matching its command line, and that stream's file was restored from its `.orig`; a duplicate
  builder was spawned by an orchestrator message and stood down, no damage; and an M16d reviewer, stopping its own hung probe, ran `taskkill //F //IM timeout.exe`, which kills every session's `timeout.exe` wrapper on the machine (their children survive); the damage, if any, was not measured. stop a process by its PID, never by image name or command line. gate on the merged tree
  green: 27795 passed in 84 s (`tests/itf1788`) + 5535 in 747 s (the rest) = 33330, 2026-09-28 at `511056d` (the merged code; the doc commit after it changes docs and one comment only). not pushed. the streams' records (`h3-records/`) folded into the docs and removed. then `.scratch/` audited before any delete (one auditor over the M15 leftovers, five over the M16 streams, read-only): 0 unclear, and its salvage transcribed the same session into plan §2 M16a-M16e (the prototypes' measurements, the reviewers' clean results and own breaks, the design probes), `v2-plan.md` (why `mul_rev` stays trv; numpy's own loops and pandas) and here (a third incident; pown-huge's non-saturating and float-exponent cases; the new row trig-rev-far). it also found a test-oracle bug live in the gate: `tests/test_literals.py::test_any_text_is_an_interval_or_undefined` failed whenever hypothesis drew `'[]'` (the empty literal is not contiguous; pre-existing at `04946af`), fixed with `@example('[]')`, red on the old assertion. `.scratch/h3/`, `.scratch/h3b/` and `.scratch/m13/` are audited and hold nothing untracked, but stay on disk: neither Recycle Bin route works from the session (the VisualBasic call is unsupported, the Shell COM move blocks on a dialog), so they go to the Recycle Bin by hand
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
* **2026-09-27** M13 finished, as an orchestrated build: M13e (reverse ops, `multiinterval/reverse.py`:
  sqr, abs, pown, cosh, mul, sin, cos, tan, pow_rev1, pow_rev2) and M13g (`DecoratedInterval`,
  `multiinterval/literals.py`'s 1788 text syntax, the four constructors, `set_dec`, the signals
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

## retired from `HANDOFF.md`, 2026-10-08

the banner, open items, open questions and still owed as they stood before the 2026-10-08 restructure, verbatim:
the banner had become a second session log. what is still live in them was carried into `HANDOFF.md`.

### banner (2026-10-06)

* **everything pushed, CI and fuzz green (2026-10-06)**: `origin/master` at `1330182` (`48631f5..1330182`: the rename,
  vectors-ext (a), m14b-open's parse items, T1, Q23, Q24; the owner's go). prepush green: x10 27798 in 65 s + 7143 in
  7430 s (under load), `gate:gmpy2` 34941 in 2106 s. CI run 37415051391 green (the gate 34941 passed, 288 s on python
  3.13; all jobs) and fuzz run 37415051379 green (`34941 passed in 2783.50s`, x10); checked by the session on GitHub.
  the "not pushed" in the bullets below is history. next: parallel fuzz (a research agent's report pending in
  `.scratch/fuzz-speed/`)

* **three open items done (2026-10-05/06), on `master`, not pushed**: vectors-ext (a) (`a3db14c`, the 40 quoted-string
  vectors run as upstream), m14b-open's `parse_value` and `_float_samples` (`281922b`; it also found `[0x12.5]` read as
  `[1, 2.5]`, now refused; its two leftovers were Q23, answered and built 2026-10-06 at `8edc643`: a number is what
  python reads, ASCII digits, an attached sign, `_` read, explicit separators; then Q24 at `086b3a6`: no trailing or
  doubled separator), and T1, the sabotage engine `tools/sabotage.py` (`076ad32`,
  `bf1ec7c`). records in plan §2 (M13a, "M14-breadth", T1). the push needs the x10 prepush and `gate:gmpy2`
* **renamed `multiinterval` (2026-10-05, `c6a4cca`), not pushed**: `intervals` is another project's on PyPI; the package
  directory, every import and `pip install` name is now `multiinterval`, the backend's variable `MULTIINTERVAL_BACKEND`
  (`v2-plan.md` decision log "the package is `multiinterval`"). the conda env, the repo folder and `references/` keep
  the old name. the push needs the x10 prepush and `gate:gmpy2` (every file moved)
* **trig-rev-far fixed (2026-10-05), on `master` at `8e709b3`, not pushed**: the far periodic reverse ops (`sin_rev`/`cos_rev`/`tan_rev`
  with an end at 1e22..10**400, which hung) answer in under a second with identical results: `reverse.py::_leap` skips the branches
  that round outside x, `elementary.py::_inverse_enclosure` caches f(v) (plan §2 "trig-rev-far"). gate 27795 + 6509 (2026-10-05).
  the push needs the x10 prepush and `gate:gmpy2` (`elementary` is a backend file)
* **everything pushed, CI and fuzz green (2026-10-05)**: `origin/master` at `48631f5` (the DST-gap oracle fix, prepush
  x10: 27795 in 25 s + 6490 in 4385 s); CI run 37223604634 green (all 8 jobs; the gate 34285 passed on python 3.12,
  3.13, 3.14 and `gate-gmpy2`, in 289-498 s) and fuzz run 37223604624 green (`34285 passed in 2933.42s`, x10).
  checked by the session on GitHub, not only the babysitter's word
* **pushed `5888c6e..a984e26` (2026-10-04, the owner's go)** after a green prepush (x10: 27795 in 63 s + 6489 in
  5344 s): CI run 37213772220 green; fuzz run 37213772177 red, `1 failed, 34283 passed`, a test oracle that assumed
  every aware wall time exists (a DST gap; plan §2 "fuzz-dst-gap"), fixed and pinned, the library unchanged but a
  docstring line
* **Q22 answered and H4 done (2026-10-04)**: open/closed flags are strict (a bool or numpy's, else TypeError,
  `cuts.flag`), the v1 README kept as `references/v1-readme.md`, and `archive/v1/` deleted with its two differential
  tests, after a parity audit found every v1 capability possible in v2 (839,971 side-by-side cases; plan §4 "the
  parity audit", §2 "q22-h4"; `references/v1-parity-2026-10-04/`)
* **M8, the time layer, built and merged (2026-10-04), not pushed**: `DateTimeInterval`, `TimeDeltaInterval`,
  `NEG_INF`/`POS_INF` (`multiinterval/time_interval.py`) as D30 decided; three reviews, a fix round (one pre-existing
  numeric bug: a numpy `timedelta64` read as an int, now refused), 77 sabotage breaks red (plan §2 M8, its review
  round). the owner's confirmations are Q21. H4 done the same day (above)
* **pushed `c55ce20..5888c6e` (2026-10-04, the owner's go)** after a green prepush (x10: 27795 in 92 s + 6184 in
  4358 s); CI green, fuzz red on a test oracle that predated D26 (plan §2 "fuzz-mulrev-point"), fixed, not pushed:
  the fix's push needs an x10 prepush
* **the quadratic parse fixed (2026-10-04), pushed at `5888c6e`**: `parse(' ' * 30000 + 'x')` 37 s -> 0.0002 s, the
  tokenizer's white-space runs possessive (`6450502`; plan §2 "M14-breadth", "the parse fixed"). gate green on
  its branch's worktree (27795 + 6242), which the main checkout's ledger does not see: the push's prepush runs
  the x10 fuzz anyway
* **the owner's answers to Q9-Q20 built (2026-10-04)**: the owner accepted every recommendation of
  `references/owner-questions-2026-10-03/` (2026-10-03); four streams built them and were merged into `master`
  (plan §2 "owner-answers"; D27-D29). behaviour changes: pown of exact operands past 2 ** 22 bits is the
  enclosure, pown to nearest correctly rounded (no libm), hex `repr` past 4300 digits, a method mixing the
  two classes outward (was a defect), elementwise `==` against an ndarray, `<<`/`>>` (dropped the next day), the 1788 layer's
  `nan` for the empty set's numbers, `OutwardMultiInterval.rounded()`; CI gains a `gate-gmpy2` job. no
  question is open for the owner. **pushed** `912558b..c55ce20` (2026-10-04, the owner's go); CI run 37147322181
  green (all 8 jobs; the gate 34029 passed on python 3.12-3.14, `gate-gmpy2` 34029 with `name()` gmpy2) and fuzz
  run 37147322232 green (34029 passed in 3458 s, x10)
* **everything pushed, CI and fuzz green (2026-10-03)**: `origin/master` at `912558b`, CI run 37107899644 and fuzz run 37107899636 green. this push carried the CORE-MATH sample, the overnight fuzz additions, two test-oracle fixes (plan §2 "fuzz-mixed-points") and one library fix the fuzz on CI found, a crossed piece in the exact class (plan §2 "fuzz-rootn-crossed"; its semantics are Q20, the owner's to confirm). the CORE-MATH and M14-breadth bullets below are history
* **CORE-MATH worst cases built (2026-10-02), not pushed**: a gate sample (`tests/test_coremath.py`) and a manual full check (`tools/coremath.py check`; asked for at the end of a session that changed the scalar evaluator, `CLAUDE.md`); the first full check found 0 mismatches in 51M calls. `.scratch/coremath-cache/` is kept on purpose
* **M14-breadth done (2026-10-02)**: six streams of properties (plan §2 "M14-breadth"); five library bugs it
  found are fixed and pinned, one a soundness hole (outward floor/ceil/trunc past 2 ** 53); the x10 fuzz found one
  more (a mixed-type point's `repr`) and six test oracles, all fixed. pushed at `233fdd4` with the owner's go
  (2026-10-02: "Okay go", after "pause the fuzzing, lets do M14-breadth finished then run fuzzing"); CI run
  36989262364 and fuzz run 36989262372 both green (the session log's top entry)
* **what has run on this code is in the run ledger (2026-10-01)**: `tools/gate.py status` (read it at
  session start). every gate and fuzz run goes through `tools/gate.py run <phase>`, and
  `tools/prepush.sh` runs only what the ledger says a push still needs (`CLAUDE.md`; the testing skill;
  plan §2 "run ledger"). a run made before the ledger existed is not in it: the first fuzz-x10 rows
  come from the next prepush
* **fuzzing is green on GitHub**: `origin/master` at `93d9e3e`, CI and fuzz green (fuzz twice in a row,
  2026-09-30 and 10-01). docs-only pushes skip the fuzz: the owner confirmed it 2026-10-01 ("yes docs
  can skip that"), as built in `ccc6c3f`. `ccc6c3f`, `8be9bd1` and the ledger's commit are not pushed,
  awaiting the owner's go; the ledger's commit changes source, so its prepush is a full fuzz run
* **everything is on `master` (2026-09-29, the owner: "make sure everything is on the master
  branch")**: `master` fast-forwarded to `v2` at `8a506f3` (it was `v2`'s ancestor) and pushed after
  a green gate (27795 + 5608 = 33403 passed at `4f51e86`, 2026-09-29; the one commit after it is
  docs only). branches `v2` and `pown-huge` deleted (both contained in `master`); `master` is the
  working branch from here. `fuzz.yml` is on the default branch now; since 2026-09-29 it runs on
  every push to `master` (no schedule), after `tools/prepush.sh` locally (`CLAUDE.md` "push"). branch names `v2` below are history. CI green at `8a4cc2f`. the fuzz
  rerun found fuzz-floordiv-overflow, a test-oracle bug, fixed and pushed at `97d9824`
* **fuzz-symmetry fixed on `v2`, not pushed (2026-09-29)**: the fuzz run's one failure was the
  library's: `-` of a point whose cuts differ in type (an exact 1/2 and a float 0.5) came back in one
  type, so a reverse op was not odd. `-` is now the typed cut mirror and `+` the identity
  (`multiinterval/ops.py`). gate green: 27795 + 5608 = 33403 passed, 2026-09-29. record: plan §2 "fuzz-symmetry"; `v2-plan.md`
  2026-09-29 decision-log entry. the fuzz rerun that checks it on GitHub is row 1's
* **pown-huge built on branch `pown-huge` (`409b2e6`..`3ed888f`, off `v2` at `7288e81`), merged
  into `v2` by fast-forward 2026-09-29 (the owner's go), not pushed**: outward pown of a float never builds a power past
  `elementary.EXACT_POWER_LIMIT` (past it each end is `elementary.rounded_pow`, attainment sees
  `ops._NOT_A_DOUBLE`), the nearest class past `|n| = 2 ** 53` is `rounded_pow` to nearest (it lost
  n's parity: `M(-1.0) ** (2 ** 60 + 1)` was `[1.0]`, and `M(0.5) ** 10 ** 400` was `[inf]`), and the
  descriptor name is bounded (`A ** 2 ** 20000` raised python's 4300-digit ValueError). only
  `multiinterval/ops.py` changed in the library. every float reproduction of the old row now answers in
  under 0.1 s; exact int/Fraction operands still hang (Q17). gate green at `3ed888f`: 27795 passed in
  57 s + 5607 in 579 s = 33402, 2026-09-29. record: plan §2 "pown-huge"; `v2-plan.md` 2026-09-29
  decision-log entry
* **M16 (H3's second part) built on `v2` (merged in `h3-merge`, then fast-forwarded), pushed 2026-09-28**: the owner, 2026-09-27: "get
  the rest of h3 done". five streams, each on its own branch off `v2` at `04946af`, merged into
  `h3-merge` without conflicts: M16a the solver in several variables (`gradient`, `jacobian`,
  `solve`, `RootBox`, exported from `multiinterval`), M16b the 1788 layer (`multiinterval/ieee1788.py`, not
  imported by `multiinterval`), M16c the per-piece allen matrix (`allen_matrix`, `allen_relations`),
  M16d numpy interop (`multiinterval/numpy_compat.py`; numpy optional), M16e the gmpy2/mpfr backend
  (`multiinterval/backend.py`, `multiinterval/_gmpy2.py`; opt-in, the pure path by default). the choices
  the build made are D20-D24, owner questions Q12-Q16. gate on the merged tree green: 27795 passed in 84 s (`tests/itf1788`) + 5535 in 747 s (the rest) = 33330, 2026-09-28 at `511056d` (the merged code; the doc commit after it changes docs and one comment only). records:
  plan §2 M16; design: `v2-plan.md` "current design" and its five 2026-09-28 revisions
* **M15 (H3's first part) built on `v2` (`04946af`), pushed 2026-09-28 with M16**: forward-mode autodiff
  (`multiinterval/autodiff.py`, `Dual`, `derivative`) and interval newton (`multiinterval/solver.py`,
  `newton`, `Root`), exported from `multiinterval`. the choices the build made are D19, owner question
  Q11. record: plan §2 M15; design: `v2-plan.md` "the solver stack". M16 is on top of it

* branch `v2`: M13 finished and merged 2026-09-27 (branches `m13e`, `m13g`, merged in `m13-merge`,
  then fast-forwarded into `v2`); pushed, CI run 36305984327 at `a1d45a9` green (all 8 jobs, 22166
  passed on each of python 3.11-3.14, 2026-09-27). M15 and M16 pushed 2026-09-28 at `3aaf8f4`: CI run 36402681261 at `3aaf8f4` (M15 and M16's first, 2026-09-28): 7 of 8 jobs green, the gate 33330 passed on python 3.12-3.14 in 365-429 s; on python 3.11 `1 failed, 33329 passed in 393 s`, `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[longdouble-pow]`: the oracle's `Fraction(2 ** 60 + 1, 2 ** 60) ** (-inf, -2)` gave `[1.0]`. first misread as numpy 2.4 comparing a long double as its double (`42234f0` made the oracle decide exactly, and run 36406179185 failed the same way); the cause is CPython 3.11's `Fraction.__pow__`, which answers any non-rational exponent with `float(a) ** b`, so a Fraction base is rounded before the library's `__rpow__` sees it: on 3.11 `Fraction(1, 3) ** O(2)` is `(0.11111111111111109, 0.1111111111111111)`, missing 1/9, through `Dual` too (checked with a local 3.11.15; `+ - * / // % divmod` stay exact; 3.12 returns NotImplemented). nothing in the library can see it (its `__rpow__` gets a float), so the owner set python >= 3.12 (D25); `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11. the exhaustive jobs 19 s to 8 min 54 s. python 3.12 is the floor since 2026-09-28 (D25): `pyproject.toml`, `ci.yml`'s matrix 3.12-3.14; pushed at `c1552b1`, CI run 36415649083 is green (all 7 jobs, 2026-09-28): the gate 33332 passed on each of python 3.12-3.14 in 355-410 s, the exhaustive jobs 23 s (sabotage) to 12 min 12 s (modulo), no 3.11 job
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
  job and flint oracle built; the fuzz job's first GitHub run (2026-09-29, run 36507253782, ×10,
  53 min) found one failure, `tests/test_reverse.py::test_symmetry`, fixed the same day (fuzz-symmetry)

### open items (ranked)

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | m14b-open | what M14-breadth found and left (2026-10-02): number-type quirks with no wrong value (a 0 end exact among float operands: `abs(M(-1.0, 1.0))` is `[0, 1.0]`; trunc's non-negative side ints; a one-point domain clip takes its low cut's type; all three still reproduce at `c6a4cca`, 2026-10-05) | ready, small (the 4300 digits done 2026-10-04: hex, D28; the quadratic parse done 2026-10-04, `6450502`; `parse_value` strict and `_float_samples` past the doubles done 2026-10-05, `281922b`, its two leftovers are Q23) | plan §2 "M14-breadth" (left open) |
| 2 | vectors-ext | test data beyond ITF1788 (survey 2026-09-29, `references/test-vector-sources.md`). the 1788 set is complete. decided 2026-10-02 with the owner: outputs computed by MPFR are not worth vendoring (our arb oracle already checks correct rounding); what adds depth is the CHOICE of inputs, so the source is CORE-MATH's worst cases (MIT; they carry blocks of Lefevre's data, whose own files state no licence, §3f), built 2026-10-02 as `tools/coremath.py` and `tests/test_coremath.py` (§3g-§3h; the testing skill). left: (a) done 2026-10-05 (`a3db14c`: quoted strings keep their white space, the 40 vectors run as upstream states them; plan §2 M13a); (b) closed 2026-10-03 (owner): glibc's rows are conformance inputs, not hard cases; pown has CORE-MATH's integral-exponent pow rows since 2026-10-04 (`tests/coremath/pown.tsv`); (c) cuinterval's `custom.itl` (26, MIT), probably covered by `test_domain_ends_and_limits` | (c) only, low value | `references/test-vector-sources.md`; plan §2 M13a (the vendoring) |
| 3 | pown-ziv | an exact corner of about 2M bits within about 2 ** -(its size) of a rounding breakpoint, past `EXACT_RESULT_LIMIT`, runs ziv past 120 s where the old code built the power in milliseconds (`O(3 + 2 ** -2100000) ** 2`, 2026-10-04); pow had the same at 60k-bit operands before D28 and has it at 2M bits now. follow-ups: a near-1 shortcut in `rounded_pow`, or the exact build when ziv passes a precision cap and the build is affordable | ready, not scheduled; extreme sizes only | plan §2 "owner-answers"; `references/owner-questions-2026-10-03/streams/pown.md` step 6 |
| 4 | evaluate-box | a pure speed change noted by M16e's design: `applicator.evaluate_box` evaluates a float corner's exact value three times under `OUTWARD`; passing `fn`'s value into the hook would cut it to one, maybe worth as much for arithmetic as the backend, with no dependency | idea, not scheduled (M16e, 2026-09-28) | plan §2 M16e; `v2-plan.md` "elementary and step functions" (the backend) |
| 5 | later | left open by the owner's answers (2026-10-03), each "consider", optional or "if asked": `Root`/`RootBox` as frozen dataclasses if a third state appears; an `rtol` beside `tol`; outward fma, `%`, hypot, `cancel_minus` typed per corner (tighter); a to-nearest `MultiInterval.rounded()`; rootn run on cbrt's worst-case inputs; a numpy hook for the 1788 layer; an `AllenMatrix` class or a public `allen_pairs`; a strategies module after 2.0; Q20's optional pin | not scheduled | `references/owner-questions-2026-10-03/` |
| 6 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | when everything is fully done (owner 2026-09-26); M16e's backend is opt-in and can ship in 2.0 as is (Q16(f)) | plan §2 M11; D5, D17 |

### open questions for the owner

none open. Q24 was answered 2026-10-06 (trailing and doubled separators refused; juxtaposed items stay refused;
`v2-plan.md` decision log "separators exactly between items") and built (`086b3a6`). Q23 was answered 2026-10-06 and built (`8edc643`; `v2-plan.md` decision log "a number is what python reads").

Q21 was answered 2026-10-05 (the session's recommendations accepted, all as built; `v2-plan.md` decision log, "M8's choices confirmed"). Q22 was answered 2026-10-04 (`v2-plan.md` decision log, "strict flags; v1 deleted"). Q9-Q20 and the owner's-call rows were answered 2026-10-03 (the owner accepted every
recommendation of `references/owner-questions-2026-10-03/`; `v2-plan.md` "2026-10-03 revision: owner
answers"; D27-D29) and built 2026-10-04 (plan §2 "owner-answers"). Q1-Q8 answered 2026-09-26, D18
2026-09-27 (`v2-plan.md`). what the answers left for later is the open-items row "later".

### still owed

* doc-framework fixes owed (cross-repo context-framework audit, 2026-10-07; recorded from
  outside this repo, not a session here; owner's instruction: deal with these):
  * the banner (`## banner (2026-10-06)`, ~124 lines) has become a second session log. cut
    it to the current state; history already lives in the session log
  * this file is ~62 KB, with 38 session-log entries and no rule for moving them out. move
    older entries to `docs/session-log.md`, and write the rotation rule into the preamble
  * **owner question:** decisions live in three places: `v2-plan.md::decision log`,
    `v2-implementation-plan.md::0. decisions` (D1-D30) and answers recorded here. pick one
    home (e.g. `docs/decisions.md` continuing the D-ids, or plan §0) and make the others
    point to it. ask, don't choose

* fuzz gaps a census left (2026-10-03; the rest of its shortlist is built): no @given test for `DecoratedInterval.log(base)` (random base), the decorated reflected ops and divmod (examples only), the slow decorated functions (pow, hypot, trig) on float operands, exact-operand equality of the outward and nearest classes for about 25 more functions, `Builder` (low value)
* the run ledger (2026-10-01) knows local runs only: a push whose src is unchanged since `origin/master`
  trusts that master's fuzz run was green on CI (every push is watched to the end), it does not check.
  count floors (zanzibar's `MIN_TESTS_ALL`) are recorded in each row, not enforced: a gate that
  collected fewer tests would still read green
* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (plan §2 M13d sabotage). recorded, not scheduled
* M16's new tests first ran on CI 2026-09-28 (run 36402681261, see the banner); none has run
  under the fuzz profile. the long double cases (`tests/test_numpy_compat.py::test_longdouble_is_exact`,
  the `longdouble` rows of test 2) discriminate only where `np.finfo(np.longdouble).nmant > 52`
  (CI's linux, never this laptop). the 3.11 `Fraction ** interval` hole (D25) reached CI unseen
  because the local env is python 3.13; a local 3.11 (`speechenhancement`'s env) confirmed it
* M16a: a constructed n = 3 system with two zeros costs more than 120 s, so n = 3 is covered by one
  constructed zero and the sphere only, with no random n = 3 test; whether a faster jacobian
  (Q12(a)) or a tighter form of `F` would change that is not measured. the natural path to a split
  of a box already proved unique was not found (pinned by a monkeypatched `_krawczyk` only). a
  continuum costs up to 2n + 1 output boxes per box of width `tol` (review F3); whether to output it
  differently (one box per connected unproved region) is not asked, Q12 has no item for it
* sabotage rows red in their first run were not re-run after the closing tests were added (M16a:
  the four closing tests and the review's, which only add red paths); M15's table was not re-checked
  for the stale-bytecode hazard M16c found (plan §2 T1; `tools/sabotage.py` could re-run either now)
* M16b: the layer's per-call
  `warnings.catch_warnings` is not thread-safe on python 3.11-3.13 (as `decorated.py::_quietly`);
  recorded, not addressed. the pass imports the adapter a second time as
  `tests.itf1788.test_itf1788` (the vectors parsed twice, 7.5 s cold, 2026-09-27); accepted. one
  leftover gate call of M16b's verifier ended rc=1 with its output lost (its log overwritten
  mid-run), on the tree less one test edit; both later runs of that call green; recorded in plan §2
  M16b, not explained; a likely cause, found after the merge: `tests/test_literals.py::test_any_text_is_an_interval_or_undefined`, in that call, asserted `result.is_contiguous`, false for `text_to_interval('[]')`, the empty literal, so it failed whenever hypothesis drew `'[]'` (pre-existing at `04946af`, a test-oracle bug; found by M16e's verifier). fixed 2026-09-28 at the merge: `@example('[]')`, red on the old assertion, and the assertion is now `result.is_empty or result.is_contiguous`
* M16c: the other relations over cut tuples in `relations.py` (`before`, `adjoins`, ...) also read
  normalized operands and do not assert it; only `allen_relations`, whose wrong answer would be
  silent and partial, does. the methods are unaffected
* M16e: the whole suite under `MULTIINTERVAL_BACKEND=gmpy2` runs in CI's `gate-gmpy2` job since 2026-10-04
  (Q16(e)) and locally as `gate:gmpy2` when a backend file changed. the backend is verified only with gmpy2 2.3.1 / MPFR
  4.2.2 on windows (python 3.13); CI's linux jobs run `tests/test_backend.py` with the PyPI wheel, no
  other MPFR build has been run. free-threaded builds untested (the three contexts are shared module
  objects; `backend._use` is a global). the speed numbers were taken beside four other streams' runs:
  a quiet-machine re-measure (`tools/backend_speed.py`) is owed before any number goes into the
  README. gmpy2 3: `auto` and `[test]` stop below it; when it ships, run the differential against it
  before `backend.CEILING` and the `[test]` pin move together
  (`tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` keeps them equal)

* pown-huge: `ops._check_marker_premises` refuses at import an `EXACT_POWER_LIMIT` below the
  marker proof's floor (36550 bits), but a limit lowered at run time, after import, still stalls
  (`O(0.5, 1.0) ** 1074` past a 60 s timeout; review SAB-3). recorded, not guarded
