# fuzz/prepush parallelization study (2026-10-06)

research only; no tracked file changed. private copy of HEAD 1330182 (src s:6fb5edf5749a) in
`.scratch/fuzz-speed/tree` (deleted at the end). python `C:/Users/user/anaconda3/envs/intervals/python.exe`
3.13.15, pytest 9.1.1, hypothesis 6.167.1. pytest-xdist 3.8.0 + execnet 2.1.2 installed with
`pip install --target .scratch/fuzz-speed/site --no-deps` (NOT into the env) and put on PYTHONPATH per run.
machine: 12 logical CPUs, shared. the main checkout's prepush (fuzz-x10:rest, then gate:gmpy2) was running
during the first part of this study (PIDs 23780 gate.py / 22448 pytest); other sessions' python.exe noted per run.

## log (appended as measurements land)

* 10:27 setup: copy made with `git archive HEAD | tar -x`; xdist installed to side dir (pip.log).
  load at 10:27: prepush pytest 22448 (1 core), adhoc-microphone-array pytest 8020 + sabotage.py 13344, hawker-bot.

### hypothesis 6.167.1 source facts (read 10:30, for Q2/Q3)

paths are under `C:/Users/user/anaconda3/envs/intervals/Lib/site-packages/`.

* **seed**: `hypothesis/core.py::get_random_for_wrapped_test`: with `derandomize=False` (the fuzz profile and
  `tests/conftest.py::pytest_collection_modifyitems` rewrap both set it) and no `@seed` / `--hypothesis-seed`
  (`_hypothesis_pytestplugin.py` SEED_OPTION -> `core.global_force_seed`), each test's Random is seeded from
  `threadlocal._hypothesis_global_random = Random()` (os.urandom). so separate processes (xdist workers, or
  10 independent runs) draw independent seeds. grep: no `@seed`, no `target(` anywhere in tests/ or multiinterval/.
* **size ramp**: `internal/conjecture/engine.py::ConjectureRunner.generate_new_test_cases`: `small_test_case_cap =
  min(max_examples // 10, 50)`; while `valid_test_cases <= small_test_case_cap` each new case is capped at 5x the
  length of the zero-extended prefix (small inputs first). share of budget in the small phase: x1 at
  max_examples=100 -> 10 of 100 (10%); x10 (1000) -> 50 of 1000 (5%). so 10 runs of x1 spend 2x as many calls
  on deliberately small inputs as 1 run of x10 (100 vs 50); for max_examples=60: 6/60 (10%) vs 50/600 (8.3%).
* **zero data**: the same function runs the "simplest" choice sequence first, once per run: 10 runs -> 10 calls
  on the identical minimal input (9 wasted), 1 run -> 1.
* **dedup**: within a run, `generate_novel_prefix` + `self.tree.simulate_test_function` (DataTree) never
  re-runs a choice sequence already tried, and `__tree_is_exhausted` stops a test whose finite input space is
  used up (`should_generate_more`). across 10 independent processes there is no shared tree: duplicates across
  runs are possible and an exhausted small space is re-walked 10 times.
* **targeting**: `_should_optimise_now` / `_run_optimise_pass` (max_examples >= 1000 interleaves passes, < 1000
  one pass at half budget) only acts on `target()` observations; the suite has none, so irrelevant here.
* **mutations**: `generate_mutations_from` runs after each generated case once health checks are done
  (`health_check_state is None`); it duplicates same-label subtrees. it works within a run's tree; it scales
  with the number of generated cases, the same per call at x1 and x10.
* **database replay** (`reuse_existing_test_cases`): replays every primary (failing) entry, then tops up
  from the secondary corpus to `desired_size = max(2, ceil(0.1 * max_examples))`; the replayed calls count
  against `max_examples`. in a GREEN run nothing is written: `save_choices` is called only for interesting
  (failing) cases, and the pareto front is saved only when `data.target_observations` is non-empty
  (engine.py `test_function`, `if data.target_observations and self.pareto_front ...`). the main checkout's
  `.hypothesis/` is 318 KB with 5 entries under examples/ (10:30). so replay costs ~nothing on this green suite.
* **db concurrency**: `database.py::DirectoryBasedExampleDatabase.save` is content-addressed (file name = hash
  of the value), written to `tempfile.mkstemp()` and `rename`d into place; any OSError on rename (on Windows a
  FileExistsError when two writers race on the same value) unlinks the temp file and is swallowed; `fetch`
  and `delete` swallow OSError too. so concurrent writers (xdist workers, 10 runs) cannot corrupt it; the
  worst case is a lost delete or a duplicate re-save of the same entry (idempotent).
* **explicit @example**s run before the engine (`core.py` explicit examples), outside `max_examples`: 10 runs
  pay them 10 times, 1 run once.
* **health checks** run in every run's first cases (`HealthCheckState`); too_slow is suppressed by the fuzz
  profile, the others (filter_too_much, data_too_large, large_base_example) fire per run: 10 runs = 10 chances.

### parallel-safety: static survey of tests/ (10:35; empirical check under xdist below)

* **wall-clock assertions** (not hypothesis tests; same at x1 and x10, but CPU contention inflates them):
  `tests/test_modulo.py::test_attainment_is_constant_time` (< 0.05 s, the tightest),
  `tests/test_literals.py::test_invalid_text_is_refused_in_linear_time` (< 1.0 s for 8 inputs),
  `tests/test_elementary.py::test_rational_log_of_large_operands_is_fast` (< 5 s per input),
  `tests/test_fmt.py::test_white_space_runs_parse_in_linear_time` (< 10 s per param).
  `tests/test_sabotage_tool.py` waits on child processes with `time.sleep(0.5)` / `time.monotonic()` deadlines.
* **global state**: `backend._use(...)` (tests/test_backend.py, a context manager) and `warnings.simplefilter`
  inside `warnings.catch_warnings()` blocks are process-local; xdist workers are processes, so they cannot
  leak across workers. within one worker the order differs from serial, so an order dependence would show as
  a pass-count or failure difference (checked below).
* **files**: tests that write go to `tmp_path` (test_gate_ledger 13 uses, test_sabotage_tool 75); xdist gives
  each worker its own basetemp (`pytest-of-user/pytest-N/popen-gwK`), and concurrent pytest sessions get
  separate numbered basetemps under a lock (`_pytest/pathlib.py::make_numbered_dir_with_cleanup`, LOCK_TIMEOUT
  3 days). reads of the repo itself: test_gate_ledger `ROOT` (git check-ignore / ls-files, pyproject),
  test_numpy_compat (pyproject), read-only.
* **subprocess-heavy tests** (spawn python/pytest/git children, so one xdist slot uses more than one core):
  test_sabotage_tool (10 subprocess sites), test_gate_ledger (5), test_pown_huge (4), test_errors (3), test_fuzz_profile,
  test_backend, test_ieee1788_layer, test_numpy_compat, test_time_interval (2 each).
* **shared `.hypothesis/` subtrees**: `examples/` (safe, see db concurrency above), `constants/`
  (`internal/constants_ast.py`: a cache keyed by the source's sha1, written with a plain non-atomic
  `write_text`; a reader racing a writer may read a truncated file -> fewer constants or a parse error that falls
  back to recomputing; every writer writes the same bytes, so benign), `unicode_data/`, `observed/` (only with
  observability on).

### CI job times (GitHub API, read 10:40; runs of 2026-10-03/04, ubuntu-latest, 4 vCPU, serial pytest)

| run | job | job s | pytest step s |
|---|---|---|---|
| 37223604624 (48631f5) | fuzz (x10) | 2957 | 2938 (pip 12) |
| 37147322232 (c55ce20) | fuzz (x10) | 3478 | |
| 37223604634 (48631f5) | gate py3.13 / 3.12 / 3.14 | 308 / 484 / 557 | 294 (3.13) |
| 37223604634 | gate (gmpy2, 3.13) | 437 | 417 |
| 37147322181 (c55ce20) | gate 3.13 / 3.12 / 3.14 / gmpy2 | 509 / 485 / 482 / 352 | |
| 37223604634 | exhaustive modulo / ops exact / float / sabotage | 397 / 388 / 316 / 29 | |

ci workflow wall: 9-13 min (the exhaustive and gate jobs run concurrently on separate runners); fuzz 35-64 min.

### Q1 part 1: x1 per-test durations (10:28-10:45, default profile = the local gate, one worker, file by file)

command: `x1_loop.sh` (one `pytest -q -p no:cacheprovider --durations=0 <file>` per test file, plus tests/itf1788, multiinterval, README.md); logs `x1/*.log`, per-file table `x1/summary.tsv`, aggregate `x1_durations.txt` (`durations.py x1/*.log`).
load: the prepush pytest (PID 22448, fuzz-x10:rest, 1 core), another repo's pytest and conformance runs (2-3 python.exe), cpu% about 11-25.

result: 34941 passed (= gate:itf 27798 + gate:rest 7143), sum of per-file walls 918.7 s (includes ~1 s collection per file).

```
items with a duration line: 1915, functions: 491, total 786.7 s
  top  1 items:     48.0 s =   6.1%
  top 10 items:    121.5 s =  15.4%
  top 50 items:    241.4 s =  30.7%
  top  1 functions:     48.0 s =   6.1%
  top 10 functions:    172.2 s =  21.9%
  top 50 functions:    392.3 s =  49.9%

top 25 items:
     47.99  tests/test_solve.py::test_every_zero_is_enclosed
     15.53  tests/test_solve.py::test_every_zero_is_enclosed_on_a_budget
     12.84  tests/test_reverse.py::test_trig_rev_hull_past_the_cap
      8.76  tests/test_sabotage_tool.py::test_a_timeout_kills_the_tree_and_restores
      7.31  tests/test_coremath.py::test_the_sample_is_correctly_rounded[atan]
      6.75  tests/test_modulo.py::test_prototype_fuzz
      6.17  tests/test_pown_huge.py::test_identity_with_the_exact_construction
      5.97  tests/test_reverse.py::test_mul_rev_is_exactly_the_points_that_fit
      5.30  tests/test_fmt.py::test_spellings
      4.89  tests/test_reverse.py::test_mul_rev_isotone_and_distributive
      4.54  tests/test_orders.py::test_orders_sound_at_sampled_points
      4.29  tests/test_solver.py::test_simple_irrational_zeros_are_proved_unique
      4.09  tests/test_fmt.py::test_any_text_parses_or_raises_value_error
      3.92  tests/test_functions.py::test_atan2_sound_on_exact_sets
      3.84  tests/test_reverse.py::test_trig_rev_exactly_the_points_with_f_in_c
      3.77  tests/test_orders.py::test_interior_laws
      3.72  tests/test_sabotage_tool.py::test_mixed_verdicts[control-PASSED]
      3.70  tests/test_pow_rev.py::test_pow_rev_sound_at_sampled_points
      3.55  tests/test_gate_ledger.py::test_the_plan_end_to_end
      3.53  tests/test_pow_rev.py::test_pow_rev_isotone_and_distributive
      3.48  tests/test_fmt.py::test_white_space_runs_parse_in_linear_time[many-separators]
      3.37  tests/test_oracles.py::test_general_exact_operands
      3.21  tests/test_pown_huge.py::test_one_limit_for_pown_pow_exp2_exp10
      3.17  tests/test_modulo.py::test_sound
      3.14  tests/test_sabotage_tool.py::test_a_head_snapshot_does_not_see_the_working_tree

top 25 functions (sum over params):
     47.99  tests/test_solve.py::test_every_zero_is_enclosed  (1 items)
     19.39  tests/test_coremath.py::test_the_sample_is_correctly_rounded  (25 items)
     17.11  tests/test_functions.py::test_sound_on_exact_sets  (30 items)
     15.53  tests/test_solve.py::test_every_zero_is_enclosed_on_a_budget  (1 items)
```

slowest files at x1: test_functions 95 s, test_reverse 84 s, test_solve 80 s, itf1788 77 s, test_sabotage_tool 42 s, test_fmt 33 s, test_ops_properties 31 s.

### the slowest test, per case (10:47, light probe, 1 worker)

`HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=1 HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER=1 pytest tests/test_solve.py::test_every_zero_is_enclosed`
-> 44 s wall (log `obs_solve_x1.log`, observations `obs_solve_x1/`). the test pins `max_examples=10` and has 6 `@example`s.
per-case timing from the observations: **6 explicit examples = 33.4 s (max 15.5 s)**, 16 generated cases (10 valid)
= 16.7 s (heavy tail: 7.6, 6.7, 1.5, then ~0.2 s each). so this test is ~2/3 fixed cost (explicit examples) at x1;
at x10 expect ~33 s + ~100 valid generated cases at ~1 s mean = roughly 150-250 s (heavy-tailed). note for Q3:
10 runs of x1 pay the 33 s of explicit examples ten times (334 s) where 1 run of x10 pays it once.

### fixed cost of a run (10:49-10:53, 1 worker)

`pytest -p explicit_only --ignore=tests/itf1788` with scratch plugin `plug/explicit_only.py` (loads a profile with
`phases=[Phase.explicit]`: every hypothesis test runs only its @examples, nothing generated) -> **216 s wall**,
5924 passed, 1219 skipped (hypothesis tests with no @example skip). log `explicit_only_rest.log`. load: prepush
pytest 22448 + 2-3 other-repo python.exe.
so a full run's fixed part (collection, plain tests, explicit examples, no generation) is ~216 s (rest) + ~77 s
(itf1788, which has no hypothesis) = **~290 s per process**, independent of the multiplier.
sanity check of linearity: x1 total 919 s (loop above) -> generated part ~630 s; serial x10 on the ledger is
65 + ~4400 s (4385 s on 2026-10-05, 5344 s on 2026-10-04) -> generated part ~4100-5000 s, i.e. 6.5-8x, not 10x.
part of that is load noise (the ledger's gate:rest at x1 ranges 342-1454 s across days), part is tests whose small
input space is exhausted before max_examples (`__tree_is_exhausted`), which scale less than 10x.

### database concurrency, empirical (10:58)

`dbstress.py`: two processes at once, 4000 random save/fetch/delete each on one `DirectoryBasedExampleDatabase`
(4 keys x 30 overlapping values) -> 0 exceptions in either, 0 malformed entries after, no leaked temp files of
the value size in %TEMP%. consistent with the source reading: concurrent use is safe on Windows with 6.167.1.

### what tools/gate.py would need (read 11:01)

* **xdist inside a phase** (`-n K`): nothing in the keying changes. a fuzz-x10 phase still runs every test at 10x its
  max_examples (xdist splits tests between processes, never a test's examples), so the verdict means the same.
  needed: `phase_spec` adds `-n K --dist worksteal` for the fuzz and gmpy2 phases (K from an env var such as
  `GATE_WORKERS`, default chosen for the laptop; `-n 0` = serial); `run_phase` records `workers=K` in the row's
  facts (recorded, never matched, like the package versions); `_counts` already reads xdist's summary line
  (`N passed in Xs`, same format: checked in the -n runs below). `tests/test_gate_ledger.py` pins `phase_spec`'s
  exact arguments (around lines 187-190 and 331) and must be updated with it. pytest-xdist must be in
  pyproject's `[test]` extra (CI installs `.[test]`) and in the local env (install when no run is active).
* **10 x x1 as a fuzz-x10 verdict**: the phase name carries the multiplier, and `fuzz_covered` accepts a phase iff
  `_fuzz(phase)[0] >= PUSH_MULTIPLIER`. a 10 x x1 run would need (a) a new phase grammar, e.g. `fuzz-x1r10:rest`
  (multiplier 1, 10 repeats), in `PHASE_RE`, with `_fuzz` returning the effective budget m*r and the owner deciding
  that m*r counts as x10 (it is not the same search: see Q3); (b) `run_phase` to launch r processes (no
  `--hypothesis-seed`, so seeds differ), one log each, and record PASSED only if all r passed with equal item counts,
  taking the tree ids before the first and after the last; (c) fuzz.yml to do the same, or the local verdict no
  longer stands for "as fuzz.yml"; (d) new pins in tests/test_gate_ledger.py. far more change than (a) xdist.
* 11:58 coordinator (owner's request): heavy runs may start now, at most 4 workers total, under the prepush's load (fuzz-x10:rest at 92%, then gate:gmpy2); re-take key timings at up to 8 workers after the prepush.

### the prepush's own serial timings today (ledger rows, s:6fb5edf5749a)

| phase | started | dur s | load during it |
|---|---|---|---|
| fuzz-x10:itf | 10:02:33 | 65 | light |
| fuzz-x10:rest | 10:03:39 | **7430** (124 min) | my 1-worker probes 10:27-10:53, then from 12:00 my R1 (-n 4); other sessions' python (2-10 procs) |
| gate:gmpy2 | 12:07:30 | **2106** (35 min; ~754 s on 2026-10-04) | ran entirely beside my R1 (x10, -n 4) plus another repo's fit_k.py x3, pytest tiles, synth_scenes: CPU 82-91 % at 12:30-12:40, 12 python.exe |

so the serial fuzz phase was 7495 s today versus 4410 s on 2026-10-05 and 5407 s on 2026-10-04: load moves it by +-50%.
the gmpy2 row is an unplanned Q4 data point: serial gate:gmpy2 concurrent with an x10 fuzz at -n 4 on a busy laptop = 35 min.

### R1: full suite, x10 fuzz profile, xdist -n 4 --dist worksteal (12:00:49-13:31:30, under the prepush)

command: `xrun.sh R1_x10_n4 4 HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 -- --hypothesis-show-statistics` (pytest -q -p
no:cacheprovider --durations=0 -n 4 --dist worksteal, whole suite incl. itf1788). log `R1_x10_n4.log`, aggregate `R1_durations.txt`.
load: the prepush (fuzz-x10:rest's last 7 min, then gate:gmpy2 for 35 min), another repo's fit_k.py x3 + eval_module +
pytest tiles + synth_scenes; CPU 82-91 % at 12:30-12:40, 39 % at the end; 8-12 python.exe.

* **wall 5439 s (90.7 min); 1 failed, 34940 passed** (the serial count 34941 = 34940 + the 1 failure: every item ran).
* **the failure is a wall-clock assertion**: `tests/test_fmt.py::test_white_space_runs_parse_in_linear_time[many-separators]`,
  `assert (1766044.75 - 1766034.54) < 10` (10.22 s); its siblings took 5.88 s (`many-separators-trailing`) and 4.76 s
  (`-doubled`). at x1 alone the same case took 3.48 s. so under parallel load this test is flaky, not wrong.
  the other timing tests passed with margin here: test_elementary `..._is_fast` 0.01 s, test_literals `..._linear_time`
  0.01 s, test_modulo `test_attainment_is_constant_time` (< 0.005 s, not listed).
* sum of per-item durations 21326 s = 4 workers x ~5330 s: the 4 workers were busy to the end (worksteal balanced it;
  no idle tail). top 1 / 10 / 50 items = 4.8 / 28.3 / 45.9 % of the time; top 50 functions 61 %.
* slowest items in R1: test_reverse::test_mul_rev_the_largest_set 1019 s, test_minmax_fma::test_fma_is_mul_then_add_on_exact_sets
  776 s, test_reverse::test_mul_rev_isotone_and_distributive 736 s, test_minmax_fma::test_fma_with_floats_rounds_the_exact_result_once
  610 s, test_relations::test_set_relations 597 s, test_solve::test_every_zero_is_enclosed 549 s.

### the machine explains most of it (13:40)

`Win32_Processor`: **13th Gen Intel Core i7-1365U, 10 cores / 12 threads = 2 P-cores (4 threads) + 8 E-cores, a 15 W U-series
part.** only 4 hardware threads are fast; E-cores run well below P-core speed and an all-core load lowers every clock. so
xdist on this laptop cannot scale like "12 CPUs" suggests, and timings swing with what other sessions run.

the R1 per-test durations are inflated far beyond the 2.9x average, by test (standalone reruns, 13:33-13:43, 1-2 workers):

| test (x10) | in R1 (-n 4, loaded) | alone |
|---|---|---|
| test_reverse::test_mul_rev_the_largest_set | 1019 s | 42 s (file collection only); 35 s with the whole suite collected (`-k`) |
| test_minmax_fma::test_fma_is_mul_then_add_on_exact_sets | 776 s | 22 s (file alone: 98 s for the whole file) |
| test_relations::test_allen_semantics | 505 s | 14.7 s alone; 23.2 s after tests/itf1788 in the same process |
| test_relations::test_set_relations | 597 s | < 10.6 s alone; 17.4 s after itf1788 |

hypothesis's statistics say where: in R1 `test_mul_rev_the_largest_set` reports "~12-1420 ms in data generation" per case,
alone "~3-30 ms". the slow ones all draw `tests/strategies.py::exact_cut_tuples` (st.fractions + lists, generation-heavy).
ruled out: the full collection itself (the `-k` run with all 34941 items collected was fast) and hypothesis's local-constants
pool (`internal/conjecture/providers.py::_get_local_constants`; all local modules hold ~6000 constants, ~200 integers).
a long process does drift (itf1788 first made test_relations' tests 1.5-2x slower), but most of the 25-35x is CPU
starvation on this 2P+8E part under 80-90 % load. **so R1's 90 min is a loaded-machine number, not xdist's.**

### scaling on a fixed subset, x10 (13:43-13:59, after the prepush; load light: cpu 15-30 %, 4-8 other python.exe)

subset S = tests/test_relations.py tests/test_minmax_fma.py tests/test_orders.py (180 items), `xrun.sh S_x10_n<K> <K>
HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 -- <S>` (`scale.sh`), logs `S_x10_n*.log`.

| workers | wall s | speedup | sum of item durations s | longest item s |
|---|---|---|---|---|
| serial | 477 | 1.00 | 475.0 | 42.9 (test_orders::test_orders_sound_at_sampled_points) |
| -n 2 | 234 | 2.04 | | |
| -n 4 | 137 | 3.48 | | |
| -n 8 | 103 | 4.63 | 478.6 | 40.5 (test_orders::test_orders_match_1788_definitions) |

180 passed in every one. at -n 8 the sum of per-item durations did not grow (478.6 vs 475.0 s): with the laptop quiet,
8 workers did not slow each test down; the loss from 8x to 4.6x is the floor (40 s items) plus per-worker startup
(each worker collects the suite: ~15 s) and the tail.

### R3: full suite, x10 fuzz profile, xdist -n 8 --dist worksteal (13:59:53-14:20:52, prepush finished, light load)

command: `xrun.sh R3_x10_n8 8 HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 -- --hypothesis-show-statistics`. log `R3_x10_n8.log`,
aggregate `R3_durations.txt`. load: only another repo's hawker_bot besides mine; cpu% 8 at start, 22 mid-run; 8 workers
at ~95 % of a thread each (1080-1145 s CPU each at 20 min).

* **wall 1257 s (21.0 min), 34941 passed, rc 0: the same count as the serial gate/fuzz (27798 + 7143), nothing failed.**
* vs serial x10 (itf + rest) on the ledger: 4450 s (2026-10-05) / 5407 s (10-04) / 7495 s (today, loaded): **3.5x-6x**.
* sum of per-item durations 9801 s (8 x ~1225 s: balanced to the end). that is ~2.2x the serial CPU of a quiet day
  (~4400 s): on the 2P+8E i7-1365U each of 8 workers runs at about half a P-core.
* x10 shares: top 1 / 10 / 50 items = 2.1 / 11.3 / 28.5 %; top 10 / 50 functions = 19.8 / 52.4 %.
* **longest single item at x10 (the floor for any split by test): tests/test_solve.py::test_every_zero_is_enclosed
  209 s**, then test_solver::test_every_zero_is_enclosed 191 s, test_solve::..._on_a_budget 120 s, test_pown_huge::
  test_identity_with_the_exact_construction 105 s, test_reverse::test_mul_rev_isotone_and_distributive 91 s.
* the timing test passed this time: test_fmt `[many-separators]` 5.00 s (< 10), so its flakiness depends on load.
* hypothesis statistics: 1389 test runs "Stopped because settings.max_examples", 1 "nothing left to do" (an exhausted
  space). so at x10 exhaustion is rare here; the earlier sub-10x guess from tree exhaustion was wrong (corrected).

### Q3 measured: 1 x (x10) vs 10 x (x1) vs xdist, on the 3 slowest x10 property tests (14:28:56-14:38:22)

tests: test_solve::test_every_zero_is_enclosed (pins 10), test_solver::test_every_zero_is_enclosed (pins 40),
test_solve::test_every_zero_is_enclosed_on_a_budget (pins 40). `q3.sh 5 <3 ids>`; fuzz profile, observability without
coverage (`HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER=1`), one `HYPOTHESIS_STORAGE_DIRECTORY` per process; logs and
observations in `q3/`, table `q3/times.tsv`, distinct counts `q3_obs.txt` (`obs.py`). load: light (hawker_bot only, cpu ~10-20 %).

| arm | wall | process-seconds | explicit-example calls | generated cases (valid) | distinct valid generated inputs |
|---|---|---|---|---|---|
| A: 1 process at x10 | **313 s** | 313 | 11 | 1135 (900) | 99 + 391 + 397 = **887** |
| B: 10 processes at x1, 5 at once | **152 s** | **635** (49-88 s each) | **110** | 1302 (900) | 100 + 349 + 394 = **843** |
| C: xdist -n 3 at x10 | **101 s** | ~300 | 11 | (as A) | (as A) |

* B buys its wall time with 2x the compute (635 vs 313 process-seconds): every process repeats the @examples (6+4+1 per
  run, the 6 of test_solve's alone cost ~33 s each run, see above), collection/import (~1-2 s), and the zero/simplest case.
* B is not a better search: fewer distinct valid inputs (843 vs 887). on test_solver's test, 15 inputs were generated by
  more than one of the 10 runs and 349 of 400 valid cases were distinct (A: 391 of 400); B also hit more invalid draws
  (179 vs 122 gave_up on that test), and its inputs were smaller (mean repr length 115 vs 127, p90 143 vs 170): the
  size-capped early phase (`small_test_case_cap`) is 10 % of each x1 run vs 5 % at x10.
* C (xdist) needs no change to hypothesis's search at all and was fastest here because the three tests are independent.
  B's one real advantage is that it can split a single test: with 10 processes a 209 s test becomes ~10 x 25 s. xdist
  cannot go below the longest test (209 s at x10). but the whole-suite floor is 209 s and R3's wall was 1257 s, so the
  floor is not what bounds the suite; the CPU is.
