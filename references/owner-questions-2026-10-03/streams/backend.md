# owner answers: Q16 backend (references/owner-questions-2026-10-03/backend.md), implementation record

NOTE: the harness refused writes to the main checkout's .scratch/ (worktree isolation), so this record lives in
the worktree: C:\Users\user\PycharmProjects\intervals\.claude\worktrees\agent-ae2a001020e4db7d4\.scratch\owner-answers\backend.md

agent worktree branch worktree-agent-ae2a001020e4db7d4, fast-forwarded from 912558b to master 09435ca (the
report lives there). started 2026-10-03.

## design decisions (from the report)
* Q16(e) push requirement: the report's E2 recommendation says the gmpy2 phase is "required before a push only
  when the evaluator's files changed". implementing: tools/gate.py plan requires a green gate:gmpy2 on this src
  when any of intervals/{backend,_gmpy2,elementary,ops}.py changed since the base. commit does NOT require it
  (CLAUDE.md gate is the pure path).

## step 1, Q16(d): [fast] pinned (done 2026-10-03)
* pyproject.toml `fast = ["gmpy2>=2.3,<3"]` (+ two comment lines); the `test = [...]` line untouched.
* tests/test_backend.py::test_the_fast_extra_installs_what_auto_takes (new sibling of
  ::test_the_test_extra_installs_what_auto_takes): asserts extras['fast'] == [f'gmpy2>={FLOOR},<{CEILING}'].
* run: `-k extra_installs`: 2 passed (2026-10-03).
* sabotage (2026-10-03): fast back to "gmpy2>=2.3" -> the new test red (1 failed, 1 passed); fast "gmpy2>=2.4,<3"
  -> red (1 failed); restored from a copy, cmp equal, CRLF kept.

## step 2, Q16(e): CI job + gate:gmpy2 ledger phase (code done 2026-10-03)
* .github/workflows/ci.yml: new job `gate-gmpy2` ("gate (gmpy2, python 3.13)", ubuntu, timeout 30, job env
  INTERVALS_BACKEND: gmpy2), steps checkout, setup-python 3.13, `pip install -e ".[test]" numpy` (NOTE for the
  numpy stream: this install line also carries the trailing `numpy` token; drop it with the others),
  `python -c "import intervals.backend as b; print(b.name()); assert b.name() == 'gmpy2'"`, `python -m pytest -q`.
  existing jobs untouched; header comment line 1 reworded. fuzz.yml untouched (no gmpy2 fuzz job).
* tools/gate.py: PHASE_RE gains `gate:gmpy2`; GMPY2; BACKEND_FILES = intervals/{_gmpy2,backend,elementary,ops}.py;
  ::phase_spec now owns INTERVALS_BACKEND for every phase (None = removed for the pure phases; 'gmpy2' for
  gate:gmpy2, whose args are [] = the whole suite incl. README/module doctests, as CI runs it);
  ::run_phase no longer pops the variable itself; ::_key (src for fuzz and gate:gmpy2); ::gmpy2_covered (keyed by
  src); ::gate_covered excludes gate:gmpy2 (commit = pure path); ::plan returns words joined by '+' (fuzz|docs,
  then gmpy2): gmpy2 needed iff changed is None or a BACKEND_FILES path changed since the base and no green
  gate:gmpy2 on this src; ::report lists gate:gmpy2 and prints the joined plan; module docstring paragraph.
* tools/prepush.sh: loops over the plan's '+' words; `gmpy2` runs `tools/gate.py run gate:gmpy2`; PREPUSH_FULL
  keeps gmpy2 if planned. exercised with a stub python over every word x FULL (2026-10-03): calls as expected,
  bogus word -> rc 2.
* COMMIT/PUSH: commit does not require gate:gmpy2. push requires it only when a backend file changed since
  origin/master (report E2). => CLAUDE.md push §1 (prepush description) needs a line for the orchestrator:
  "and gate:gmpy2 (the whole suite forced to gmpy2, ~10-15 min) if intervals/{backend,_gmpy2,elementary,ops}.py
  changed since origin/master and no green gate:gmpy2 run on this src is recorded".
* tests/test_gate_ledger.py: ::test_a_phase_gets_its_environment (gate:gmpy2 sets gmpy2 whatever the caller has:
  auto/python/unset; docs strips), ::test_phase_names_are_closed (full specs, gmpy2 spec, more bad names),
  ::test_a_verdict_is_keyed_by_phase_and_code (gmpy2 row covers no gate part), ::test_the_push_plan (moved the
  non-backend rows to intervals/kernel.py; gmpy2 rows per BACKEND_FILES, src keying, red rerun, docs+gmpy2),
  ::test_the_plan_end_to_end (backend file commit -> fuzz+gmpy2 -> gmpy2 -> red gmpy2 run -> green -> README
  edit -> docs), new ::test_ci_runs_the_gmpy2_phase (ci.yml job = phase spec, only that job sets the variable,
  name step before pytest, fuzz.yml sets nothing), new ::test_the_backend_files_are_the_modules_naming_it.
* run 2026-10-03: tests/test_gate_ledger.py 47 passed; with tests/test_backend.py and README.md: 786 passed.

### claim checked: "forced cannot pass on the pure path" (2026-10-03, .scratch/probe_forced.py, probe4.py)
* a fake gmpy2 on PYTHONPATH (missing: raises ImportError; old: version 2.2.1/MPFR 4.2.1):
  forced `import intervals` rc=1 with the backend's ImportError (both fakes); auto -> 'python', rc 0.
* the WHOLE suite forced with gmpy2 missing: rc=2, "Interrupted: 94 errors during collection" -> holds for the
  job's command.
* BUT one file alone passes: `pytest tests/test_cuts.py` forced with gmpy2 missing: 24 passed (+1
  PytestConfigWarning). cause: pytest's filterwarnings line imports intervals.errors, the package import fails
  half-way and leaves intervals.cuts/kernel/fmt/rounding/errors in sys.modules, so `from intervals.cuts import`
  never re-runs __init__. so the report's claim is true for the whole suite, not for a subset; the CI job's
  explicit `assert b.name() == 'gmpy2'` step is load-bearing and is pinned (::test_ci_runs_the_gmpy2_phase).

### sabotage table (2026-10-03, .scratch/sabotage_ledger.py; each one exact replacement, restored byte for byte)
| # | break | result | red tests |
|---|---|---|---|
| S1 | gate:gmpy2 spec strips the variable | RED 3 failed | phase_gets_env, ci_runs_gmpy2, phase_names_closed |
| S2 | gate:gmpy2 runs PARTS['rest'] only | RED 2 | ci_runs_gmpy2, phase_names_closed |
| S3 | run_phase pops INTERVALS_BACKEND after the changes | RED 1 | phase_gets_env |
| S4 | plan ignores backend files | RED 2 | push_plan, plan_end_to_end |
| S5 | gmpy2 verdict keyed by code | RED 2 | push_plan, plan_end_to_end |
| S6c | gate_covered accepts gate:gmpy2 (exclusion and suffix filter both) | RED 1 | verdict_keyed_by_phase_and_code |
| S6/S6b | only one of the two guards removed | green (expected: belt and braces, the other guard holds) | - |
| S7 | ops.py dropped from BACKEND_FILES | RED 3 | backend_files_naming_it, push_plan, plan_end_to_end |
| S8 | ci.yml gmpy2 job without the env | RED 1 | ci_runs_gmpy2 |
| S9 | ci.yml gmpy2 job runs tests/test_backend.py only | RED 1 | ci_runs_gmpy2 |
| S10 | ci.yml gmpy2 job without the name step | RED 1 | ci_runs_gmpy2 |
| S11 | fuzz.yml job sets INTERVALS_BACKEND (a gmpy2 fuzz job) | RED 1 | ci_runs_gmpy2 |
| S12 | the pure gate job sets INTERVALS_BACKEND | RED 1 | ci_runs_gmpy2 |

## step 3, Q16(b)/(f): README (done 2026-10-03)
* README.md "a faster backend, optional": `auto` now says "else the pure path, silently"; the speed clause stays
  qualitative, no number ("a few times for the elementary functions at a float, less over a whole set, little
  for arithmetic"); a new sentence after the example block: "reporting a result, say which backend computed it:
  `intervals.backend.name()` is `'python'` or `'gmpy2'`" (prose, not a doctest: it differs per backend).

## step 4: tools/backend_speed.py docstring (done 2026-10-03)
* `h3-records/gmpy2.md` -> `v2-implementation-plan.md` §2 M16e, "speed" (the tables are at the "**speed**" bullet
  of the M16e section, ~line 3817).

## local gate:gmpy2 whole-suite run: launched 2026-10-03 (worktree ledger), result below when done

## step 5: docs (done 2026-10-03/04; prose only, outside the ledger ids)
* .claude/skills/testing/SKILL.md: "which run, when" before-a-push row (+10-15 min if a backend file changed);
  ledger section: phases list + `plan` words + which phases strip the variable + a new `gate:gmpy2` paragraph
  (whole suite, CI job, cannot pass pure except one file alone, src-keyed, never the gate, push only on
  BACKEND_FILES change, no gmpy2 fuzz job); prepush section (gate:gmpy2 besides); "other tools" bullet (CI no
  longer "only the differential").
  NOT changed: the skill's "its sabotage table (13 breaks, each red) is in v2-implementation-plan.md §2 'run
  ledger'" -- the S1-S12 table above belongs in that §2 record; left to the orchestrator (§2/§0 are theirs).
* v2-plan.md current design: "the backend" bullet (README names backend.name(); [fast] pinned to auto's window,
  owner Q16(d)); testing section "the backend differential" bullet (gate-gmpy2 job, gate:gmpy2 phase, push-only
  requirement, cannot-pass-pure caveat, no gmpy2 fuzz job; replaces "the build ran the whole gate once more
  forced to gmpy2"); "later" non-dyadic bullet (kept until a workload measures it, owner Q16(c)).
  NOT changed: decision log (the 2026-09-28 M16e revision says "[fast] (gmpy2>=2.3, unpinned); no CI change" --
  historical, correct as of then; the orchestrator's new decision-log entry supersedes it).
* for the orchestrator: CLAUDE.md push §1 needs the gate:gmpy2 clause (see step 2); HANDOFF "still owed" M16e
  bullet ("only one local whole-suite run") closes once CI's gate-gmpy2 job has run green after the push.

## the local gate:gmpy2 whole-suite run (the evidence the CI job will pass)
* `tools/gate.py run gate:gmpy2` (worktree ledger, not the main checkout's), 2026-10-03 23:59 -> 00:12:
  **PASSED, 33833 passed in 774.5 s (777 s wall), rc=0**, on c:80b912f84d31 s:7dd0bf154a6f (HEAD 09435ca+dirty =
  this branch's changes before commit; the commit carries the same bytes). command
  `python -m pytest -q -p no:cacheprovider` with INTERVALS_BACKEND=gmpy2; py 3.13.15, hypothesis 6.167.1,
  pytest 9.1.1, numpy 2.5.2, gmpy2 2.3.1 (MPFR 4.2.2), windows, shared loaded laptop.
  (2026-09-28's forced run was 4805 + 18246; today's suite is larger: 27795 itf + 6038 rest incl. doctests.)
* pure gate on the same code: gate:itf PASSED 27795 in 58 s (2026-10-04 00:12); gate:rest running.
* pure gate:rest PASSED 6038 in 763 s (2026-10-04 00:13); `status --require commit` exit 0 (27795 + 6038 = 33833,
  the same count as the forced run).

## committed
* 5402c02 on branch worktree-agent-ae2a001020e4db7d4 (parent 09435ca = master). `tools/gate.py plan --base master`
  after the commit: `fuzz` (src changed: gate.py, prepush.sh, ci.yml, tests, pyproject, backend_speed.py), no
  gmpy2 (no BACKEND_FILES changed). not pushed.
* reproducers kept beside this file (worktree .scratch, gitignored, lost with the worktree): sabotage_ledger.py
  (S1-S12; `python .scratch/sabotage_ledger.py [S1 ...]`), probe_forced.py and probe4.py (the forced-pure claim;
  they recreate their fake gmpy2 dirs, probe4 needs probe_forced's `fake-gmpy2-missing` dir first).
* left undone: CLAUDE.md push §1 clause, HANDOFF, v2-plan decision log, plan §0 D row / §2 sabotage record
  (orchestrator's, per the brief); the skill's "13 breaks" count of the ledger's table not updated (it points at
  plan §2, which the orchestrator extends with S1-S12).
