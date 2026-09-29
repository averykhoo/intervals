# intervals: working rules

the durable contract. what is true now (open items, questions, the session log) is in `HANDOFF.md`;
read it in full at session start. the design is `v2-plan.md`, the milestones `v2-implementation-plan.md`.

## gate (before every commit)

`C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root. on this shared
laptop it runs past the 10-minute tool limit, so run it as two calls, `tests/itf1788` and
`--ignore=tests/itf1788`, rc captured into the log, never through a pipe.

## push (owner, 2026-09-29: fuzzing runs fully autonomously or not at all)

nobody reads CI email. a push is only done by a session that stays to see its runs finish:

1. **before**: `tools/prepush.sh` on the committed tree (background it: at x10, 27795 in 53 s + 5611 in 4899 s,
   about 83 min on this laptop, 2026-09-29 at `97d9824`). it is
   the fuzz job of `.github/workflows/fuzz.yml` run locally (`HYPOTHESIS_PROFILE=fuzz`, x10), every
   test included, so it covers the gate. push only if it exits 0.
2. **push** `master` (pushing still needs the owner's go).
3. **after**: a babysitter agent runs `tools/ci_watch.sh <sha>` in the background: it waits for both
   workflows of the commit (`ci`, about 12 min; `fuzz`, about an hour), prints each result, and for
   a red run saves the failed log and artifact to `.scratch/ci/<run>/` and prints the falsifying
   example. the babysitter reproduces a failure locally and diagnoses it, library or test oracle,
   with evidence; it does not push.
4. **at session start**, if the last push's runs were not seen to finish: `tools/ci_watch.sh`
   (origin/master's commit).

## fixing a fuzz failure

* reproduce with the log's "Falsifying example" pasted as an `@example` on the failing test (an
  `st.randoms()` argument prints as `HypothesisRandom(...)`: give it `random.Random(0)` and check the
  example still fails). the `@reproduce_failure` blob only works with CI's exact hypothesis version.
  the artifact's `.hypothesis/` copied into the repo root should replay the run's examples (the fuzz
  profile keeps a database on CI since 2026-09-29; the runs before that saved none). not yet tried
  on a run that saved one, and a database from a different hypothesis version may not replay: fall
  back to the `@example`.
* decide library or oracle before fixing; keep the `@example` as the pin, and show it red on the old
  code (clear `__pycache__`, `PYTHONDONTWRITEBYTECODE=1`) and green with the fix. record the find in
  `v2-implementation-plan.md` §2 (as "fuzz-symmetry" and "fuzz-floordiv-overflow" are) and in
  `HANDOFF.md`.

## hygiene

* `.scratch/` is the gitignored throwaway area; a finding recorded only there is lost.
* stop a background process by the PID it had at launch, never by image name or command-line match.
  a monitor's `tail -f` outlives its session: stop it when done (sixteen orphans from earlier
  sessions held `.scratch/` open until 2026-09-29).
