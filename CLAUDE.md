# intervals: working rules

the durable contract. what is true now (open items, questions, the session log) is in `HANDOFF.md`;
read it in full at session start. the design is `v2-plan.md`, the milestones `v2-implementation-plan.md`.

## what has run on this code: `tools/gate.py status`

the run ledger (`.gate-runs/`, gitignored) answers "did the gate and the fuzz run on the code in
front of me?" so memory does not have to. every gate and fuzz run goes through `tools/gate.py run
<phase>`, which records the verdict against content ids of the code (not the commit: a run before
`git commit` still counts after it; a markdown edit stales nothing, a README edit only the doctests).
`tools/gate.py status` prints what is green on this code, whether a commit is covered, and what a
push still needs. read it at session start and before every commit and push; a run made outside
`tools/gate.py` is not in it.

## gate (before every commit)

`python -m pytest -q` from the repo root, as two recorded calls (past the 10-minute tool limit in
one): `tools/gate.py run gate:itf` (~1 min), then `tools/gate.py run gate:rest` (~10-14 min, in the
background). commit when `tools/gate.py status --require commit` exits 0. python is
`C:/Users/user/anaconda3/envs/intervals/python.exe`.

## push (owner, 2026-09-29: fuzzing runs fully autonomously or not at all)

nobody reads CI email. a push is only done by a session that stays to see its runs finish:

1. **before**: `tools/prepush.sh` on the committed tree (background it). it runs what the ledger says
   is missing: the fuzz job of `.github/workflows/fuzz.yml` run locally (`HYPOTHESIS_PROFILE=fuzz`,
   x10, every test, so it covers the gate; 27795 in 53 s + 5611 in 4899 s, about 83 min on this
   laptop, 2026-09-29 at `97d9824`) unless a green x10 run on this source is already recorded; the
   READMEs' doctests if only a README changed; nothing if only markdown or `references/` changed
   since `origin/master` (owner, 2026-09-30: docs skip the fuzz). push only if it exits 0 (its last
   step is `tools/gate.py status --require push`).
2. **push** `master` (pushing still needs the owner's go).
3. **after**: a babysitter agent runs `tools/ci_watch.sh <sha>` in the background: it waits for both
   workflows of the commit (`ci`, about 12 min; `fuzz`, about an hour), prints each result, and for
   a red run saves the failed log and artifact to `.scratch/ci/<run>/` and prints the falsifying
   example. the babysitter reproduces a failure locally and diagnoses it, library or test oracle,
   with evidence; it does not push.
4. **at session start**, if the last push's runs were not seen to finish: `tools/ci_watch.sh`
   (origin/master's commit).

## CORE-MATH worst cases: a manual check, local only (owner, 2026-10-02)

the gate runs a vendored sample of CORE-MATH's hard-to-round inputs (`tests/test_coremath.py`, rows in
`tests/coremath/`). every input of the pinned files, three directions against MPFR, is `tools/coremath.py check
--jobs N`: never in CI, never in prepush, run by hand. at session start, beside `tools/gate.py status`, read
`tools/coremath.py status`; when it shows the scalar evaluator (elementary and what it imports) changed a lot since
the last full check, or CORE-MATH changed our functions' files upstream, ask the owner whether to run it.

## how to run each test

the repo skill `testing` (`.claude/skills/testing/SKILL.md`): the gate, the fuzz profile, prepush,
the CI watch, reproducing and pinning a fuzz failure, sabotage checks, the exhaustive harnesses.

## hygiene

* `.scratch/` is the gitignored throwaway area; a finding recorded only there is lost.
* except `.scratch/coremath-cache/`: do not delete it, not in a `.scratch/` sweep either. it holds the pinned
  CORE-MATH files `tools/coremath.py` reads (about 280 MB, their sha256 in `tests/coremath/MANIFEST.tsv`); if lost,
  `tools/coremath.py fetch` downloads them again.
* stop a background process by the PID it had at launch, never by image name or command-line match.
  a monitor's `tail -f` outlives its session: stop it when done (sixteen orphans from earlier
  sessions held `.scratch/` open until 2026-09-29).
