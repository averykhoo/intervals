# multiinterval: working rules

the durable contract. what is true now is in `HANDOFF.md` (read it in full at session start) and on the
board (`context-engine task list`). decisions: one record each in `docs/decisions/` (`D-n`), written through
`context-engine`, never in HANDOFF; `docs/decisions.md` is the log they were imported from on 2026-11-02, frozen.
records of finished work: `docs/records.md`. the v2 plans (design, milestones, every record up to v2) are
archived, read-only, in `docs/archive/v2/`.

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
`C:/Users/user/anaconda3/envs/intervals/python.exe`. then the record engine's lint (seconds), so the whole
gate is `python -m pytest -q && context-engine lint`.

## push (owner, 2026-09-29: fuzzing runs fully autonomously or not at all)

nobody reads CI email. a push is only done by a session that stays to see its runs finish:

1. **before**: `tools/prepush.sh` on the committed tree (background it). it runs what the ledger says
   is missing: the fuzz job of `.github/workflows/fuzz.yml` run locally (`HYPOTHESIS_PROFILE=fuzz`,
   x10, every test, so it covers the gate; on `GATE_WORKERS` pytest-xdist processes, default 8 (D34):
   27798 in 28 s + 7182 in 1547 s, about 26 min on this laptop, 2026-10-08 at `662e5f1`; serial it was
   about 83 min) unless a green x10 run on this source is already recorded; the
   READMEs' doctests if only a README changed; nothing if only markdown or `references/` changed
   since `origin/master` (owner, 2026-09-30: docs skip the fuzz); and the whole suite on the forced
   gmpy2 backend (`gate:gmpy2`, 206 s on 8 workers, 2026-10-08) when a file of `tools/gate.py::BACKEND_FILES`
   changed since the base (owner, 2026-10-03: Q16(e); CI's `gate-gmpy2` job runs it on every push).
   push only if it exits 0 (its last step is `tools/gate.py status --require push`).
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
--jobs N`: never in CI, never in prepush, run by hand. nothing to read at session start: at the END of a session
that changed the scalar evaluator (`elementary` and what it imports; `tools/coremath.py status` lists the files and
the commits since the last full check), ask the owner whether to run it (2547 s at 4 jobs, 2026-10-02).

## the record engine: `context-engine` (adopted 2026-11-02)

* `context-engine` means `.context-engine/env/python.exe -m context_engine` from the repo root, with
  `CE_ACTOR=claude-code/<model-id>` and `CE_SESSION=<key>` set in every command (shell state does not persist).
  that env is the engine's own, gitignored, never this repo's (`envs/intervals`). on a fresh clone, rebuild it:
  `conda create -y -p .context-engine/env python=3.12`, then `.context-engine/env/python.exe -m pip install
  "context-engine @ git+https://github.com/averykhoo/context-engine@309641bd84837277ef9ddc7251b851a4e815dbe5"`.
* `[gate]` in `context.toml` is off: `tools/gate.py` keeps the gate's ledger (its two calls outlast one tool
  call), so `context-engine commit` does not check the gate; `tools/gate.py status --require commit` first.
  `context.toml` and `docs/ledger/` are prose to `tools/gate.py` and the fuzz workflow, like markdown.
* ids: decisions `D-n` (the log's `D17` is `D-17`), board `IV-n`, owner questions `ASK-n` (`Q1`-`Q27` are all
  answered, in the decision records), stories `S-n`, goals `G-n`. never reuse or renumber one.
* commit by path through `context-engine commit <paths> -m ...` (never `git add -A`, never `git stash`); an
  agent commit ends in one trailer paragraph: `Session: <key>` and `Co-Authored-By:`, no blank line between.

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

<!-- context-engine kit 0.5 begin -->
framework: 0.5

This repo runs the context-engineering framework through `context-engine`. Its rules are the file
imported on the next line, which the framework owns: an upgrade replaces it whole, so it is never
edited here, and this repo's own rules go in this file, outside these markers. If those rules are
not in your context, read that file; if it is missing, run the engine's lint
(`<engine-python> -m context_engine lint`), which names the fix.

@docs/framework/contract.md
<!-- context-engine kit 0.5 end -->
