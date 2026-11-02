---
type: Runbook
title: "Fan-out: delegating to agents, workflows, and the build rounds"
source: FRAMEWORK §7.5, §6.6, P10, P11, P12
created: 2026-10-13
mode: replaced
liveness: LIVING
---

# Fan-out

How a session hands work to agents and builds in rounds. The rules are FRAMEWORK §6.6 and the
framework contract § Rules (when to use a workflow, the coordinator, model choice), and who is
sent for what is §6.6's roster; this file is how to follow them. The repo's values (its default
and cheaper models, for one) are in its `CLAUDE.md`.

## Delegating reading and auditing

- **The coordinator sends an agent** for every check, lookup or verification, however small
  (FRAMEWORK §6.6); only trivial bookkeeping (one engine op, `git status`) stays inline.
- **Inside a task agent, the test is:** would doing it inline flood the agent's context with
  file contents? Then delegate it and take back a table. A lookup whose file and symbol are
  already known is done inline there.
- **The prompt names** the question, the verdict form (`file::symbol` evidence, never file
  dumps), and the one file the agent persists to.
- **Persist incrementally, one file per agent:** each finding is appended as it lands, not
  buffered to the end, so an agent that dies leaves a partial result; two agents on one file lose
  work. Only an agent that can write files can persist (not `Explore`).
- **On return,** check the file exists and is not short; if it is, write the returned report to
  it before doing anything else with it.
- **An agent that delegates waits for its own agents** before it hands back: an agent left
  running in the background can die with its parent's return, its report unwritten.
- **A report is evidence, not a finding.** What heads for a record, a commit or a golden is
  checked by a verifier: another agent than the one that did the work, persisting like any
  other. The coordinator reads the verifier's report, not the files, and makes the final call.
  Reports that contradict each other are reconciled, never averaged. Agents audit; the session
  decides and deletes.
- **The verifier is a skeptic by default:** after a build or a review, it tries to refute each
  claim or finding before the session acts on any, which removes false findings cheaply.
- **Clean up:** the session deletes the run's scratch files once their conclusions are in tracked
  files (P10), in the same session.

## Watchers

A background watcher (CI after a push, a release run, a long job) only watches: no fix, no
re-run, no cancel, no push. Its brief names what it watches by commit hash, how to read it, the
verdict to return per run, and its persist file; the CI watcher's brief is in
[procedures.md § Push and CI](procedures.md#push-and-ci-68). Use the cheapest model that can
do it: haiku, or sonnet when it must read the logs.

- **Never return on the first poll.** A watcher given one long open-ended wait ends its turn
  instead of blocking, and the cost moves up to the agent that waits out the run. The loop lives
  in deterministic code (the workflow script, or the coordinator's `run_in_background` shell
  until-loop), not in the cheap model's judgement.
- **Each watcher makes one bounded blocking wait** (the adhoc trial used 9 minutes or less), then
  returns `finished` and `alive`; the loop starts the next watcher until `finished`.
- **Confirm a terminal state twice, a minute apart** (process dead, run finished) before reporting
  it: one poll can race a start and report a live run dead. A watcher's own timeout is not a finish.

## Build rounds

The steps are FRAMEWORK §6.6's, in its order and with its numbers.

| # | Step | How | Why |
|---|---|---|---|
| 1 | Map | read what exists: the item, its Read first, the code and records it touches; a mapping agent returns the files and who will own each | a plan made without the map rebuilds what is there |
| 2 | Plan, with a critic | the plan names each wave, each builder's files and the order of dependent work; a critic agent attacks it before anything is built | a plan error is cheapest to fix here |
| 3 | BDD: criteria first, `planned` | write the criteria the work must meet, status `planned`, before the code | what "correct" means is fixed before the code it judges |
| 4 | Build in waves with disjoint file ownership, and no fan-out on a dependency chain | in a wave, each builder owns files no other builder touches; work that needs another builder's output waits for the next wave | two writers on one file lose work; a chain split across agents makes each guess the other's interface |
| 5 | Integrate, with sabotage | merge the waves, run the gate, and see every new check fail ([sabotage.md](sabotage.md)) | the waves are proven together, and no new check is believed unseen |
| 6 | Adversarial review through several lenses | one reviewer per lens (for example: fidelity to the spec, a fresh deploy, checks that pass when they should fail), then a skeptic pass on the findings; the findings and what became of each go into the item's `## Log`, and a risk that stays open becomes a `residual:` line in its `## Design review` where it has one (FRAMEWORK §5.2) | one reader misses what a lens built to look for it finds |
| 7 | Flip the statuses | a criterion moves to `tested` once its claiming test exists (FRAMEWORK §3.3), and only after that test was sabotaged where the repo's contract requires it; the item closes, and closing files its residual risks in the register | a status says what is true only once it is |

## Worktrees for parallel builders

- One worktree and one branch per builder; the task agent merges, and a verifier agent checks
  the merge.
- **Create each worktree from current main,** not `origin/main`, which can be many commits behind
  the last push; a stale base is rebased before any edit.
- **Gitignored inputs (models, data, fixtures) are absent in a worktree:** point the env vars that
  name them at the main tree, or expect skipped tests and say which in the report.
- **A builder's Write tool can be confined to its worktree:** it persists its report in the
  worktree's `.scratch/`, copied out after each append, since a copy goes stale after a restart.
  Each worktree's `.scratch/` needs its own sweep before the worktree goes.
- The gate in a worktree imports that worktree's code, and the worktree is checked out with the
  line endings the byte checks expect ([gate.md](gate.md)).
- Board, ledger and other working-state writes stay with the session, in the main tree.
- A builder whose text cites a record not written yet gets that id from the session, which
  reserves it in the main tree (`context-engine id reserve`) and writes the record with `--id`
  ([manual-mode § context-engine id reserve](manual-mode.md#context-engine-id-reserve)); never a
  placeholder such as `DEC-<NEW>`, nor a guessed number.
- Commit by path; never `git stash`: in a shared tree it swallows another session's work (P12).
- A builder's commit carries the session's key like any agent commit: its message ends in one
  trailer paragraph, `Session: <key>` and `Co-Authored-By:` together with no blank line between
  them ([manual-mode § context-engine commit](manual-mode.md#context-engine-commit) gives the
  template). Git reads trailers only from the last paragraph, so a blank line inside it makes the
  commit read as foreign once it is merged.

## Workflows

- A workflow script is a tracked file, so a run can be repeated and reviewed (FRAMEWORK §6.6).
- Every agent in it persists its own output, as above; a fan-out that dies at its last agent is
  otherwise paid for twice.
- A workflow cannot express a dependency on another run, or on a board item's `deps`. The
  coordinator sequences runs: it launches the dependent one only after the earlier is reviewed and
  committed, and passes the output by file.
