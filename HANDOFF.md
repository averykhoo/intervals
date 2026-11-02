# Orientation note

LIVING. What is true now, the next session's first steps, and facts not derivable from the code
(FRAMEWORK §5.1). The board is `context-engine task list`; this note keeps no copy of it.

The banner is written at the first clean close by `context-engine banner set --seen none`,
which puts a `## Banner (<session key>)` section before this note's first section. Do not
write that heading by hand.

## Banner (2026-11-02a)

- The context-engine is adopted in this repo on branch `ce-cutover` (off `fa4cf12`): the board, the decisions and
  the session ledger now go through `context-engine`. Not merged, not pushed.
- Pushed: `origin/master` is `478654a` (CI and fuzz green, 2026-10-08). Local `master` is four docs-only commits
  ahead of it; `ce-cutover` sits on top of them.
- NOW: IV-1 (x2-cosh-zero: `x ** 2` and cosh give exact 0 and 1 at an operand end 0.0; D35). The other old rows
  and the still-owed bullets are IV-2..IV-22, found by their old names too (`task show from-cuts-reason`).
- Owed to the owner: the call on merging `ce-cutover`, and the go to push (full gate and `tools/prepush.sh` first,
  then a CI watcher).

## Next session

- Take NOW from `context-engine task list`; a task the owner assigns overrides the ranking. The old board
  rows keep their names: `context-engine task show x2-cosh-zero` resolves the alias (so do `H1` and `T1`).
- Rules for gate, commit, push and CORE-MATH are `CLAUDE.md`'s. Before a commit, `tools/gate.py status
  --require commit`; before a push, `tools/prepush.sh` and a CI watcher after it.

## Pushed state (read 2026-11-02; `git log origin/master..master` regenerates it)

- `origin/master` is `478654a` (pushed 2026-10-08 with m14b-open and fuzz-xdist): CI run 37754291429 and fuzz
  run 37754291528 were both green, seen to the end.
- Local `master` is four docs-only commits ahead of it (`4d04fe3`, `acdd9c3`, `64b9dcc`, `fa4cf12`: the handoff
  prose, D35, D36 and the Q27 entries). Pushing them is the owner's go. The engine adoption sits on branch
  `ce-cutover` on top of `fa4cf12`; none of it is pushed or merged.

## Verified figures (dated; re-measure rather than difference)

- Gate on the code of 2026-10-08: 27798 + 7182 passed (`tools/gate.py status` says what is green now). The fuzz
  and gate:gmpy2 phases run on 8 pytest-xdist workers (D34); prepush takes about 30 min.
- CORE-MATH full check clean at `7dd2c0d` (2026-10-08): 17959805 inputs, 3 directions, 0 mismatches, 2930 s at
  4 jobs; the rows are in `references/coremath-runs.tsv`. Run it again only if the scalar evaluator changes.

## Release 2.0.0 (board item H1, which waits on x2-cosh-zero, from-cuts-reason and release-meta)

- `pyproject.toml` is `2.0.0.dev0`; no tag yet. Owner, 2026-09-26: release "when everything is fully done".
- release-meta builds D36: `LICENSE` (MIT), the `pyproject.toml` metadata, a tag-push publish workflow by PyPI
  trusted publishing, and `CHANGELOG.md` from 2.0.0 (its first entry the release notes, from the v1 -> v2
  surface map).
- The owner registers the pending publisher on pypi.org before the first tag. The first upload, by the publish
  workflow, is what claims `multiinterval` on PyPI.
- The CORE-MATH full check again if the scalar evaluator changed since `7dd2c0d`.

## Where things live

- Decisions: `docs/decisions/` (records); the pre-import log `docs/decisions.md` stays as the frozen original that
  the README and the code cite by heading. Records of finished work: `docs/records.md`. The v2 plans:
  `docs/archive/v2/`.
- Owner questions Q1-Q27 are all answered (D-n records); their wordings are in git history and, for Q9-Q20, in
  `references/owner-questions-2026-10-03/`. New questions are `ASK-n` on the board.
- Session history: the engine ledger `docs/ledger/session-log.md` from the adoption on; `docs/session-log.md`
  holds the entries of 2026-09-25..2026-10-05 and the banner, rows and bullets retired from the old handoff on
  2026-10-08; the eight newest entries (2026-10-05..2026-10-08, with their commit ids) are only in the archived
  old handoff, `docs/archive/HANDOFF.md`.
- The old handoff, archived whole (its banner, ranked table, still-owed bullets and rotation rules):
  `docs/archive/HANDOFF.md`. Its table rows and bullets are board items IV-1..IV-22; each cites its source row.
