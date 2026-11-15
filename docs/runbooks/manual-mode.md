---
type: Runbook
title: "Manual mode: every engine operation done by hand, and why each step exists"
source: FRAMEWORK P16, §7.5
created: 2026-10-08
mode: replaced
liveness: LIVING
---

# Manual mode

Every operation the engine (`context-engine`, `python -m context_engine`) performs, written as steps a person
or an agent can follow with a text editor, git and a SHA-256 tool. It exists for two reasons
(FRAMEWORK P16):

1. **Fallback.** A repo with no Python can still run the framework: slowly, by hand, with the
   same files and the same rules.
2. **Reference behaviour.** Each step says **why** it exists. When the engine errors, refuses
   oddly or writes something that looks wrong, this is the statement of what the right result
   was, so the agent can finish the job by hand *and* fix the engine.

In the engine repo, a test fails the gate if any `context-engine` subcommand has no
`### context-engine <group> <op>` section here, or a section has no **Why**. The order in which
a procedure runs these operations is the companion runbook, [procedures.md](procedures.md).

## When the engine and this runbook disagree

- **A refusal is not a bug.** Refusals are the engine doing its job: every one prints a remedy.
  Do what the remedy says. Only treat a refusal as a bug when the Why below says the write should
  have been allowed.
- **The engine is wrong** (it wrote what this runbook says it must not, or refused what it must
  allow): finish the step by hand from this runbook, then fix the engine with a test that fails
  first (break the code, watch the test go red, restore), and name the step here in the commit message.
- **This runbook is wrong** (the engine is right and the text is stale): fix the text in the same
  commit as the code it describes. The Why is the tiebreaker: whichever side serves it wins.
- **Neither is clearly right:** it is a decision. Record it as a decision, or ask the owner if it
  changes what they asked for.

## When the MCP tools are missing

The MCP server is a thin layer over the same functions the CLI calls, so nothing is lost
without it:

1. Run every operation as `<interpreter> -m context_engine ...` for the rest of the session.
2. Register the server once for this machine: `claude mcp add --scope local context-engine --
   <interpreter> -m context_engine.mcp`. Check it with `claude mcp get context-engine`.
3. Tell the owner to restart Claude Code; the tools appear at the next start. The first tool
   call that reaches the engine writes `<lockfile dir>/mcp-seen` (by default
   `.context/mcp-seen`, never tracked), and orient's `MCP server not seen` part goes; deleting
   the file brings it back.

**Why:** Claude Code starts MCP servers only when a session launches, so nothing done mid-session
brings the tools in. The local registration carries the interpreter's absolute path outside the
repo, so no machine path is checked in. A repo that also keeps a `.mcp.json` naming the
interpreter through a variable (`${CE_PYTHON}`, say) gets it only if the variable was in Claude
Code's starting environment; an `env` block in `.claude/settings.local.json` reaches the agent's
shell but not `.mcp.json`.

## What every write does

These hold for every operation below. The per-operation sections only list what is extra.

| Rule | By hand | Why |
|---|---|---|
| **Take the lock** | No lock exists by hand. Instead: re-read the file immediately before editing it, and run `git status` and `git log -3` before committing. | Several sessions share one checkout. The engine's OS lock stops two writers losing each other's change; by hand, re-reading is the only defence. |
| **Check the caller** | Know your session key (like `2026-10-08e`) and your actor (`human:<id>`, `process:<id>`, or `<producer>/<version>` such as `claude-code/<model-id>`). Every write below records them. A subagent working one task under the coordinator's key also sets its lane, `CE_AGENT=<task id>` (the global `--agent`, MCP `agent`): every line it writes carries `"lane": "<task id>"`, so its commits take its own lines and no one else's. | Each write has to say which session made it and who did it. Without that, nobody can find a change or trace it back (OKF §7). |
| **Validate before writing** | Check the result against its kind's schema (table below). If it fails, do not save it. | A bad record that reaches disk breaks every later reader. The engine checks before writing any byte. |
| **Change only your own lines** | Edit only the frontmatter keys the operation names. Keep every other line exactly as it was: order, quoting, comments, unknown keys. Keep the file's line endings. | Byte-stable writes keep diffs reviewable and keep keys that other tools own (OKF §4.1). |
| **YAML shapes** | Lists in flow style: `deps: [PRJ-2, PRJ-3]`. Dates unquoted: `created: 2026-10-08`. Null is the bare key: `closed:`. Quote any value with `: ` in it. | Quoted dates fail the `date` type, and an unquoted colon makes the YAML mean something else or fail to parse. |
| **New keys go last** | A key the record does not have yet (`status`, `closed`, `answered`, `done`, `evidence`, `resumed`, `expired_to`, `body_sha`) is added after the last key. An existing key is replaced where it stands, keeping an end-of-line comment. | Keeps every untouched line in place, so the diff shows only the change. |
| **New records** | Frontmatter, the closing `---`, one blank line, the body, one final newline. `deps: []` is written even when empty. | One shape for every record keeps hand-written and engine-written files identical. |
| **Bump `updated`** | On every write to a board item (task or question), set `updated: <session key>`. | It is how a reader sees a record was touched this session without reading the op log. |
| **Log the operation** | Append one line to `docs/ledger/ops.jsonl` (format below). | The op log is the audit trail: who changed which keys, in which session. G-W7 reads it. |
| **Commit the write** | At a milestone, commit the op log with every path its uncommitted lines name, by path and at once; every other write stays uncommitted and goes into the next commit. The milestones (`[engine] commit = "milestones"`, the default): task close; record new, amend, retire and confirm; decision supersede; ask answer; baton add; pause open; session close, session pause and session abandon; banner set. The uncommitted lines are those `git diff HEAD -- docs/ledger/ops.jsonl` adds; if git has never seen the op log, every line. Leave out a named path git ignores, or one that is gone and git never knew. `git add -- <paths git has never seen>`, then `git commit --only -m "engine: <op> <id>, <op> <id>" --trailer "Session: <key>" --trailer "Co-Authored-By: <name> <email>" -- docs/ledger/ops.jsonl <paths>` (with many paths, `--pathspec-from-file`; this operation's own `<op> <id>` first; a task close and a pause take the titles their sections give). The two `--trailer`s make one last paragraph, which is where git reads trailers (two `-m`s would split them); `<name> <email>` is the `Co-Authored-By:` value § context-engine commit resolves for the last writer of the lines committed (`CE_CO_AUTHOR`, else the actor's model), and with an actor that has no `/` (a person) leave that `--trailer` out. Take only your session's lines (their `session` is your key) of your lane (their `lane` is your `CE_AGENT`; a line with no `lane` is the coordinator's; a session close, session pause or pause record takes every lane of its session), plus another session's or lane's line naming a path you commit (that file holds its write), every decision's line when you commit the decision index (it is generated from all of them), and a line that does not parse; for those add `-m "Also carries <n> op log lines left uncommitted by session <key>, with the files they name."` before the trailer. Leave every other line uncommitted, and say whose and how many you left. With none left, commit the op log as it is; with some left, the op log you commit is HEAD's plus the lines you take, built in a temporary index: `GIT_INDEX_FILE=<tmp> git read-tree HEAD`, then with the same variable `git add -- <paths>`, `git update-index --add --cacheinfo 100644,<blob>,docs/ledger/ops.jsonl` where `<blob>` is `git hash-object -w --no-filters <a file of HEAD's op log plus the lines you take>`, and `git commit -F <message file>`, the file holding the same message, its last paragraph the `Session:` and `Co-Authored-By:` lines; then, without it, `git reset -q HEAD -- docs/ledger/ops.jsonl <paths>`. Never rewrite the working op log. The orientation note (`[working] handoff`, HANDOFF.md) is the one file that goes as less than the working file: when the lines you take name it and every one of them is `banner.set`, HEAD has it, and `git diff HEAD -- HANDOFF.md` changes lines outside the banner (a hand edit, maybe another session's), commit HEAD's note with the new banner in it, built in the temporary index: `git update-index --add --cacheinfo 100644,<blob>,HANDOFF.md` where `<blob>` is `git hash-object -w --no-filters <a file of HEAD's HANDOFF.md with its banner section replaced by the working one, in HEAD's line endings>`, and add HANDOFF.md to the `git reset`; the hand edits stay uncommitted for their author to commit by path, a note given by path to your own commit goes whole, and say `HANDOFF.md: its banner only`. Taking every line, whoever wrote it, is the sweep, for a session known dead (§ context-engine commit). Your own commits take the same lines and files with them (§ context-engine commit). Never `--no-verify`; if the commit fails, `git rm -q --cached` what you added. Do not commit during a merge, rebase, cherry-pick, revert or bisect, on a detached HEAD, with `GIT_DIR` or `GIT_INDEX_FILE` set, before `context.toml` is tracked, while the op log lacks lines HEAD has, or while a record it would carry does not parse (a hand edit in progress): commit when that is fixed. The engine does all this itself and says `not committed: <why>` in the reply of a write it would have committed, when it cannot. With `"every-write"` every write operation commits; with `"off"` the engine commits only a pause and says nothing, and the session commits no later than its close or pause (an older `context.toml` saying `true` means `"milestones"`, and `false` means `"off"`). | Lint reads the op log and the records together (G-W7), so a record committed without its line fails lint on a clean checkout. By path, so what another session staged or left untracked stays as it was. At milestones, because what they commit is costly to lose or must be at HEAD now, while a write left uncommitted costs little to lose: the next commit carries it, whoever makes it (FRAMEWORK §5.5 rule 4). |
| **Engine-owned keys** | Never hand-edit `type`, `id` or `body_sha` on an existing record. | Ids are never reused or renumbered (P3). `body_sha` pins an owner's words (G-D10). |

**Op log line**, one JSON object per line, UTF-8, `\n` endings:

```json
{"at": "2026-10-08T18:10:28+08:00", "session": "2026-10-08e", "actor": "claude-code/<model-id>", "op": "close", "id": "PRJ-8", "path": "tasks/PRJ-8-dogfood-adoption.md", "fields": ["state", "status", "closed", "moved", "updated", "body"], "mechanical": false}
```

`at` is local time with its offset. `fields` lists the keys changed, plus `"body"` if the body
changed. `mechanical` is `true` only for housekeeping and the `--mechanical` flag. `as`, only on
`hk.stamp_asked` and `task import` lines, maps a session-keyed field to the key it was set to when
that is not the writer's session: `"as": {"last_asked": "<the receipt's key>"}` (G-W7 checks the
field against it); a `task import` line maps each of `moved`, `updated`, `closed` and `last_asked`
the record holds to its kept legacy value.
`lane`, only on a line written with `CE_AGENT` (`--agent`, MCP `agent`) set, comes right after
`session`: the task id the subagent works on; a line without it is the coordinator's.
`ids` and `paths`, only on a line that stands for several records (`decision accept`), list each
record's id and file, in the same order; `id` is then the list folded into ranges (`DEC-n..m, DEC-k`)
and `path` the first file. Every file in `paths` goes into the commit that carries the line, as
`path` does.

| Operation | `op` | `id`, `path` | `fields` |
|---|---|---|---|
| any new record (task, question, decision, baton, pause) | `new` | the record | every frontmatter key, `type`, `id` and `body_sha` included |
| task set, promote, dep, comment, touch, close, reopen, section | `set`, `promote`, `dep`, `comment`, `touch`, `close`, `reopen`, `section` | the item | the changed keys, then `updated`, then `body` if the body changed |
| ask raised | `ask.raised` | one line per question, then one with id = session key, path = the ledger | `["last_asked", "updated"]`; the ledger line `["asked"]` |
| ask later, ask answer | `ask.later`, `ask.answer` (answer also writes a `new` line for its `DEC-n`; `ask new --answered` writes the question's `new` line first; with `--grant`, no `DEC-n`, and a second `ask.answer` line with id = session key, path = the ledger) | the question | the changed keys, `updated`, `body`; the `--grant` ledger line `["granted"]` |
| ask decide | a `new` line for its `DEC-n`, then `ask.decide` | the question; then a second `ask.decide` line with id = session key, path = the ledger | the changed keys, `updated`, `body` (`state`, `status`, `closed`, `last_asked`, `agent_decided`); the ledger line `["decided"]` |
| baton done, pause resume | `baton.done`, `pause.resume` | the record | the changed keys |
| baton expire | `new` then `hk.expire_batons`, both mechanical | the task, then the baton or pause | as above |
| hk close | `close`, mechanical (the task close's line) | the item | as task close |
| hk stamp | `hk.stamp_asked`, mechanical, one line per question, with `as` | the question | `["last_asked", "updated"]` |
| session start, close, pause, abandon | `session.start`, `session.close`, `session.pause`, `session.abandon` | id = session key (for abandon, the abandoned stub's), path = the ledger | `["kind"]` |
| session tasks | `session.tasks` | id = session key, path = the ledger | `["tasks"]` |
| banner set | `banner.set` | id = session key, path = `HANDOFF.md` | `["banner"]` |
| record stamp, record amend | `stamp` (one line per id), `amend` | the record | `["body_sha"]`; `[]` |
| record correct | `correct` | the record | `["body"]` (the stamp is untouched) |
| record retire | `retire`: one line for the successor, then one for the story | the record | the successor: `["supersedes"]`; the story: `["story_status", "status", "superseded_by", "body"]` (no `superseded_by` without a successor) |
| record confirm | `confirm` | the story | `["story_status", "body"]` |
| decision supersede, decision link | `supersede`, `link`: one line per record changed | the record | the changed keys |
| decision relink | `relink` | the decision | `["code"]` |
| decision set | `set` | the decision | the changed keys (no `updated`: decisions do not carry it) |
| decision accept | `accept`: ONE line for every decision accepted | id = the ids folded into ranges, path = the first file, `ids` and `paths` = every one | `["decision_status"]` |
| decision import | `import`, one line per record | the record | every frontmatter key |
| task import | `import`, one line per record, with `as` when it holds a session-keyed value | the record | every frontmatter key |
| task close of an item with residuals | `register.append`, before the `close` line | the task, path = the register | the new register ids |
| register retire | `register.retire`; with `--into`, a second line for the target | id = the entry (then the target), path = the register | `["retired"]`; the target's `["merged"]` |
| id reserve | `id.reserve`, one line per id | id = the reserved id, path = `""` (it writes no file) | `[]` |
| doc freeze | `doc.freeze` | id and path = the doc | `["liveness"]` |
| doc archive | `doc.archive`, three lines | id = the old path; path = the old path, then the archived copy, then the redirect table (`--lines`: the doc, then the archive file, then the table) | `["moved"]` (`--lines`: `["pruned"]`), `["archived"]`, `["redirect"]` |
| criterion new, criterion set | `criterion.new`, `criterion.set` | id = the criterion, path = the criteria file | `["row"]`; the header names of the columns changed |

### Record kinds (the kit's defaults)

| Kind | Id | Directory | Mode | Required keys | Enums and types |
|---|---|---|---|---|---|
| decision | `DEC-n` | `docs/decisions` | append-only | title, actor, decided, session, decision_status | decision_status: BUILT, ACCEPTED, PROVISIONAL, DEFERRED, SUPERSEDED, REJECTED; decided date; session key; tags, supersedes, superseded_by, relaxes, narrowed_by, creates, amends, retires, tasks, code, refs lists |
| story | `STORY-n` | `docs/stories` | append-only | title, actor, date, story_status, goals | story_status: live, unspecified, unconfirmed, retired; date date; goals, supersedes, superseded_by lists |
| task | `<prefix>-n`, the prefix init was given (`PRJ-n`) | `tasks` | replaced | title, pri, state, created | pri: NOW, NEXT, LATER, SOMEDAY, HOLD; state: open, closed; created date; moved, updated, closed keys; deps, labels, review lists |
| question | `ASK-n` | `tasks` | replaced | title, pri, state, created | as task, plus blocks list, last_asked key |
| baton | `BATON-n` | `docs/working` | stamped | title, session, state, created | state: open, done, expired; session, done keys; created date |
| pause | `PAUSE-n` | `docs/working` | stamped | title, session, state, created, branch, resume_step | state: open, done, expired; session, resumed keys; created date; uncommitted, deferred lists |

The Mode column is the kind's `mode` value; `stamped` is the session-stamped mode (FRAMEWORK
§8.0.1).

The repo's own `context.toml` is the authority: its `[kinds.<name>]` tables set each kind's
prefix, directory, required keys and types, and a repo may have changed any of them. Read it
before writing a record by hand.

Every record also has `type:` (Decision, Story, Task, Question, Baton, Pause) and `id:` matching
its file name. OKF `status`, if present, is `draft`, `stable` or `deprecated`; record states use
their own keys (`state`, `decision_status`, `story_status`) because OKF reserves `status`.

### Session keys

A key is `YYYY-MM-DD` plus letters: `a` to `z`, then `aa`, `ab`, and so on. Keys sort by date,
then by letter count, then alphabetically, so `z` comes before `aa`.

### The two hashes

- **Body hash (`body_sha`)**, for append-only records: take the body (everything after the
  frontmatter's closing `---` line). Cut it at the first line that is exactly `## Amendments`
  (trailing spaces allowed) if there is one. Convert CRLF to LF, strip leading and trailing
  newline characters (only `\n`: a line of spaces stays in), and add one `\n`. Then
  `sha256:` + the full hex SHA-256 of those UTF-8 bytes. A record with `corrected:` lines is
  hashed with its corrections undone first (§ context-engine record correct), so the hash always
  pins the words as typed. *Why: it makes "verbatim" checkable. Rewriting an owner's words
  changes the hash (G-D10), and amendments sit below the cut so they never break it.*
- **Banner hash**: take the `## Banner (<key>)` heading line through to the next `## ` heading,
  or to the end of the file if there is none. Convert to LF and strip leading and trailing `\n` characters. Then `sha256:` + the **first 16** hex
  characters. *Why: `banner set` must refuse when another session rewrote the banner since you
  read it (FRAMEWORK §5.5 rule 2).*

By hand: `printf '%s\n' "$text" | sha256sum` (Git Bash), or
`Get-FileHash -Algorithm SHA256` on a file holding exactly the canonical bytes (PowerShell).
The hex is **lowercase**: `Get-FileHash` and `certutil -hashfile` print uppercase, so lower it.
The bytes are UTF-8 **without a BOM**: PowerShell 5.1's `-Encoding utf8` writes one, so write
the file another way (the Write tool, or `[IO.File]::WriteAllText(path, text)`). *Why: a
different case or one extra BOM gives a hash that never matches the engine's.*

---

## Sessions

### context-engine session start

1. Read the newest key in `docs/ledger/session-log.md`. Entries are `## <key> · kind: <kind>`,
   newest first. Also read the banner key in `HANDOFF.md`'s `## Banner (<key>)`.
2. Take the larger of the two. If it is from an earlier day than today, the new key is today's
   date + `a`. Otherwise it is that key + 1 letter.
3. Insert a stub as the **first** entry, straight after the preamble's `---` line, followed by
   its own `---` separator:

   ```
   ## 2026-10-08e · kind: open

   - **opened:** 2026-10-08T19:02:11+08:00 by claude-code/<model-id>
   - **start head:** <git rev-parse HEAD, the full hash>
   ```

   Read HEAD in the same step as the key (step 1), before anything else is committed. Outside a
   git checkout, or before its first commit, leave the `start head` line out.
   **A follow-up session** (`--follows <key>`: it fixes or finishes what a closed session left)
   first checks that `## <key> · kind: close` is in the ledger, and stops if the entry is
   missing or of another kind (an open session is worked in, a paused one resumed, an
   abandoned one's baton done). Its stub then carries `- **follows:** <key>` between the
   `opened` and `start head` lines. Never edit the closed entry. Your close or pause later keeps
   the `follows` line, first in the entry, above `rows`.
   **The next turn of a conversation** (`--continues <key>`: the owner replied after that
   session's eager close) does the same with `- **continues:** <key>`: the same check that the
   entry is `kind: close`, the same place in the stub, kept the same way. A session carries one
   of the two links or neither, never both.
4. Leave it uncommitted: a session start is not a milestone, so it stays uncommitted and goes
   into the next commit, yours or another session's (§ What every write does, **Commit the
   write**).

`HANDOFF.md` may be missing, or have no `## Banner (<key>)` heading yet (before the first clean
close): then no banner key counts in step 2, and nothing refuses (`orient` says `no banner`,
`banner show` gives hash `none`, and the first close, given a banner with `--seen none`, writes
the heading).

The engine re-renders the whole ledger on every write: the preamble ends in a `---` line, entries
are separated by a blank line, `---` and a blank line, and the last entry has no `---` after it.
Match that by hand.

**Why:** two sessions minting a key at the same time could otherwise collide. The stub reserves
the key the moment it is minted, shows the session is live (`orient` lists open stubs), and
leaves evidence if the session dies (G-W11) (FRAMEWORK §5.5 rule 1). The banner key
counts too, because a session may have written the banner before its ledger entry. The start
head is where this session's reconciling of foreign commits ends and the next session's begins
(procedures.md § Reconciling foreign commits). The ledger is append-only, so a closed session is
never reopened: a follow-up links back to it with `follows`, and reads what it left (its rows,
digest and still-owed lines, which orient shows) before starting. A continuation links back with
`continues`, so a design conversation is a thread of short sessions, each closed eagerly, rather
than one session left open that looks live after it dies (G-W11); follow the `continues:` links
back to read the thread.

### context-engine session close

The close in one call: steps 1 to 3 check everything before anything is written, then step 4
runs the close housekeeping, step 5 writes the entry, step 6 the banner when one was given, and
step 7 reads the board the close leaves.
The engine makes them one commit; by hand, commit them together at the end.

1. Your stub must exist and still say `kind: open`. A session closes or pauses only once. The
   entry you write must add no G-D1, G-D5 or G-D8 failure to the ledger (as § context-engine
   banner set, in its step 4, does for the banner).
2. You need: `rows` (the items you touched, not empty); a summary of 1 to 7 lines (the owner
   digest); a `guards` receipt (gate and lint results, verbatim); a `read` receipt (what you read
   to start work). `asked`, `pushed` and `Still owed` are optional; an empty `Still owed` is
   written as `nothing`. Unless `[working] gate_receipt` is `''`, the `guards` text must match it (by
   default the kit's pattern), joined onto one line: the gate's pass count (`1112 passed`), or
   that it was not run and why.
3. Check the asked ids: the ids in your `asked` text **plus** those in the stub's `- **asked:**`
   line (written by `ask raised`). Together they must name every open owner question at NOW or
   NEXT, and every overdue one. A question is overdue if `last_asked` is empty, or at least 5
   closes/pauses have happened since it, or it is at least 7 days old. The receipt written is your
   `asked` text if you gave one, otherwise the stub's line.
   Then the stub's `- **granted:**` line (written by `ask answer --grant`), if any: it becomes
   the `granted:` receipt, and your summary, its lines joined with runs of whitespace read as one
   space, must contain each named question's `## Grant` words read the same way; stop otherwise,
   and add a line like `ASK-n granted: "<words>"` yourself.
   Then the stub's `- **decided:**` line (written by `ask decide`), if any: it becomes the
   `decided:` receipt. Each id on it must carry `agent_decided: {decision: DEC-n}` (stop otherwise:
   restore it from git), and your summary must have, for each, some one line that names both ids
   (a line may name several pairs): stop otherwise, and add a line like `ASK-n decided by the
   agent as DEC-n (overturn: ask answer ASK-n)` yourself, one per question.
   Then check the stub's `- **tasks:**` line (written by `session tasks`): your `rows` must
   name every id on it, closed or not, with how it ended.
   Then `pushed`, if given (a push orient flagged as made while no session was open): before its
   first colon, one or more hashes, comma- or space-separated, each 7 to 40 hex characters that
   `git rev-parse --verify <hash>^{commit}` resolves; after it, a note (what was pushed, and
   whether the gate ran first). Stop otherwise. Write it with its whitespace collapsed.
   Then the banner. **Only if one of its durable lines changed** (FRAMEWORK §5.1: never for
   what orient generates), you have a new banner and the hash `banner show` gave you: do §
   context-engine banner set's steps 1 and 4 now, and stop if either refuses; the receipt is
   `banner: set`. If the banner's key is this session's (you did § context-engine banner set
   before the close), the receipt is `banner: set` too. Otherwise the receipt is `banner: kept
   <the banner's key now>`; with no `## Banner (<key>)` heading at all, stop: the first close
   writes the first banner (`--seen none`). And stop if no `kind: close` entry in the ledger
   says `banner: set` or `banner: kept <the banner's key now>`: the banner predates the receipt
   and may still hold lines orient now generates, so cut it to its durable lines and write it
   (`banner: set`) this once.
4. Unless told to skip it (`--no-housekeeping`), do § context-engine hk run's close phase, and
   note its count line (`hk run close: <n> writes, <m> op log lines`) as the `housekeeping:`
   receipt. Outside a git checkout with history, skip it and write `hk run close not run: no git
   history here (run `hk run close` once there is)`.
5. Replace the stub's text, keeping its position, with (bullets wrapped at 100 columns,
   continuation lines indented):

   ```
   ## <key> · kind: close

   - **rows:** PRJ-8 (runbook written); STORY-11 (amended)
   - **receipts:**
     - guards: `<gate>` green, N passed; context-engine lint: ...
     - read: ...
     - asked: none open
     - granted: ASK-9
     - decided: ASK-10
     - housekeeping: hk run close: 0 writes, 0 op log lines
     - banner: kept 2026-10-08d
     - pushed: 5c61449: main to origin, gate green first
   - **summary** (the owner digest):
     - one line per point, 1 to 7 of them
   - **Still owed:**
     - nothing
   - **start head:** <the stub's start head, unchanged>
   ```
   Leave out the `asked:` line if it is empty, the `granted:` and `decided:` lines unless the stub has one, the `pushed:` line unless a push is recorded, the
   `housekeeping:` line if step 4 was skipped
   on request, and the `start head` line if the stub has none.
   A follow-up session's stub carries `- **follows:** <key>` (a continuation's, `- **continues:**
   <key>`): keep it as the entry's first bullet, above `rows` (a pause keeps it the same way).
   A stub's `- **tasks:**` line goes next, after `follows` and above `rows` (a pause keeps it
   the same way).
   Op log: `session.close`, fields `["kind"]`.
6. With a new banner, do § context-engine banner set's steps 2, 3 and 5, under your session
   key.
7. Last, count the open board items and those at NOW, as the close leaves them (step 4 may have
   closed one). If NOW holds fewer than its cap (`[working.caps]`, 1 by default) while any item not
   at HOLD is open, G-W5 is red: the close stands (a warning, never a refusal), but say so in the owner
   digest, with G-W5's message, naming the NEXT items (or saying NEXT is empty), and ask the owner
   which one to promote (§ context-engine task promote). Do not pick it yourself. The engine says
   the same at the end of its reply's first line, after `WARNING:`, whoever emptied NOW; it says
   nothing when NOW is at or over its cap, when nothing but held (HOLD) items is open, or when
   `[guards] off` names G-W5.

**Why:** the close is what the next session trusts. The receipts prove the gate ran and say
what was read. The asked check stops owner questions going unraised for days (G-W6, G-W8). The 7-line cap keeps the digest something the owner will actually read (FRAMEWORK §5.6).
One call that checks first and writes last is what makes a close cheap enough to make after
every reply; the banner is left alone unless a durable line changed, because orient
reads the rest from the records, and the `banner: kept` receipt is what lets G-W1 tell a close
that kept the banner from one that forgot it. An empty NOW at the close is a warning, not a
refusal: a close once everything asked for is done is legitimate, and the next NOW item is the
owner's pick, so the close only makes sure the owner hears of it.

### context-engine session pause

1. As for close step 1. You need `rows` and `deferred` (the steps skipped: gate, commit, ...).
2. Replace the stub with:

   ```
   ## <key> · kind: pause

   - **rows:** ...
   - **deferred:** ...
   - **start head:** <the stub's start head, unchanged>
   ```

   Leave out the `start head` line if the stub has none.

**Why:** a session interrupted mid-work still leaves a ledger entry (one per session, no
exceptions, FRAMEWORK §7.1). More than 2 pauses in a row without a close fails G-W2, because
skipped gates and commits pile up.

### context-engine session abandon

`context-engine session abandon <key> [--force]`, or `session abandon --stale` (every stale
stub; over MCP `session_abandon(key, force, stale)`). By hand, for each stub:

1. It must be another session's `kind: open` stub; never your own (close or pause that instead).
2. Its last activity is the newer of its newest op log line (its `at`) and its newest commit
   whose `Session:` trailer names it (as orient's `last active` shows). It is stale when 3 or
   more `session.close` or `session.pause` lines of other sessions come after that in the op log
   (and, when the last activity is a commit, are no older than the commit; with no op line and
   no commit, no older than its `opened` time). Abandon a stub short
   of that only on the owner's word.
3. Change its heading to `## <key> · kind: abandoned`, keep its lines, and add `- **abandoned:**
   <your key>: never closed or paused; last active <time>, <n> closes or pauses since` (with
   `; on the owner's word (--force)` when it was short of the window), then `- **derived
   digest:** <n> commits with `Session: <key>` (<short hashes, newest first, at most 8, then
   (+<m> more)>); `Closes:` <the ids their Closes: trailers name, or none>; <n> op log lines`.
   The commits are `git log -F --grep="Session: <key>" --format='%h %(trailers:key=Session,valueonly,separator=%x2C) / %(trailers:key=Closes,valueonly,separator=%x2C)' HEAD`,
   kept where the Session trailer is exactly `<key>`; the op log lines are those whose
   `session` is `<key>`. Say `1 commit`, `1 op log line` in the singular, and leave out the
   parenthesis when there are no commits.
4. Its op log lines HEAD lacks are what it left in flight: when there are any, file a baton (§
   context-engine baton add) naming how many and the files they name, to review them and commit
   them with `context-engine commit --sweep`, or undo them.
5. Log `session.abandon` with the stub's key as the id and the ledger as the path, then commit
   it at once (a milestone).

**Why:** a dead session must not look live, and its unfinished work must reach someone (G-W11).
The derived digest keeps what git and the op log can say about it, so a conversation that died
without a close loses only its prose judgement. By the time a stub is stale it was shown to the owner at the start of each of those sessions, so
housekeeping abandons it without asking again; the reply lists each one, so the close digest
can say so. If the owner says a stub's conversation continues here, adopt its key instead (set it
as `CE_SESSION`, or pass it as `session=`) and close it, rather than abandoning it.

### context-engine session touch

`context-engine session touch` (over MCP `session_touch`), from the live session itself. By hand:

1. Your session's stub must be `kind: open`. If it is not, there is nothing to touch: after a
   close, `session start --follows <key>`; after a pause, `session start` and resume its pause
   record; after an abandon, `session start` and do the baton it was abandoned with.
2. Log one `session.touch` line with your key as the id and the session, the ledger as the
   path and `[]` as the fields. Write no file. It is not a milestone: it goes with your next
   commit.

**Why:** a stub is stale when 3 or more other sessions closed or paused after its newest op line
or commit (§ context-engine session abandon, step 2), and a session that only reads writes
neither, so a live one could be abandoned under it. The owner's rule for the window is that a live
session resets it to zero. Until the line is committed, another
clone or CI does not see it; to make it seen, commit the op log (§ context-engine commit).

### context-engine session tasks

`context-engine session tasks <ids>` (over MCP `session_tasks(ids)`), at intake (procedures.md §
Intake, step 7). By hand:

1. Each id must be an existing item of a board kind (a task or a question), and of the form
   capital letters, a hyphen, digits (`PRJ-12`): the `tasks:` line is read back in that form, so
   an id of another form would drop out of the check unseen. For a request with no item yet, file
   a task first (§ context-engine task new). Your session must have an open stub.
2. In your stub, delete any `- **tasks:**` line and add one at the end, with no blank line
   before it: `- **tasks:** <every id named so far this session, first-named first, each once>`.
3. Log `session.tasks` with your key as the id, the ledger as the path and `["tasks"]` as the
   fields. It is not a milestone: leave it uncommitted.
4. To see where they stand (`session tasks` with no ids, and orient, which shows them to your
   session only), read each item's `state`: when every one is `closed`, do the clean close.

**Why:** the clean close fires as soon as every task the owner asked for is done (FRAMEWORK
§6.3), and without a recorded set that is judged from memory. The close
refuses rows that do not name each one (§ context-engine session close, step 3).

## Reading

### context-engine orient

Read and report, in this order, in at most 6000 bytes in total (`orient_max_bytes`), counted as
UTF-8 with LF line ends. Every part shown keeps its heading, but for one (below); a part said to
come only when something holds is left out, heading too, when it does not, and says whose bytes
its own come with. The bytes go first to parts 0, 2, 3 and 5, which a session must act on, then
to 4, then to 6 and 7, then to the banner (1), and the `Last close` part (3) takes what is left;
so cut that part first, and leave it out, heading too, when even its heading and cut note would
cut the banner; then cut the banner. Cut a part between items (a line and the indented lines
under it) and end it with a note naming the command that lists the rest (`banner show`, `baton
list`, a `grep` of the ledger for `kind: open`, `task list`, `task show <id>`, `register show
--engine`, and for each part below that comes only when something holds, the place it names):

0. only when G-D0 fails (§ lint), and `[guards] off` does not name it: a `Contract: fix this
   first` section, one line per failure with its remedy;
1. the banner (`HANDOFF.md` § Banner), headed `Banner (<key>)` with the banner's key, or headed
   `Banner` alone with the body `no banner` when there is none;
2. open batons and pauses (`docs/working/`, `state: open`), headed `Batons and pauses: do these
   first`: your own session's first, then pauses before batons, each pause with its resume step
   and branch; say `none` when there are none;
3. other sessions with an open stub in the ledger, headed `Other open sessions: show the owner,
   one line each`, which the session then shows the owner, one line each (`procedures.md` §
   Session start): the key, its `opened` value, and `; last active <time> (<what>)`, the newest
   of its op log lines (`op <op>`, by its `at`) and of the commits whose `Session:` trailer names
   it (`commit <sha>`, by committer date), so a session that died without a close shows how long
   it has been quiet; say `none` when there are none; a stub with a `follows:` line adds `;
   follows <key>` after its `opened` value; a stub with a `continues:` line adds `; continues
   <key>` the same way;
   then, only when your own stub carries `follows: <key>`, a `Follows <key>: what it closed
   with` part (its bytes come with part 4's): `- rows: ...` from that closed entry, then one
   `- digest: ...` line per summary item and one `- still owed: ...` line per `Still owed` item;
   or, when it carries `continues: <key>`, a `Continues <key>: what it closed with` part, the
   same lines after a first `- thread: <oldest> > ... > <key> > <your key>` line (follow each
   entry's `continues:` back, at most 30 keys);
   then, unless the newest `kind: close` entry is the one that part shows, a `Last close <key>:
   what it closed with` part with the same lines for it (its `- thread:` line only when it
   continued one; its bytes come after the banner's, as the ledger holds it whole): the close's
   digest is read here, not copied into the banner;
   then, only when your own stub carries a `tasks:` line, a `Tasks the owner asked for: close
   when every one is closed` part (its bytes come with parts 0, 2, 3 and 5's): one `- <id>:
   <state>` line per id the line names (`missing` when no record has the id), then `- <n> of <m>
   still open`, or `- every one is closed: do the clean close now`; the rest is `session tasks`;
4. board items at NOW and NEXT (as `context-engine task list`), headed `Board: NOW and NEXT`; say
   `empty` when there are none;
   then, only when a re-rank is due (§ The last rerank), a `Rerank due: tell the owner` part:
   one line per reason, then where the re-rank is (`procedures.md` § Rerank);
   then, each only when it has a line, in this order:
   - a `Next work waits on parked items: tell the owner` part (its bytes come with part 4's):
     for each open NOW or NEXT item, and each board id on a banner line holding the word
     `next`, whose `deps` names an open SOMEDAY or HOLD item, `- <id> (<pri>, named next in the
     banner) waits on <dep>, which is SOMEDAY (parked): drop the dep, promote it, or re-rank <id>`
     (for a HOLD dep, `which is HOLD (held)`; the `, named next in the banner` only for a banner id); the rest is `task show <id>`;
   - a `Plans to freeze: their items are closed` part (its bytes come with part 4's; off
     with G-D4): each ACTIVE-PLAN doc G-D4 reads whose named board items are all closed, with how
     to freeze it: `doc freeze <path>`, or, when it has a `[guards] liveness_given` entry, that
     entry set to FROZEN; the rest is `lint` (G-D4);
   - an `Unspecified stories and the items G-T1 counts` part (its bytes come with parts 6
     and 7's; off while `[guards] stories` is unset): each `story_status: unspecified` story,
     with the open board items naming it, which G-T1 counts (check one is the task to specify
     it), or saying no open board item names it (G-T1 is red: `task new` the work to specify it);
     the rest is `lint` (G-T1);
   - a `Housekeeping owed` part (its bytes come with part 4's): a line counting the stale `kind:
     open` stubs, with their keys (your own included, since the hook holds no key: if one is
     yours, `session touch`), a line counting the open questions an `asked` receipt names more
     newly than their `last_asked` (only with a question kind), then a last line saying to run
     `hk run start`; the rest is `hk run start --check`. It only shows what is owed: the session
     runs it (procedures.md § Housekeeping, at session start);
   - a `Foreign commits to reconcile` part (its bytes come with parts 6 and 7's): the
     foreign commits of your session's range, less those a `kind: foreign` entry names (none
     without a key), then those of the newest five abandoned sessions' ranges, at most 8 lines a
     range and then `- (+<n> more in <whose>)`, each `- <sha7> <subject> (<author>; <whose>)`,
     `<whose>` being `this session's range` or `abandoned <key>'s range`; the rest is
     `procedures.md` § Reconciling foreign commits;
   - the `Templates still unfilled: ask the owner` and `Charter unconfirmed: ask the owner`
     parts (described below);
   - a `Deprecated config: tell the owner` part (its bytes come with parts 6 and 7's):
     one line while `[working] decision_index` is set, saying the engine no longer writes a
     decision index: delete the key from `context.toml`, and `git rm` the index file when it
     exists; `decision index` prints the listing and `site build <dir>` publishes it; the rest is
     FRAMEWORK §6.5;
   - an `MCP server not seen: register it` part (its bytes come with parts 6 and 7's): one line
     while `<lockfile dir>/mcp-seen` (by default `.context/mcp-seen`) is absent, naming the
     `claude mcp add` command and the restart, and saying the CLI carries every operation until
     then; none when the lockfile sits at the repo root; the rest is kit README § 1. By hand: `ls
     .context/mcp-seen`;
   - the `Upstream` part (described below);
5. owner questions to raise now, headed `Raise in chat now, one line each`: open at NOW or NEXT,
   or overdue (see session close step 3); say `none` when there are none;
6. the NOW item's brief, `## Traps` and `## Read first`, headed `Top item: <id> <title>` (with no
   NOW item, headed `Top item` with the body `no NOW item`, and `: rerank due (above)` after it
   when a re-rank is due);
7. in a repo whose `[working] register` is not the engine's own `REGISTER.md`: one line,
   `Engine register (<version>): <N> residual risks, context-engine register show --engine`.

Two parts ask the owner for what bootstrap left unfilled (their bytes come with parts 6 and 7's),
each only when it has a line. `Templates still unfilled: ask the owner`: a line for the charter
(`[guards] charter`, else `docs/charter.md`) and one for the criteria file (`[working] criteria`,
else `docs/criteria.md`) when it is still the kit's template whole, or when it still holds one of
the template's `*(...)*` placeholders, naming each (kit README § 2a). `Charter unconfirmed: ask
the owner`: one line while the charter's frontmatter says `charter_status: unconfirmed`, an
agent's reconstruction (off while `[guards] charter` is unset or G-I1 is off): bring it to the
owner as an owner question, record the answer as a decision, then set `charter_status:
confirmed` (FRAMEWORK §3.1, §11.4). Lint fails neither, since every repo starts from the
templates and a migrated one from an unconfirmed charter, so orient asks at every session start
until they are done.

Where a branch has an upstream, the `Upstream` part (its bytes come with parts 6 and 7's): `- HEAD
is <a> commits ahead of <upstream> and <b> behind (as of the last fetch)` (`git rev-list
--left-right --count HEAD...@{upstream}`), then the newest `update by push` entry of `git reflog
show --date=iso-strict <the upstream's full ref>`: `- the last push to <upstream>: <short sha>, at
<time>` when an open session's window (its `session.start` op line to its close, pause or
abandon) holds that time, or the line saying it came while no session was open, with its
`--pushed` remedy. Once a `kind: close` entry's `pushed:` receipt names a prefix of that commit's
hash before its colon, it reads `- the last push to <upstream>: <short sha>, at <time> (while no
session was open; recorded in <key>'s close)`, naming the newest such close.

**Why:** one bounded read at session start replaces opening a dozen files. The
order is the order of urgency: a contract that did not load, then work carried over from earlier sessions and live sessions,
then new work; the banner leads as the context for the rest, but it is cut before any part but
`Last close`, so a long banner never hides what the session must act on, and a past digest the
ledger holds never hides the banner's durable lines. A `SessionStart` hook runs it in an adopted repo (kit README § Hook), with no
session key; a failing hook is silent, so a session that finds no orient output in its context
does this by hand.

### context-engine routes

Read the `[[routes]]` entries in `context.toml`, in order. Each names a component, its path
(plus `pattern`, how files under it are named), its change `mode`, and an optional `note`.

**Why:** before writing anything, know where it belongs and whether it may be edited, replaced,
or only appended to. The table used to sit in `CLAUDE.md`; it lives in config now so the engine
can check it (G-R1) and every repo can carry its own (FRAMEWORK §2).

### context-engine coverage

Read the files G-D12 reads (`[guards] coverage_spec` and `coverage_maps`; § lint): with
neither set, or G-D12 red on its config, its spec's extraction or its maps, there is no status
to give, so stop and say why. Otherwise list each component of the spec's §2 table, each §5.6
and §6.n procedure heading, each G-id it names, and each operation §8.0.2 names (§ lint, G-D12),
by class, G-ids in order of their letter and number, operations by name. Give each one the first
status that holds:

1. **built**: a G-id `context-engine lint` reports (§ lint lists every one, G-D6 among them,
   which checks nothing in the engine), or one an operation enforces at write time (G-W8,
   `session close`'s refusal; G-W10, `hk run`'s count), or an operation the engine registers
   (its name with the dot as an underscore is an MCP tool, `hk_run`);
2. **deferred**: the newest coverage map's row for it has the class DEFERRED, whatever open
   board item names it (a mention cannot tell work from an aside); note the row's reason, which
   says what un-defers it;
3. **prose-only**: that row has the class PROSE-ONLY; note the row's home, where the rule lives;
4. **open work**: the board item the row's class names (`EXISTING <id>`, `NEW <id>`) is open, or
   an open board item names it as G-D12 reads a mention;
5. **done**: the items so found are all closed;
6. **map only**: it has a row but no board item, open or closed, is so found; note the row's class;
7. **unowned**: nothing at all (G-D12 is red for it).

Beside each, name the row's board item with its state and up to five open items naming it (or,
with neither, up to five closed ones), and head the list with the totals per status.

**Why:** the spec carries no build marks, since an adopting repo's build is not the spec's, so
the status is generated from what G-D12 already reads instead of written by hand. It is read on
demand, never stored: a stored table would go stale with every board close, and a closed item
or a deferred row is the board's and the map's word, not proof the code exists.

### context-engine site build

`context-engine site build <dir>` (MCP `site_build(outdir)`), a read: it writes the records as a
static HTML site into `<dir>` and nothing in the repo (no lock, no op log line, no commit).
`<dir>` is relative to the repo root, or absolute; it must be new or empty, and outside the repo
or git-ignored in it (`git check-ignore`), or the op refuses. CI runs it on every push to `main`
and deploys `<dir>` to GitHub Pages (`.github/workflows/pages.yml`); on another host, wrap the
same op in that host's pages job. By hand, write these pages, each with a link bar to the six
listings:
1. `index.html`: the repo's name, what the site holds, links to the listings.
2. `decisions.html`: § context-engine decision index's groups and lines, each id a link to its
   page, `superseded by` ids too.
3. `stories.html`: the stories (`[guards] stories`) by `story_status` in the kind's enum order:
   id, title, actor, date.
4. `board.html`: the open items of every `[working] board` kind by tier in `pri`'s enum order,
   by prefix and number, with their `brief`; then every closed item, with its `closed` key.
5. `coverage.html`: § context-engine coverage's tables, or why it is not available here.
6. `register.html`: each `## <prefix>-n` entry of `[working] register` (`RISK-n`, or the
   prefix a register kept from before uses) with its fields, anchored `#<prefix>-n`.
7. One page per decision, story and board item at `<kind>/<id>.html`: its path, its frontmatter
   keys and values, and its body as escaped, wrapped markdown text (not rendered).

On every page, an id the site holds (a record above, or an entry id in the register) is a link to it,
and every `http(s)://` URL is a link; other ids stay text.

**Why:** a reader without the CLI (the owner on a phone, anyone on GitHub) gets the listings that
a stored index once gave, and more, without anything generated being stored: a stored copy goes
stale and couples sessions through commits. The op is the engine's, not a CI
script, so a repo on another host publishes the same site.

### context-engine lint

Check every record, then the working state. `--working` does only the second half, and the
first bullet's check that every record parses.

- **schema**: every record in a kind's directory whose name starts with its id parses (UTF-8,
  valid YAML, a closing `---`), has the right `type` and `id`, every required key, enum values in
  range, values of the declared type, and an OKF `status` from draft/stable/deprecated. A record
  that does not parse fails here by name, and every other guard passes over it. *Why: a hand
  edit is legal, so one left half-done is a failure naming the file, never a crash of lint or
  of a commit that does not carry it.*
- **G-D10**: every append-only record (decision, story) has `body_sha`; its `corrected:` lines
  each match the body; and with them undone, the body hash above matches `body_sha`. *Why:
  owner words are verbatim, so a rewrite must show up, and so must a bracket whose `corrected:`
  line was dropped.*
- **G-D11**: every `.md` in a bundle directory (`docs/decisions`, `docs/stories`, `tasks`, and
  each kind's directory), subdirectories included, parses and has frontmatter with a `type`. The exceptions are `log.md` and
  `index.md`, and `index.md` may carry no frontmatter key but `okf_version`. *Why: a typeless file is invisible to
  every tool.*
- **G-W1**: the banner key is not older than the newest `kind: close` entry, unless that entry's
  `banner:` receipt is `kept <the banner's key>` and another close says `banner: set` or kept
  that same banner; and once a session has closed, the orientation note has a `## Banner
  (<key>)` heading. *Why: a close that neither replaced the banner nor said it kept it may have
  left the next session reading an out-of-date banner; one that says it kept it judged that no
  durable line changed, and orient generates the rest (FRAMEWORK §5.1). A banner no close has
  written since closes carry the receipt may still copy what orient generates, so it is cut
  once before any close keeps it.*
- **G-W2**: at most 2 `kind: pause` entries above the topmost close (G-W2 and G-W6 read the file
  top-down; G-W1 compares keys).
- **G-W3**: no open baton or pause from a session with 2 or more closes/pauses after it. *Why:
  work carried over from earlier sessions must be done or become a task, not drift.*
- **G-W4**: on every board item, `state: closed` if and only if `status: deprecated`; no
  open question carries `answered:` (an answered question never reopens, § task reopen); every
  open question's body states its context, two or more options and a recommendation with its
  reasoning, as § ask new says; on every decision, `decision_status: SUPERSEDED` if and only
  if `status: deprecated`; and every decision that is `ACCEPTED` and not the owner's own (its
  `actor` is not `owner` or `human:<id>`, or it has `ruled_by`) is named in the `ids` of an
  `accept` op log line (§ decision accept). *Why: the owner answers a question in one line only when it carries
  what they need to decide; the doc guards and G-I3 skip a deprecated decision as history, so a
  hand-set `status` hides one; an agent's call accepted by hand leaves no record of its review.*
- **G-W5**: NOW holds exactly 1 open item (if any not at HOLD are open: a board of held items
  only may leave NOW empty). NEXT holds at most 5.
- **G-W6**: the newest close has `rows:`, a `guards:` receipt and a `read:` receipt; its
  `guards:` text matches `[working] gate_receipt` unless that is `''`; and where it has a
  `- **tasks:**` line, its `rows:` names every id on that line. *Why: the close's refusal of
  rows that miss a named task (§ context-engine session close) does not cover a close written or
  edited by hand.*
- **G-W11**: no `kind: open` stub, any session's, is stale as § context-engine session abandon
  step 2 counts it (3 or more closes or pauses of other sessions after its last activity). In a
  shallow clone (`git rev-parse --is-shallow-repository` says `true`) report nothing: the commits
  that are a session's last activity may be cut off. *Why: a dead session must not look live;
  the remedy asks the owner first, since a live session that only reads looks the same until it
  runs § context-engine session touch.*
- **G-W12**: an open item at NOW that carries `review:` passes the design-review checks of
  `task promote`. *Why: the promotion refused it, so a failing review there was edited by hand.*
- **G-W13**: every `- residual:` line of a closed item carrying `review:` has a register entry
  with the same risk text and that task in `from:`; every register entry has `affects`, `why not
  eliminated`, `control` and `from` lines; no two entries share an id or a number. *Why: the register is
  what is handed over, so a residual missing from it is a risk nobody downstream is told about.*
- **G-R1**: every `[[routes]]` path in `context.toml` exists, unless the route says
  `optional = true`. *Why: a routing table that points at a moved file sends every reader to
  a file that is not there.*
- **G-D0**: the contract (`contract` in `[engine]` of `context.toml`, `CLAUDE.md` by default)
  exists; it has at least one line starting `framework:`; every such line's value is
  `major.minor` or `major.minor.*` (no patch, no `-draft`; words after the value are a comment);
  and that major.minor, and the version in any `<!-- context-engine kit X.Y begin -->` marker,
  equal the first two numbers of the installed engine's version (`<engine-python> -m pip show
  context-engine`, or `context_engine.__version__`). *Why: the pin says which engine the repo's
  rules were written for; a stale one means the rules and the tool disagree.* When the contract
  holds a kit block of that version: the text between its markers is exactly the engine's
  `kit/bootloader.md` block (not the inline block of a kit from before `kit/bootloader.md`, not a hand
  edit, and with its end marker), and every `@<path>` line in it names a file, resolved against
  the contract's own directory, that exists and equals the engine's `kit/contract.md` once line
  endings are ignored (`git diff --no-index --ignore-cr-at-eol`). *Why: Claude Code loads the
  framework's rules through that import, and the framework owns the file; a missing or edited
  copy means a session runs on rules nobody shipped.* With that kit block, and a decision kind
  (`[working] decision`) whose `decision_status` enum lists values, the enum holds `ACCEPTED`.
  *Why: `init` never edits an existing `context.toml`, so an upgraded repo keeps the old enum, and
  its owner decisions are born PROVISIONAL against its contract.*
- **G-D1**: for each `[[routes]]` entry with `max_lines`, `max_bytes` or `max_width`, the routed
  file (or each `.md` under a routed directory) has at most that many lines and bytes, and no
  line wider than that many characters except table rows (lines starting `|`). *Why: a place
  with no cap bloats (FRAMEWORK P6); once the cap is reached, the way out is due: move the text,
  never just cut it (a rule to the contract or a runbook, a ruling to a decision, owed work to a
  task, what has gone stale to the archive, the ledger or an evidence doc), and delete it only
  once it has landed there, or rules are lost in the trimming.*
  With `--headroom`, `lint` also lists each such file at or over 95% of `max_lines` or
  `max_bytes`, with the same remedy, as a warning that fails nothing; none while `[guards] off`
  names G-D1.
- **G-D2**: in each kind's directories no two files carry the same id, by file name or by `id:`;
  and no id appears in two `new` or `import` lines of the op log. Every alias (`aliases:`, a
  record's earlier ids) is one word of printable ASCII with no quote, comma, bracket, brace, `/`
  or `::`; and, with case and leading zeros ignored, it is not the record's own id, not in the
  form of any kind's series (`<prefix>-n`) or of the criteria file's ids, not listed twice, and on
  no other record. *Why: an id is never reused (P3), and an old id must still name exactly one
  record.*
- **G-D3**: in each file `[guards] links` lists (each `.md` under a listed directory; skip a
  record with `status: deprecated`), outside fenced code: every relative link `[x](path#anchor)`
  and `[x]: path` names a file that exists with exactly that case, relative to the doc, and its
  anchor is a heading there (GitHub's slug: lower case, punctuation dropped, spaces to hyphens);
  every backticked `path::Name.attr` names a file that exists case-exact (a bare file name may be
  anywhere in the repo) whose text has each dotted part as a word. In each file `[guards] paths`
  lists, also every backticked file name or path (as G-T3 reads them, history notes skipped)
  names a file or directory that exists case-exact, from the repo root or the doc's directory
  (a bare name anywhere). A bare name not at the repo root that matches two or more files whose
  texts differ, line endings aside, is red (`git diff --no-index --ignore-cr-at-eol` tells),
  unless it is in a `paths` doc whose own directory holds that name. A file deleted but not
  staged counts as gone.
  *Why: a dead reference sends
  the reader to a file that is not there, and on Windows a link with the wrong case still opens,
  so the mistake goes unseen there.*
- **G-D4** (its liveness half only when `[guards]` lists `liveness` or `history` paths; its
  freshness half, the `stale_after` grep at the end, always): read the first
  `liveness_lines` lines (10 by default) of each `.md` there. It declares its liveness by a
  frontmatter `liveness:` key (quotes and a `#` comment allowed; any other value is red), or by a
  line outside fenced code that opens a block (the first line, or one after a blank line, a
  heading or the frontmatter) and starts, after spaces, a blockquote's `>` and `**`, `__`, `*` or
  `_`, with the whole word LIVING, ACTIVE-PLAN or FROZEN, or with `ARCHIVED <date>` (FROZEN). The
  topmost counts; the word mid-line or on a wrapped line declares nothing. A doc with none must
  match a glob of `liveness_given`, which gives it one; a doc that declares one must agree with
  its entry; a glob that matches no file is red. Under a `history` directory a doc is FROZEN or
  ACTIVE-PLAN, never LIVING. With `[working] board` set, an ACTIVE-PLAN doc names a board item in
  those lines, and at least one item it names there is open. A `liveness` list holding no doc is
  red. Whatever those lists hold, also grep for `stale_after` in every record (skip one with
  `status: deprecated`, or a `state` of closed, done or expired), in the criteria file's
  `stale_after` column (skip a `retired` row) and in the register's `- **stale_after:**` lines:
  each value is a timestamp with its offset (`2026-12-01T00:00:00Z`; a bare date is red), and
  none has passed (compare with `date -u +%FT%TZ`). *Why: a reader cannot tell whether a doc's
  status lines are still true unless it says whether it is kept current (its liveness), a plan
  left ACTIVE-PLAN after its item landed reads as work in progress, and an item past its
  `stale_after` is relied on after the date it said to re-check it by (FRAMEWORK §9.3).*
- **G-D5** (only when `[guards] numbers` lists paths): in each `.md` there whose liveness, read as
  for G-D4, is not FROZEN or ACTIVE-PLAN, outside fenced code and frontmatter, skipping every line
  with a `YYYY-MM-DD` anywhere on it (a date, a session key, a dated file name) and blanking every
  backticked or double-quoted span: no number (digits, or a spelled number from three to twenty)
  stands directly before a noun of `count_nouns`, in any case (an `open`, `closed` or `new` may
  stand between), unless `at most`, `at least`, `more than`, `fewer than`, `less than` or `up to`
  comes right before it (the number is then a bound, a rule's value, not a count). A number glued
  on its left to a letter, digit, `.`, `,`, `-`, `§`, `#` or `/` is part of a token (`R6`, `1.5`,
  `§6`, `G-D5`) and is not a count. A list whose docs are all FROZEN or ACTIVE-PLAN is red. *Why: a count typed in prose is
  never re-derived, so it goes stale unseen (P5); delete it and point at what counts it, and put
  a rule's value or a label (`Step 3 checks`) in backticks.*
- **G-D6**: nothing to check. The engine stores no generated index (listings are printed on
  demand, FRAMEWORK §6.5), so there is none to compare with its regeneration; the id stays one
  lint knows, so `[guards] off` may name it, and `coverage` reads it as built. *Why: the guard
  is for a repo's own indexes still built by parsing prose, checked from a cutoff id (FRAMEWORK
  §8.3), and the engine builds none.*
- **G-T3**: in each file `[guards] existence` lists, skipping fenced code, blockquotes and lines
  with `*HISTORY —*`: every backticked name of a file (a path with a `/`, or a name ending in a
  text extension such as `.py` or `.md`) exists case-exact, and every `file::symbol` resolves as
  for G-D3. Spans with spaces, `*`, `<`, or a leading `.` or `/` are not names. Also, every
  `code:` link of a decision that is not `status: deprecated` names a file that exists
  case-exact from the repo root and has each dotted part of its symbol as a word, by the rule
  of § context-engine decision link. *Why:
  normative text that names a file which is gone states a rule nobody can follow, and `why()`
  answers from decision links.*
- **The correctness chain** (G-I1, G-I2, G-T1, G-T2, G-T6). Each is off while a source it needs is
  unset: `[guards] charter` (G-I1), `[guards] stories`, the kind holding the stories (G-I1, G-I2,
  G-T1, G-T6), `[working] criteria` (G-T1, G-T2, G-T6) and `[guards] tests` (G-T2). A set path
  that does not exist is red. List three things: the charter's goals (each table row whose first
  cell is a goal id, `GOAL-n`, or `G-n` in a charter that uses it; a legacy `G3` is red there, but
  reads as goal 3 in a story's `goals`; `GOAL-3`, `G-3` and `G3` are one goal, by number),
  the criteria (each table row whose first cell is an id, with its `from` and `status` cells by the
  table's header; a source column may be headed `from`, `story` or `source`), and the claims (each
  match of the `[guards] claim` pattern, by default `criterion("AC-n")` or `criterion: AC-n`, in
  every file under `[guards] tests`). No goal row, no criterion row, a table of criteria missing
  its `status` or source column, or no claim at all is red: the format changed under the pattern.
  Then: **G-I1**, every story that is not `retired` names at least one goal and only goals of the
  charter, and every goal is named by one such story; no goal id leads two rows; where the
  charter has frontmatter, its `charter_status` is `confirmed` or `unconfirmed` and its
  `status`, if any, is `draft`, `stable` or `deprecated` (an `unconfirmed` charter is not red,
  but tell the owner it awaits their review, as orient does). **G-I2**, every
  story that is not `retired` has an `actor` of `owner`, `human:<id>` or `agent (reconstructed
  from ...)`, a `date` of YYYY-MM-DD, a `story_status` of live, unspecified or unconfirmed, and
  `unconfirmed` goes with an agent's actor only; every story with an agent's actor that is `live`
  or `unspecified` has a `confirm` line in the op log (§ context-engine record confirm), and no
  story such a line confirmed is `unconfirmed` again; every `retired` story has a `retire` line in
  the op log (§ context-engine record retire), and every story such a line retired is still
  `retired` with `status: deprecated`; only a `retired` story has `status: deprecated`; every id
  in a story's `superseded_by` or `supersedes` is a story, and only a retired story has
  `superseded_by`. Not read: a retired story's frontmatter after its retirement, its amendments, and
  whether `supersedes` mirrors `superseded_by`; the op log is trusted as written. **G-T1**, every
  `live` story's id is in the `from` cell of a criterion that is not `retired`, and every
  `unspecified` story's id is a whole word in an open board item. **G-T2**, every `tested` or
  `manual` criterion is claimed, every claimed id is a criterion, and every status is planned, tested, manual, deviates or retired. **G-T6**, every
  `from` cell names at least one id, each a story or decision that exists (a criterion's actor is
  its source's). *Why: the chain from goal to test is what makes correctness checkable; a link that
  points at nothing looks traced and is not, and a criterion marked tested whose test was deleted
  keeps reading as tested.*
- **G-I3**, delegated rulings (only when `[working] decision` is set). Every decision that is not
  `status: deprecated` and has a `ruled_by` key: `ruled_by` is not blank and not `owner`; `actor`
  is not `owner` or `human:<id>` alone (a ruling made on the owner's delegation is `owner asked`;
  `owner` and `human:` in any case, so `Owner` counts);
  and the text above `## Amendments` has the owner's delegating words, then a `## Ruling` heading
  with the ruling under it, neither part empty. A decision without `ruled_by` has no `## Ruling`
  heading. Not read: a delegated ruling recorded with neither (only the transcript can tell),
  whether the words above `## Ruling` are the owner's, whether `ruled_by` names who ruled, and a
  text written with a `## Amendments` line of its own (it reads as a real amendment; § context-engine
  record new and § context-engine ask answer step 2 stop it at the write, whatever `[guards] off`
  names). *Why:
  a ruling stamped `actor: owner`, or quoted as the owner's words, becomes frozen owner intent that
  nobody can amend away (FRAMEWORK §6.5).*
- **G-I4**, duplicate titles (only when `[working] decision` is set). No two decisions that are
  not `status: deprecated` have the same `title`, compared with case ignored and each run of
  spaces read as one space; REJECTED and DEFERRED decisions count. Taking the decisions in id
  order (by number, so 9 before 10), each one after the first of a shared title fails, naming the
  earlier ones. Fix it with `decision supersede <earlier> <later>` when the later replaces the
  earlier, or else retitle one to say how it differs (§ context-engine decision set). Not read:
  titles that differ in a word or in punctuation, two decisions that conflict in meaning under
  different titles, and a superseded decision. *Why: two current records with one title are one
  decision recorded twice (a replacement recorded with the old title and never superseded in) or
  two that conflict; either way `decision find` shows both as standing.*
- **G-D12** (only when `[guards]` sets `coverage_spec` or `coverage_maps`; red if only one is
  set, if the spec is missing, or if the `coverage_maps` glob is absolute or has a `..`, matches
  no file, or matches a file with no `YYYY-MM-DD` in its name): outside fenced code in the spec,
  list (1) the bold component name in each numbered row of §2's table, (2) each `### 5.6` and
  `### 6.n` heading as `§6.n`, (3) every `G-` id (`G-D3`, not `XG-D3` or `G-D3x`), and (4)
  each backticked `noun.verb` name followed by `(` or the closing backtick inside §8.0.2 (from its
  `###` heading to the next heading of that level or higher), not one whose verb is a file
  extension (`gate.md`), nor the operations it writes bare (`list`); fewer than 20, 10, 30 or 30
  of a class is red (the engine's floors; a repo whose spec they do not fit sets `[guards] off =
  ["G-D12"]`). An operation the engine registers (its name with the dot as an underscore is an
  MCP tool) owns itself; check only the rest. Take the newest map by the date and session letters in its
  file name, ordered as session keys (`z` before `aa`). Each listed thing must be a whole cell
  in its class's column of a table in that map: under `Component` for a component (the cell
  may add ` (...)`), under `§` in a table with a `Procedure` column, or a `Ritual` column (the
  old name, kept by frozen maps), for a procedure, under `Guard` for a G-id, under `Operation`
  for an operation. Or it appears as a whole word in some board file, open or closed (`§6.1` is
  not `§6.13`, `G-D1` is not `XG-D1`; an operation as `hk.ratchet` or `hk_ratchet`).
  A component's former name, which the engine still accepts, counts as its name in both places:
  `Invariants` for the Residual-risk register, `Law / NFRs` for Non-functional requirements, so
  a map frozen before the rename still owns it. *Why: a spec that gains a procedure or guard
  nothing owns drifts unseen; the map is frozen, so the fix is a new dated map or a `task new`
  naming it, never an edit.*
- **G-D13** (only when `[guards] reader_docs` lists paths; a listed path that is missing is red):
  in each file there (each `.md` under a listed directory), outside fenced code and frontmatter,
  first blank every link destination (`](...)`), every reference definition line (`[x]: ...`) and
  every URL (`scheme://...`, `<scheme://...>`); then no id of a series the repo keeps stands
  alone (not glued to a letter, digit or `-` on either side): a kind's `prefix` in
  `context.toml`, the prefix leading the criteria rows (`[working] criteria`), the register's
  entry prefix (`[working] register`), and `GOAL-n` and `G-n` while `[guards] charter` is set.
  Inline code is read too. A guard id (`G-D1`), a section (`§8.0`) and a session key are not
  ids. *Why: the README is the page a stranger reads first; an id is a pointer they cannot
  follow, and what is built and what is next belongs on the board, so give the reader the
  words, or point at the board or the records site.*
- **G-D7**: in each file `[guards] signals` lists, outside fenced code: at most `warn_max` ⚠, at
  most `bold_caps_max` bold spans with a run of four capitals and no lower case (`**NEVER**`),
  and none of `retired_glyphs`. *Why: a signal stands out only while it is rare.*
- **G-D8**: the orientation note (`HANDOFF.md`; `handoff` in `[working]`) has at most one
  `## Banner (<key>)` heading, no other heading containing a session key, and no banner line that
  starts with a session key as a bullet or in bold. *Why: a note meant to be replaced, not added
  to, grows back as sessions stack updates on it; this is the stacking check.*
- **G-D9**: every line of `context.toml` that assigns a number (inline tables too) has a comment
  with `provenance: MEASURED` or `provenance: JUDGEMENT` and `wrong if` (or `wrong as above`),
  and the comment does not cite an example. *Why: a tuned value with no provenance cannot be
  re-judged, and a spec example is neither a measurement nor a judgement.*
- **G-W7**: for each board item and each of `moved`, `updated`, `closed`, `last_asked`: the value
  equals the `session` of the newest op log line for that id whose `fields` lists the key; with
  no such line, the value is older than the log's first session. Every op log line parses as
  JSON, and a `lane`, where a line has one, is a lane id (a letter or digit, then up to 63
  letters, digits, `.`, `_` or `-`). A line carrying `as` (housekeeping's `last_asked` stamp, or a `task import` keeping a
  legacy board's history) is checked against the key `as` gives for that field instead. *Why: those keys are written only by operations, so a
  mismatch is a hand edit.*
- **G-W9**: for each op log line with `"op": "close"` and `"mechanical": true`: the item's
  `## Log` has a line `- <that session> (<actor>): closed: done in <sha> (its `Closes: <id>`
  trailer, checked in git)` where `git log -1 --format='%(trailers:key=Closes,valueonly)' <sha>`
  names the item (a commit git does not know fails in a full clone and is trusted in a shallow one)
  and `git merge-base --is-ancestor <sha> HEAD` succeeds, or `... closed: closed in
  session <key> (its ledger entry's rows name it closed)` where that finished ledger entry's
  `rows:` says `<id> (closed` and the `## Log` above that line has no `closed:` line from
  `<key>` and no `reopened:` line from a later session. *Why: a mechanical close is
  housekeeping's, which acts only on recorded evidence; one without it hides a judgement nobody
  recorded.*

A guard id listed in `[guards] off` is not reported (only ids lint reports may be listed), and
the write-time stops that check the same rule are off with it (among them G-I1 and G-I2 in
§ context-engine record new for a story, G-I3 there and in § context-engine ask answer for a
decision, G-W4 in § context-engine ask new, G-D4 for `stale_after` in § context-engine record
new, task set and decision set). Two stops are no guard's twin and stay on: the
`## Amendments` stop of record new and ask answer, and record new's stop on a decision given a
`status` key, `decision_status: SUPERSEDED`, `supersedes` or `superseded_by`. A path listed in `[guards]` that does not exist is
itself a failure.

**Why:** lint is the check on the stored files. It catches what a write-time refusal cannot: hand edits, and
records that were valid when written but stale now. A non-empty result is a failure (exit 1).

### context-engine plan check

Before a builder starts on a plan (FRAMEWORK §6.6, the build ladder), check every citation in
the plan file, given from the repo root or as an absolute path (refused, exit 2, when it names
no file). Outside fenced code and frontmatter, read each backticked span as § lint's **G-D3**
reads one: a `path::Name.attr` names a file that exists case-exact (a bare file name may be
anywhere in the repo) whose text has each dotted part as a word; any other span shaped as a
file name or a path with a `/` (no spaces, globs or `<placeholders>`, not starting with `.` or
`/`) names a file or directory that exists case-exact, from the repo root or, for a plan inside
the repo, from the plan's own directory. A bare name not at the repo root that matches files
whose texts differ fails too. A span followed by ` (new)`, with at most one space between, is
one the plan creates: skip it, and every other citation of that path, of a path under it, of
its bare name, or of that `path::Name` and the names under it (`Name.attr`), wherever it stands
in the plan. That holds only while the path does not exist: a marked path that exists (a bare
name: any file of that name) is not new, so its mark skips nothing and fails as `marked (new) but
exists`; a plan that adds to a file marks each new `path::Name`, not the file. Write each failing
span as `line <n>: <problem>`. Any failure, or a plan with no
citation at all (none checked and none marked new), fails the check (exit 1), and the plan goes
back to its planner.

**Why:** a builder handed a plan that cites a file or symbol that is not there guesses, or builds
on a misreading; a mechanical check before the build catches it at no model cost, by the rules
G-D3 already holds the docs to. A plan that cites nothing would pass a check that read nothing.

## Reserved ids

### context-engine id reserve

For a batch of records that cite each other (linked tasks, a decision and the tasks it names), or
a draft that cites a record not yet written: hold their ids first, cite those, then write each
record with `--id`. Only a board kind or an append-only kind (the ones `task new`, `ask new` and
`record new` create).

1. How many: one id or more, up to twenty at a time.
2. The first id: one above the highest number of the series in a file name, an `id:` key, a `new`
   or `import` line of the op log, or an `id.reserve` line. The rest follow it.
3. Append one `id.reserve` line per id to the op log (§ What every write does): `path` is `""`,
   `fields` is `[]`. It is not a milestone: the next commit carries the lines.
4. Write each record as its operation says, with the reserved id (`task new --id`, `ask new --id`,
   `record new --id`), in this session. An id another session reserved is never taken: the
   `id.reserve` line's `session` says whose it is. An id reserved and never written stays a gap.
5. Before committing, every reserved id the batch cites must be written: lint G-D3 fails a body
   that names an id no record has.

**Why:** a batch whose bodies name each other's ids, written before the ids existed, cited the
wrong records, and the bodies are append-only (adhoc AAR entry 31); an agent that cannot learn an
id wrote a `DEC-<NEW>` placeholder instead (entry 32, lint G-D2). Held in the op log, a reserved id
is one no other write hands out, so the citation is right before the record exists.

## The board (tasks and owner questions)

### context-engine task new

1. Pick the kind: task (`<prefix>-n`, `PRJ-n` say) or question (`ASK-n`; prefer `context-engine ask new`).
2. The brief is at most 120 characters. Every named dep must exist. The tier's cap must have room
   (NOW 1, NEXT 5, counting open items of both kinds). Otherwise, demote something first.
3. Id: the highest number in the series + 1. Count file names *and* `id:` keys, open and closed.
   Never reuse a number, and skip every id an `id.reserve` line of the op log holds. An id this
   session reserved is written with `--id` (§ context-engine id reserve). An item imported from a
   repo's own board keeps its number when it has one (`--id ASK-8` for an imported `Q8`), if no
   file and no `new` or `import` line of the op log ever had that id and no other session
   reserved it; it is written as the engine writes ids (no leading zeros).
4. File: `tasks/<id>-<slug>.md`. Slug: lowercase the title; replace each run of characters other
   than ASCII `a-z` and `0-9` (accented letters included) with `-`; strip `-` from both ends. If
   it is over 48 characters, take the first 48 and drop everything from the last `-` on, even a
   whole word. An empty slug gives `tasks/<id>.md`. The slug never changes afterwards.
5. Frontmatter, in order: `type`, `id`, `title`, `brief` (if any), `pri` (default LATER),
   `state: open`, `deps: [...]`, `goals: [GOAL-n, ...]` (for a task where `[guards] task_goals` names
   the kind: at least one goal of the charter, spelled as the charter spells it), `source: session <key>`,
   `created: <today>`, `moved: <key>`, `updated: <key>`. Body, unless you write your own:

   ```
   <title>.

   ## Traps

   - none yet

   ## Read first

   - none yet

   ## Log
   ```

**Why:** one file per item, with no renames, so every reference keeps working. The caps keep the
board a ranking, not a pile (G-W5). The brief is a constraint, not a summary (FRAMEWORK §5.2).

### context-engine task import

Bring a legacy board in with its history (FRAMEWORK §11.4): `task import <file> [--kind <board
kind>]`. Nothing is written if any check fails.
1. The file is JSON Lines, UTF-8 (a byte-order mark is read past): each non-blank line is one
   object, `{"id": "<id>", "fields": {...}, "body": "<text>"}`, `body` optional, no other key.
   Name every line that is not one at once, and stop.
2. The kind is a board kind. Each id is new, as § context-engine task new step 3 says for a kept
   number, and no two lines share one. Write each value as given, with nothing defaulted (no
   `moved`, `updated`, `source` or body the line lacks); a string under a key the kind types
   `date` or `int` becomes one. A value under a key the kind types `key` must be a session key
   (a date and letters): a plain date (`moved: 2026-07-29`) stops the import, naming its line and
   id; keep such a value in a key of the repo's own (`legacy_moved`), never converted.
3. Write each file as § context-engine task new steps 4 and 5 say (its path from its title), with
   `type`, `id`, then the fields as given, and `status: deprecated` on one whose `state` is
   `closed` (a `status` the `state` contradicts stops the import). Each record must pass the
   kind's schema.
4. One op line per record (§ What every write does): `"op": "import"`, `"fields"`: every
   frontmatter key, and `"as"`: each of `moved`, `updated`, `closed` and `last_asked` the record
   holds, with its value (no `as` when it holds none). No tier cap, deps or goals check: run lint
   on the result. Not a milestone: commit it by path.

**Why:** G-W7 checks every `moved`, `updated`, `closed` and `last_asked` against the op log, so a
legacy value no operation wrote is red; the import line's `as` records the values it kept, and
G-W7 checks the file against them, so a later hand edit is still red. A plain date is refused
because a session-key field holds only a key, and converting one would invent history.

### context-engine task set

Change any key except the ones operations own. `state`, `status` and `closed` change by
`close`/`reopen`. `moved` changes by `touch`/`promote`. `pri` by `promote`. `deps` by `dep`.
`last_asked` by `ask raised`/`ask later`. `answered` by `ask answer` only, and `agent_decided` by `ask decide` only. `updated` changes on
every write. `type`, `id` and `body_sha` never change. The brief stays at most 120 characters. Values take the kind's declared
type: an empty value, `null` or `~` means null (so `brief=` clears the brief); a list is
comma-separated, with or without `[...]`; a date is `YYYY-MM-DD`; a key must look like
`2026-10-08e`. A set with no fields still bumps `updated`. `review` takes only values from
`[working] review_triggers` (`record-kind`, `guard`, `write-op`, `git-loss`, `long-lived`,
`shipped`): set it on an item above the design-review threshold (FRAMEWORK §5.2). Setting it on
an item already at NOW first checks its `## Design review` as in `task promote`. Do not remove it
from an item whose design review lists a `- residual:` line: close the item to file them, or take
the residual lines out first. `aliases` lists the item's earlier ids when it was imported from a
repo's own board (`aliases=[1h, est-activity]`), under G-D2's alias rule (§ lint). A new list
replaces the old one: name every alias that is dropped in the reply, since it no longer resolves.
Only a write that sets `aliases` checks them; a clash made elsewhere shows in lint G-D2.
`goals` on a task where `[guards] task_goals` names the kind: at least one goal of the charter,
each spelled as the charter spells it; never unset it on an open task (lint G-I1).
Stop, writing nothing, if you were given `stale_after` and it is empty, not a timestamp with
its offset (`2026-12-01T00:00:00Z`, OKF §5.5; a bare date is not one), or already passed by the
clock (one equal to now is taken), whatever the item's state: G-D4 of § context-engine lint
would be red on it (off with it). Unsetting it is always taken.

**Why:** the keys with an operation each carry a rule: caps, consistency of `state` and `status`,
`moved` meaning real progress. A raw edit would skip that rule.

### context-engine task promote

Item open; `pri` a valid tier; the target tier has room, not counting the item itself. Set `pri`,
`moved: <key>`, `updated: <key>`. **Moving the NOW item down** can leave NOW short: check NOW as
in § task close and tell the owner.

**HOLD** (the last tier, where the task kind's `pri` enum has it) means held: parked until
something outside the board releases it. A held item is never ready: G-W5's NOW rule and the
empty-NOW warnings do not count it, a re-rank never reads or moves it (`procedures.md` § Rerank),
and orient flags a NOW or NEXT item whose dep is held. Move an item into HOLD with a
`task comment` naming what releases it, and out of it (`task promote <id> <tier>`) when that
happens or the owner says so; neither move is restricted, and HOLD has no cap. Where the enum
lacks HOLD, a promote to it is refused like any unknown tier, and the remedy says how to enable
it: add "HOLD" to `pri` in `enums` under `[kinds.<kind>]`, after SOMEDAY, as the kit's
`context.toml.template` has it. A question is never held.

**Promoting to NOW an item that carries `review:`** (it is above the threshold, FRAMEWORK §5.2):
first check its `## Design review` section by hand, and do not promote while any check fails.
Start a review from this template (the engine knows it verbatim, so a copy left unchanged is
refused):

```
affects: <who can be hurt: the building session, the next session, the owner, a person using plain git, every deployed repo>
- eliminated: <a risk the design cannot produce>
- reduced: <risk> | rung: <reduce|collective> | control: <the refusal, guard or rule, by name>
- residual: <risk> | rung: <reduce|collective|individual> | why: <why it was not eliminated> | control: <the guard or rule, by name>
F: <what will need doing to it later, by whom, how often, and what makes it stale>
A: <how it is read and fixed without the engine>
M: <which forms keep it from drifting>
E: <what it reuses: one name, one format, one runbook section>
```

The checks: no template line left verbatim; an `affects:` line with text; at least one risk
line; every line starting `- ` is `- <class>: <risk>` with class `eliminated`, `reduced` or
`residual`, then ` | key: value` fields from `rung`, `why`, `control`, `affects`; a `reduced`
or `residual` line has a `rung` from `eliminate`, `reduce`, `collective`, `individual` (a
residual is never at `eliminate`; a risk controlled only at `individual` is a `residual`, P17); a `residual` line has `why` and `control`; and `F:`, `A:`,
`M:` and `E:` lines, each with text. Any other unindented line is an error.

**Why:** re-ranking is progress on the board, so it bumps `moved`, and it is where the caps are
enforced. Promotion to NOW is the last point before building starts, so it is where a design
above the threshold must show whom it can hurt and how each risk is controlled (P17, P18).

### The last rerank

The charter's `Last rerank:` line (in its § Next milestone) holds the session key of the last
full re-rank (`procedures.md` § Rerank), or `never`. Step 5 of the re-rank replaces the value with
the session's key; it is the one edit to the charter that needs no owner. The line is a record of
when the re-rank ran, never a trigger. A re-rank is due (what orient says) only when NEXT holds no
task while tasks not at HOLD are open: what NOW holds does not matter, an owner question does not
count as a task, and neither does a held task. Where `[guards] charter` is set and the line has a value, orient quotes it after the
reason (`(last rerank: <value>)`). The owner can ask for a re-rank at any time.

**Why:** the full re-rank reads the whole board, so it runs once a sprint: when the five NEXT
slots are worked through, not each time NOW empties, and not each time owner direction lands.

### context-engine task dep

Add (every one must exist and must not be the item itself) or remove ids in `deps`. Keep the
existing order and add new ones at the end.

**Why:** deps decide what an answered question or a closed task unblocks (`ask answer` reports
it), so a dangling or self dep would mislead.

### context-engine task comment

Append to `## Log` (create the heading at the end if missing) one line
`- <key> (<actor>): <text>`, with further lines indented two spaces (a blank line becomes two
spaces). The section is written as heading, blank line, entries. Never edit an earlier log line.

**Why:** the log is the item's history in place. Being only ever appended to keeps it trustworthy.

### context-engine task touch

Set `moved: <key>` and `updated: <key>`. With `--mechanical`, set only `updated`.

**Why:** `moved` means a person or agent made progress on the item. Housekeeping touches are
mechanical, so they must not look like progress (FRAMEWORK §5.2).

### context-engine task close

A message is required, and the item must be open. **Never close an owner question (`ASK-n`)
this way:** it closes only on the owner's answer, through `ask answer` (FRAMEWORK §5.3); if the
owner answered "later", use `ask later`. Set `state: closed`, `status: deprecated`,
`closed: <key>`, `moved`, `updated`. Append `- <key> (<actor>): closed: <message>` to `## Log`.
Do not move or rename the file.

**If the item carries `review:`**, first check its `## Design review` as in `task promote`; if a
check fails, fix the review, do not close. Then copy each `- residual:` line into the register
(`[working] register`; the kit's default is `docs/invariants.md`). Skip a residual already there
with the same risk text and the same task id in its `from:` line. Create the file with the
heading `# Residual-risk register` if it is missing. Each entry takes the register's next id:
the prefix its entries use (all one prefix: with two, stop and rename them to one by hand, each
keeping its number), or with no entries the prefix of the last id a `register.append` line of the
op log names, or with neither `RISK`; and the number one above the highest in the file and in any
`register.append` line, whatever the prefix, so a number is never reused,
after one blank line:

```
## INV-<n>: <the risk text>

- **affects:** <the line's `affects` field, else the review's `affects:` line>
- **why not eliminated:** <its `why`>
- **control:** <its `rung`>: <its `control`>
- **from:** <task id>, session <key>
```

Log one op line per close that added entries: `op: register.append`, `id` the task, `path` the
register, `fields` the new register ids. With residuals and no register configured, stop: configure
one first.

**Ask, before every close, whether the work left a residual risk** (a risk that was not
reasonably practicable to eliminate, FRAMEWORK §7.4), reviewed item or not, and state the
answer (`--residual`, repeatable): `none`, or one line per risk in the form `<risk> | rung:
<reduce|collective|individual> | why: <why not eliminated> | control: <the guard or rule> |
affects: <who>`, each on one line. Check each line as a design review's `- residual:` line is
checked (a rung other than `eliminate`, a `why`, a `control`, no other field) plus an `affects`;
refuse `none` given with a risk, and refuse `none` for an item whose design review lists a
`- residual:` line (its risks are filed at this close, so the answer is not "none": close without
`--residual`, or state only the risks the review does not list). File each risk as above, with the
review's, its own `affects` in its entry, creating the register on first use. Append the answer
to the close's `## Log` line as a second line, indented two spaces: `residual risk: <the register ids
filed>`, the review's included, then `; already in the register: <ids>` for each of this task's
risks (the review's and those stated) that an earlier close filed, the part before it left out
when nothing new was filed; or `residual risk: none`. A close that states nothing and has no
risk of this task's to name writes no such line, so the record tells a question skipped from a "none"; where a register is configured and the item
carries no `review:`, the engine's reply says the risk was not stated. A risk missed at close goes
into the register by hand, as above.

**If the item is labelled `hazard`** (its `labels` hold `hazard`, in any case: a suspected,
unconfirmed hazard, FRAMEWORK §7.4), refuse the close unless it states a resolution: a risk line
(the hazard is confirmed: file it as above), `none` (it was fixed or dismissed: the message names
the fix's commit or why it is dismissed), or a design review that lists a `- residual:` line. A
housekeeping close (§ hk close) is refused the same way: housekeeping never resolves a hazard.

**If the item carries a `## Kill bar`** (a bar written before a run, saying which measure and
which threshold fires it), the close must give a verdict against it, `--verdict "<outcome>:
<evidence>"`, its outcome one of `KILL` (the bar fired), `PASS` (it did not) or `VOID` (the run
cannot judge the bar, e.g. an invalid measurement), then a colon and the evidence (a results
file and its section). Without a bar, or with an empty one, refuse a verdict: write the bar first
(§ task section), before the run, so its op line shows it came first. A heading that differs
from `## Kill bar` only in case or spacing (`## Kill Bar`) is not read as no bar: stop, and move
the bar under `## Kill bar`. Add the verdict to the
close's `## Log` line as a second line, indented two spaces, `verdict against the kill bar:
<verdict>` (before any `residual risk:` line), and to the close commit's body as a paragraph
`Verdict: <verdict> (against <id>'s ## Kill bar)`, before the `--body` text.

**If an ACTIVE-PLAN doc names the item** (a doc G-D4 reads, `[guards] liveness` or `history`,
whose liveness is ACTIVE-PLAN and whose first `liveness_lines` lines name it) and every other
board item named there is closed, stop: freeze that doc first (§ doc freeze), then close. The
engine refuses the close until then, and names each such doc; with `[guards] off` naming G-D4 it
does not check.

**If the item was at NOW**, count the open items at NOW once it is closed. If NOW holds fewer
than its cap (`[working.caps]`, 1 by default) while any item not at HOLD stays open, G-W5 is red
from now on: tell the owner, naming the NEXT items (or saying NEXT is empty), and ask which one to
promote (§ task promote) before you run the gate. Do not pick it yourself. The engine closes
anyway and says the same at the end of its reply to the close. It says nothing when NOW is still at or
over its cap (G-W5's own remedy applies then) or when `[guards] off` names G-W5.

**Last, commit: a close is a milestone** (§ What every write does, **Commit the write**): the
item, the register if it changed, the op log and what was still uncommitted, at once. The message: the
title `<id>: close; <message>` (its first line, cut at a word with `...` to fit 72 characters);
then `Ops: <op> <id>, ...` naming the op lines it carries, when there is more than its own; then
the whole message, if the title cut it or it runs over a line; then the `Verdict:` paragraph, if
the close gave one; then the body given with `--body`
or `--body-file` (the `Sabotage:` lines [sabotage.md](sabotage.md) step 5 asks for, and any
receipts); then the `Also carries ...` paragraph when another session's lines go in; then one
trailer paragraph, `Closes: <id>`, `Session: <key>` and `Co-Authored-By: <name> <email>` (§ What every write does, **Commit the write**, says which; none for a person). Close a task once its work is committed,
so these lines follow the work they describe. If a close is
refused after its write was made, that write is committed without `Closes:`.

**Why:** closed items stay in place as records, so links never break. `status:
deprecated` is the OKF view of the same fact (G-W4). The message says why, e.g. `done in <commit>`.
A question closed by `task close` would carry no decision in the owner's words, and the agent
would in effect have answered its own question (FRAMEWORK §5.3).
The residuals move to the register at close because that is when the design is final, and the
register is what the next owner of the work reads (P17, FRAMEWORK §7.4; G-W13 checks it).
Every close states its residual risk, `none` included, so the record tells a risk judged absent
from a question never asked (an adopter's six closes filed none, and nobody could say which).
A verdict is linked to the bar written before the run, so a bar on record cannot be quietly
skipped at the close. A bar rewritten or emptied after the run is not refused (an open item's bar
is corrected before its run, and the engine cannot tell before from after); its `section` op line
shows when, and the reviewer reads it.
A plan that names the item is frozen before the close, not after: G-D4 turns red on an
ACTIVE-PLAN doc once every item it names is closed (FRAMEWORK §6.7: it freezes when its item
closes), and a freeze is the session's own step with no owner call in it, so the close refuses
until it is done, and the freeze goes into the close's commit.
An empty NOW is reported at close, not refused: refusing would block a close the owner wants,
and the next NOW item is the owner's call; if it were reported only at the next gate, it would
turn the gate red before anything could be committed.
The close is a commit of its own because finishing a task is the step a reader of the history
looks for: `Closes:` names the task in the commit that finished it (FRAMEWORK §6.10), and
[sabotage.md](sabotage.md) puts the `Sabotage:` lines in a commit message, which is the close's.

### context-engine task reopen

A message is required, and the item must be closed. **Never reopen an owner question that
carries `answered:`**, and never delete its `answered:`: its answer is a decision (the one
`answered: {decision: DEC-n}` names). To change the answer, `record amend DEC-n` or
`decision supersede DEC-n <new>`, where DEC-n is the decision that stands now (follow its
`superseded_by`); for a follow-up, `ask new` citing ASK-n and DEC-n. A question
closed without `answered:` (by a `task close` run before `close` began refusing questions) may be
reopened: that is its repair. **Never reopen an owner question that carries `agent_decided:`**
either (and no `answered:`): an agent closed it with a PROVISIONAL decision, and a reopened
question beside that live decision is a limbo. Overturn it with § context-engine ask answer, or
confirm the decision with § context-engine decision accept. Set `state: open`, set `closed:` to null, delete `status`, and set `moved` and `updated`.
Append `- <key> (<actor>): reopened: <message>`.

**Why:** the mirror of close, keeping G-W4 true. Reopening an answered question would leave it
open while claiming an answer; clearing `answered:` instead would orphan the decision whose
`answers:` names it, and a second `ask answer` would make two decisions answer one question
(FRAMEWORK §5.3). A question an agent decided has no owner answer to protect, but the decision
that closed it is live, so only the owner's answer or acceptance moves it on.

### context-engine task section

Replace the content under `## <name>` up to the next `## ` heading. Add the section above `## Log`
if it is missing (at the end if there is no Log), so the Log stays last. Never `## Log`: use
`comment`. On an open NOW item carrying `review:`, check the
result's `## Design review` as in `task promote`, and do not write it while a check fails. On a
closed item, never `## Kill bar`: its close judged it (comment the correction, or reopen the
item and run again). On a question, never `## Grant`: it holds owner words that only § ask
answer's one-off permission writes (comment instead).

**Why:** the `## Traps` and `## Read first` sections are replaced as they get better. The Log is history, so it is only
ever appended to.

### context-engine task list

From frontmatter only: open items by default (`--state closed|all`), optionally filtered by
`--pri` or `--label`. Sort by tier (NOW, NEXT, LATER, SOMEDAY, HOLD), then prefix alphabetically (ASK
before T), then number. One line each: pri and id each padded to 7 characters, the title, then
`  (closed <key>)` and `  [deps: a, b]` where they apply. No match prints `no matching items`.

**Why:** the board view comes from the records, never the other way round (FRAMEWORK's board size 2: one file per item). Reading
frontmatter only keeps it cheap.

### context-engine task show

Print `id · pri · state · title`, then the aliases, the brief and deps, then the body with
`## Log` cut to its 5 newest entries, newest first (`--head 0` for all), or just one `--section`.
An id that is no configured series' (`1h`, `Q9`) is looked up as an alias: find the one record
whose `aliases` lists it, and first print `<alias> is an alias of <id>`. `decision show`, `why`,
`decision find --links-to` and the ids `decision link` and `decision supersede` take resolve an
alias the same way (a criterion id, a path or a `file::symbol` is taken as given). Other writes
take only the record's own id.

**Why:** read-only and bounded. The Log grows without limit, and the newest entries are the ones
that matter. An old id in a commit message or a docstring still finds its record, and the reply
names the record, so nobody mistakes which one answered.

## Owner questions

### context-engine ask new

As `task new` with kind question, default `pri: NEXT`, plus `last_asked:` (null) and
`blocks: [...]` if it blocks items. The body states three parts, and the engine refuses one
that lacks any (with `--body` or `--body-file`; with neither it refuses too): **context**, under
`Context` or `Background`; **two or more options**, one list item (`- A. ...`, `1. ...`, `A)
...`) or `###` heading each, under `Options`, or, where it has neither, one line led by a bold
label (`**A:** ...`) each; and a **recommendation with its reasoning**, at least five words,
or `none` saying which exemption applies (too costly to reach, or it needs human judgement) and
why. A part is a heading (`##
Options`), a bold label naming one of the parts (`**Options:**`) or, in a body that writes none
of its parts as a heading or a bold label, a plain one (`Options:`); any other such line is text
of the part it is in, and only the heading `## Log` ends the body (a `Log:` line is prose). A
line indented to the text of the list item above it is that item's text (a sub-bullet, or a bold
`**Why:**` note), everything under a `###` option is its text, and a line indented past a bold
option is its text, so none of them is another option; beside a list, a bold intro or note
(`**Pick one**`, `**Note:** ...`) is no option either. A `<...>` placeholder counts as nothing.
The template:

```
**Question:** <title>

**Context:** <what led to the question>

**Why owner-only:** <the "who decides" test>

**Options:**

- A. <option>
- B. <option>

**Recommendation:** <the option, and why in a line; or "none (too costly): <why>" or "none (needs human judgement): <why>" (FRAMEWORK §5.3)>

**Blocks:** <ids or none>

## Log
```

Then raise it in chat this session. Ids in `blocks` are not checked to exist (deps are), so
check them yourself. Lint (G-W4) re-checks the three parts on every open question, so one filed
with `task new --kind question`, or edited since, fails until its body states them; a closed
question is history and is not checked.

**Already answered in chat** (`--answered <words>` or `--answered-file <words.md>`, with
`--decision-title <title>`): a question put to the owner through a chat question tool, with no
ask record behind it, that the owner has answered. Check first what § context-engine ask answer
step 1 and 2 check (the words, and for a delegated ruling `--ruling` with `--ruled-by`), and stop
before writing anything if one fails. Then file the question as above, with this default body:

```
**Question:** <title>

Asked in chat and answered there, so it was never open on the board; recorded with `ask new --answered`.

**Blocks:** <ids or none>

## Log
```

and answer it at once, as § context-engine ask answer does, steps 2 to 4, in the same commit (with
`--grant` and no `--decision-title`, its one-off permission steps instead, with this body's last
sentence naming `ask new --answered --grant`). Its
tier cap is not checked: it is closed before the commit, so it never holds a place there. Its
body need not state the three parts: it was asked and answered in chat, and the decision holds
the answer.

**Why:** a question waiting on the owner is a board item, so it is ranked and gets raised, never
buried in prose (FRAMEWORK §5.3). "Why owner-only" makes the asker check that it really is the
owner's call (P8). The context, the options and a reasoned recommendation let the owner answer
in one line without reconstructing the problem; the recommendation is skipped only where it
would cost more than the question is worth, or where no amount of reading can settle it. A question whose recommendation is cheap to undo is not: record it instead as
your own decision, with your own actor and `decision_status: PROVISIONAL` (§ context-engine
record new), and flag it in the close digest. An answer given in chat to a question with no record
is still the owner's words: `--answered` keeps it on the rails `ask answer` uses (a stamped
decision with `actor: owner` and `answers:`, the question closed with `answered:`), where a
decision built by hand needs five keys and a body written right.

### context-engine ask raised

After raising questions in chat: each must be a question, and your session must have an open
stub. Set `last_asked: <key>` (and `updated`) on each. In your stub, delete any `- **asked:**`
line and add one at the end, with no blank line before it:
`- **asked:** <every id raised so far this session>`. The engine does not check that a question
is still open.

**Why:** marking them raised resets the overdue clock. The stub line is the receipt that `session close`
checks (G-W8).

### context-engine ask later

The owner said "later". Set `last_asked: <key>` and append
`- <key> (<actor>): owner said later; still open`. Do not close it. This writes no `asked:` line
in your stub, so a deferred question at NOW or NEXT must still be named in your close's `asked`,
or demoted to LATER.

**Why:** "later" is not an answer. The question stays open, and the clock restarts (FRAMEWORK
§5.3).

### context-engine ask answer

1. The question must be open and carry no `answered:`, and you need the owner's words verbatim
   (the engine takes them as an argument, or written to a file and passed with `--file`). Open with `answered:` means someone reopened it by hand (G-W4): restore `state: closed` and
   `status: deprecated` from git and change the answer through its decision, as in `task reopen`.
2. New decision `DEC-n`: `title`, `actor: owner`, `decided: <today>`, `session: <key>`,
   `decision_status: ACCEPTED` (`PROVISIONAL` if the decision kind's enum lacks `ACCEPTED`), `answers: ASK-n`. Body:
   `Owner, answering ASK-n: *"<words>"*`. Stamp its `body_sha` at once.
   **If the owner delegated the ruling** ("you decide", and an agent or another human made the
   call; the engine takes the ruling with `--ruling` or `--ruling-file` and who made it with
   `--ruled-by`, both or neither): the words are only the owner's delegating words, `actor: owner
   asked`, `decision_status: PROVISIONAL` (the ruler's call, reviewed later through § context-engine
   decision accept), add `ruled_by: <who ruled>` (never `owner`), and the body is
   `Owner, answering ASK-n by delegating the ruling to <who ruled>: *"<words>"*`, a blank line,
   `## Ruling`, a blank line, `Ruled by <who ruled> on the owner's delegation; only the quote above
   is the owner's words.`, a blank line, then the ruling. Stop if the words hold a `## Ruling`
   heading of their own (G-I3, unless `[guards] off` names it), or if the words or the ruling
   hold a `## Amendments` line (the text below it would fall outside `body_sha` and read as a
   later amendment; this stop holds whatever `[guards] off` names), and stop as § context-engine
   record new step 2 does if the body cites an id no record has.
3. On the question, set `state: closed`, `status: deprecated`, `closed: <key>`,
   `last_asked: <key>`, `answered: {on: <today>, session: <key>, decision: DEC-n}` (with
   `ruled_by: <who ruled>` added for a delegated ruling) and `updated`.
   Append `- <key> (<actor>): answered by the owner: DEC-n` (for a delegated ruling,
   `- <key> (<actor>): answered on the owner's delegation, ruled by <who ruled>: DEC-n`).
4. Report what it unblocks: open items whose deps include it and whose deps are now all closed,
   plus everything in its `blocks`.
5. If the question was at NOW, check NOW as in § task close and tell the owner.

Without `--grant`, stop before step 2 if the title is empty: the decision needs one.

**A one-off permission** (`--grant`, no `--title`): the owner said yes to one thing (a push, a
release) and made no choice that later work must keep, so no decision is written. Step 1 as
above, then, in place of steps 2 and 3:

1. Stop, before writing anything, if a title, a ruling or `ruled_by` is given (a ruling is a call:
   drop `--grant`), if the words are empty, if a line of them starts `## ` (it would end the
   section), or if `[working] ledger` is unset or your session has no `kind: open` stub.
2. On the question, add a `## Grant` section above `## Log` holding exactly
   `Owner, granting ASK-n: *"<words>"*` (the words with CRLF read as LF and outer whitespace
   stripped). Its hash is `sha256:` and the hex sha256 of that text plus one `\n`, in UTF-8. Set
   `state: closed`, `status: deprecated`, `closed: <key>`, `last_asked: <key>`,
   `answered: {on: <today>, session: <key>, grant: <the hash>}` (no `decision` key) and `updated`.
   Append `- <key> (<actor>): granted by the owner, a one-off permission: no decision`.
   Op log: `ask.answer`.
3. In your stub, delete any `- **granted:**` line and add one at the end, as § ask raised does for
   `asked:`: `- **granted:** <every id granted so far this session>`. Op log: `ask.answer`, id
   your key, the ledger's path, fields `["granted"]`.
4. Steps 4 and 5 above. At your close, quote each grant's words in the summary
   (`ASK-n granted: "<words>"`, § session close step 3).

Never edit the `## Grant` words afterwards, nor write the section with `task section`: lint
(G-W4) fails a question whose section is missing or whose text no longer hashes to its `grant`.

**Overturning an agent decision.** A question closed by § context-engine ask decide (closed, with
`agent_decided: {decision: DEC-a}` and no `answered:`) takes the owner's answer, which overturns
the agent's call. The inputs are the owner's words and a title, as above; stop before writing
anything if `--grant` is given (a grant mints no decision, so it cannot overturn one), if the
title is empty, or if `DEC-a` is not a decision. Then, in place of step 1's open check: write the
owner's decision `DEC-n` as steps 2 does (`answers: ASK-n`, or the delegated form); on the
question set `answered: {on: <today>, session: <key>, decision: DEC-n}` (with `ruled_by` added
for a delegated ruling) and `last_asked: <key>`, leave `state`, `status` and `closed` as they are,
set `updated`, and append `- <key> (<actor>): overturned by the owner: DEC-n supersedes DEC-a`
(delegated: `overturned on the owner's delegation, ruled by <who ruled>: DEC-n supersedes DEC-a`);
then § context-engine decision supersede `DEC-a` `DEC-n`. The question stays closed and unblocks
nothing new. A closed question carrying both `answered:` and `agent_decided:` takes no second
answer. To confirm the agent's call instead, § context-engine decision accept `DEC-a`.

**Why:** an answer is a decision in the owner's own words. It is stamped so the words cannot
drift, and closing the question points to the decision rather than burying the answer in a log
line (FRAMEWORK §6.2). A ruling made on the owner's delegation is not the owner's words: inside the
owner's quote, or under `actor: owner`, it would be frozen as owner intent (FRAMEWORK §6.5). A
one-off permission makes no choice for later work to keep, so a decision would only pad the
log; its words still stay where they cannot drift, hashed on the question, and reach
the owner digest through the summary quote.

### context-engine ask decide

Closes an open owner question with the agent's own decision, when the question follows from what
the owner already said (FRAMEWORK §5.3). You need the reasoning (`why`), a title, the ids it is
derived from (`sources`), one second opinion from the arbiter (FRAMEWORK §6.6) verbatim
(`opinion`), and who gave it (`opinion_by`, a model id). Any session with an open stub may do it.

1. The question must be an owner question (`ASK-n`). Your actor must be an agent's
   (`<harness>/<model-id>`, not `human:` or `process:`): the decision's actor is `agent
   (<model-id>)`, the text after the first `/`, and never `owner` or `owner asked`.
2. Stop if the title or the why is empty, if the opinion or `opinion_by` is empty, or if
   `opinion_by` is `owner` or `human:<id>` (the owner's words are never a second opinion).
3. Stop if a line of the why or of the opinion starts `## ` (it would split the decision's
   sections; this also keeps a `## Amendments` line, whose text would fall outside `body_sha`, out).
4. Stop if there are no sources, if one names no record (resolve an alias as elsewhere) and no
   charter goal, if the question itself is listed, or if none is owner-authored: a story with
   `story_status: live`, a decision whose `actor` is `owner` or `human:<id>` (any case, no
   `ruled_by`) that is not SUPERSEDED or REJECTED, or a charter goal id.
5. Stop if the question is not `state: open` (already closed), carries `answered:` (open and
   answered is a by-hand reopen, G-W4: restore it from git), or carries `agent_decided:` (the same).
6. Stop if `[working] ledger` is unset or your session has no `kind: open` stub, if the body of
   step 7 cites an id no record has (as § context-engine record new step 2), or if closing the
   question would leave an ACTIVE-PLAN doc that names it as its last open item unfrozen (as
   § context-engine ask answer does).
   Nothing is written until every stop has passed.
7. New decision `DEC-n`: `title`, `actor: agent (<model-id>)`, `decided: <today>`, `session:
   <key>`, `decision_status: PROVISIONAL`, `refs: [ASK-n, <each source>]` (no id twice), and **no**
   `answers`. Body, with CRLF read as LF and the why, the opinion trimmed of outer whitespace
   (the opinion otherwise as given):

   ```
   Agent decision closing ASK-n, PROVISIONAL until the owner confirms it (`decision accept` on this decision) or overturns it (`ask answer ASK-n`).

   <why>

   Derived from: <the sources, comma-space joined>.

   ## Second opinion

   <opinion_by>, <today>, verbatim:

   <opinion>
   ```

   Stamp its `body_sha` at once.
8. On the question set `state: closed`, `status: deprecated`, `closed: <key>`, `last_asked:
   <key>`, `agent_decided: {on: <today>, session: <key>, decision: DEC-n}` and `updated`. Append
   `- <key> (<actor>): decided by the agent, PROVISIONAL until the owner confirms or overturns it: DEC-n`.
9. In your stub, delete any `- **decided:**` line and add one at the end, as § ask raised does for
   `asked:`: `- **decided:** <every question id decided so far this session>`, comma-space
   joined. Op log: `ask.decide`, id your key, the ledger's path, fields `["decided"]`.
10. Steps 4 and 5 of § ask answer: report what it unblocks (a decided question counts as closed
    exactly as an answered one), and check NOW. At your close, name each question with its
    decision in the summary (`ASK-n decided by the agent as DEC-n (overturn: ask answer ASK-n)`,
    § session close step 3).

Not checked: whether the question really follows from its sources, and whether the opinion agrees
with the decision. Both are judgement, and the opinion sits verbatim in the body for the owner's
review. The owner confirms the decision with § context-engine decision accept, or overturns it
with § context-engine ask answer; § context-engine task reopen refuses a decided question. Lint
(G-W4) fails a question carrying `agent_decided:` that is open, that names no decision, whose
decision is the owner's, or whose `refs` do not name the question.

**Why:** a question the owner's own records already settle should not cost the owner's attention,
but a decision an agent made must never pass for the owner's words (FRAMEWORK §6.5). So it is a
PROVISIONAL decision under the agent's own actor, grounded in an owner-authored record, checked by
a second opinion kept verbatim where the owner can read it, and named in the digest, so the owner
can overturn it. Closing the question points to that decision, as an answer does, so nothing it
blocks waits.

## Batons and pauses

### context-engine baton add

New `BATON-n` in `docs/working/`: `title` (first line of the step, cut to 100 characters, never
refused), `session: <key>`, `state: open`, `created: <today>`, `why` (`''` if none given). Body:
`<step>` + blank line + `**Why skipped:** <why or "not given">`.

**Why:** a step skipped at close must reach the next session as a record, not as a sentence in a
note that the next banner overwrites (FRAMEWORK §5.4).

### context-engine baton list

Read-only. List every open baton and pause record (`docs/working/`, `state: open`), in orient's
order (your own session's first, then pauses before batons), each pause with its resume step and
branch and each baton with its `why`; `no open batons or pauses` when there are none.

**Why:** orient's must-act part, in full: when that part is cut at the cap, this is where the rest
is, without opening every record.

### context-engine baton done

Evidence is required: the commit, file or ledger entry that shows the step done. The baton must be
open. Set `state: done`, `done: <key>`, `evidence: <text>`.

**Why:** without evidence, "done" cannot be checked.

### context-engine baton expire

Over MCP this is the tool `mcp__context-engine__hk_expire_batons`, not `baton_expire`.

Housekeeping: actor `process:housekeep` unless the caller gives one. For every open baton or
pause from a session with 2 or more closes/pauses after it:
1. new task at LATER, titled `Expired <id>: <title>` (cut to 120 characters), `source: <id>`,
   body: the task body template with its first line replaced by
   `Expired <id> from session <s>: <title>. Converted by housekeeping (G-W3); read <path>.`
   No cap check: LATER has none.
2. on the baton or pause, set `state: expired` and `expired_to: <new task id>`.

Both writes are `mechanical: true` in the op log.

**Why:** work carried over from earlier sessions must not drift for ever, and turning it into a task keeps it without
blocking orientation (G-W3).

### context-engine pause open

1. Read the branch (`git rev-parse --abbrev-ref HEAD`) and every uncommitted path, untracked files
   included (`git status --porcelain=v1 --untracked-files=all`; for a rename take the new path),
   from git itself. Leave out the op log and every path its uncommitted lines name: step 3
   commits them.
2. New `PAUSE-n`: `title` (first line of what was in flight, cut to 100 characters), `session`, `state: open`, `created`,
   `branch`, `uncommitted: [...]`, `resume_step`, `evidence` (or `none`), `deferred: [...]`. Body:
   `**In flight:** ...` + blank line + `**Resume step:** ...`.
3. Commit that file with the op log, at once: `git add -- <path>`, then
   `git commit --only -m "pause PAUSE-n (<key>): <first 60 characters of the title>" --trailer "Session: <key>" --trailer "Co-Authored-By: <name> <email>" -- <path> docs/ledger/ops.jsonl`,
   adding any path the op log's other uncommitted lines name (§ What every write does,
   **Commit the write**). (`--only` on a file git has never seen fails without the `add`. The
   `--trailer`s are the one trailer paragraph every agent commit ends with, FRAMEWORK §6.10, and
   `<name> <email>` is as **Commit the write** says.) The
   engine commits a pause whatever `[engine] commit` says, `"off"` included, and when it cannot
   commit one, it refuses but keeps the record; commit it by hand.

**Why:** the next session needs the exact state of the tree, which comes from git, not from
memory. Committing at once keeps another session's close from taking it into its own commit with
the other uncommitted op lines (FRAMEWORK §5.5 rule 4).

### context-engine pause resume

The pause must be open. Set `state: done`, `resumed: <key>`.

**Why:** closes the loop so `orient` stops listing it, and records who picked it up.

## Housekeeping (FRAMEWORK §6.11)

One of housekeeping's operations lives with its records: `baton expire` (MCP
`hk_expire_batons`). The others are here. Each
write is mechanical (`"mechanical": true`), with actor `process:housekeep` unless the caller
gives one, and none forms new intent: each carries out what a commit or the ledger already says.

### context-engine hk close

`context-engine hk close <id> --commit <sha>`, `hk close <id> --ledger <key>`, or `hk close
--scan` (MCP `hk_close(id, commit, ledger, scan)`). Closes an open item only on recorded
evidence. By hand:

1. **A commit:** it must be in HEAD's history (`git merge-base --is-ancestor <sha> HEAD`), and
   git must read `Closes: <id>` in its trailers: `git log -1
   --format='%(trailers:key=Closes,valueonly)' <sha>` names it (a `Closes:` line outside the
   message's last paragraph is not a trailer). Refuse it if the item was already closed in that
   commit's own tree (`git show <sha>:<the item's path>` says `state: closed`: it is the item's
   own close commit, and the item was reopened since), or if the item's `## Log` already cites
   the hash (it was closed on it and reopened since, or a comment named it: either way it is not
   new). The message is `done in <short sha> (its
   `Closes: <id>` trailer, checked in git)`.
2. **A ledger entry:** the session `<key>`'s entry must be finished (not `kind: open`) and its
   `rows:` must say `<id> (closed`. Refuse it if the item's `## Log` has a `closed:` line from
   `<key>` (it was closed then and reopened since) or a `reopened:` line from a later session: a
   reopen outweighs the old rows. The message is `closed in session <key> (its ledger entry's
   rows name it closed)`.
3. Then do § context-engine task close with that message, logging the `close` line with
   `"mechanical": true`. Never an owner question: it closes only on the owner's answer.
4. **`--scan`:** `git log -F --grep=Closes: --format='%H %(trailers:key=Closes,valueonly,separator=%x2C)' HEAD`;
   for each open task (not a question) a trailer names, try its commits newest first by step 1,
   and close on the first that holds. List each one no commit could close, with why.

**Why:** no tool closes on its own judgement: calling a close "mechanical" would only hide that
someone decided it. A commit made outside the engine that finished an item carries `Closes:
<id>` (FRAMEWORK §6.10 item 1), so the engine checks git, not the caller's word, and G-W9
re-checks it in the stored files. Anything else housekeeping would close is proposed to the
session, which closes it with `task close` as its own judgement.

### context-engine hk stamp

`context-engine hk stamp` (MCP `hk_stamp_asked`). By hand: for every finished ledger entry
(not `kind: open`), read the ids its `asked` receipt (or `- **asked:**` line) names. For each
open question among them, take the newest such entry's key; if the question's `last_asked` is
empty or older than that key, set `last_asked: <that key>` and `updated: <your key>`, and log
`hk.stamp_asked`, mechanical, with `"as": {"last_asked": "<that key>"}`.

**Why:** a question raised in chat restarts its overdue clock (FRAMEWORK §5.3), but an `asked`
receipt written by hand, or given to `session close --asked` without `ask raised`, stamps
nothing, so the question would come back as overdue though the owner saw it.

### context-engine hk run

`context-engine hk run start` after `session start`, and `hk run close` before `session close`,
which runs it itself, first, unless given `--no-housekeeping` (MCP `hk_run(phase, check,
anyway)`; `--check` lists the plan and writes nothing; `--anyway` runs a plan bigger than
`[housekeeping] max_writes`, on the owner's word, step 5). By hand, do the rows of
procedures.md § Housekeeping in order:

1. **start:** § context-engine session abandon with `--stale`, § context-engine hk stamp, then §
   context-engine hk comment with `--scan`.
2. **close:** § context-engine baton expire, then § context-engine hk close with `--scan`.
3. Each write as its own section says: the actor is yours, or `process:housekeep`; every line is
   `"mechanical": true` but an abandon's. When the run abandoned or closed anything (milestones),
   commit all its writes in one commit by path at the end; otherwise they go with your next commit.
4. List every operation you did, one line each, then the rows `hk run` does not do, which you do
   next: at start, foreign commits and pause records whose paths are committed; at close, `record
   stamp`, ratchets and other generated files, and `task touch --mechanical`.
5. Count them (G-W10): the operations, and the op log lines the run added (`wc -l` on the op log
   before and after). Where `[housekeeping] max_writes` is set and the plan has more operations,
   stop before the first write and ask the owner; go on only on the owner's word.

**Why:** housekeeping that is a row in a table is forgotten; one call per procedure point does every
tier-0 row in order, and its reply is the operation list the digest needs. The SessionStart hook
holds no key, so it only shows what is owed (orient's `Housekeeping owed` part); the session runs
this, and the gate only checks through lint. The plan and the run are one code path, so `--check`
lists what the run then does.

### context-engine hk comment

`context-engine hk comment <id> --commit <sha> [--text "<words>"]`, or `hk comment --scan [--since
<rev>]` (MCP `hk_comment(id, commit, text, scan, since)`). A commit becomes a dated comment on the
open task its title leads with. By hand:

1. **The range** (`--scan`): your session's range of § Reconciling foreign commits (procedures.md),
   `<baseline>..<your start head>`, or `<rev>..HEAD` with `--since`. List it oldest first:
   `git log --reverse --format='%H %s [%(trailers:key=Session,valueonly)] [%(trailers:key=Closes,valueonly,separator=%x2C)]' <baseline>..<end>`.
2. For each commit whose title leads with ids (`PRJ-3: ...`, `PRJ-3, PRJ-4: ...`: the comma-separated
   ids before the first colon) naming an open task (never a question): skip it if its message has no
   `Session:` line (it is foreign: reconcile it, and say so), if its `Closes:` trailers name the item
   (leave it for § context-engine hk close), or if the item's `## Log` already cites the hash (a
   hash in the log that the commit's full hash starts with). A commit with a `Session:` line whose
   title names an open task anywhere else (`tidy the docs for PRJ-4`, or `PRJ-3: ... PRJ-4` for
   PRJ-4) is a `tier 1` line for that task, unless its `Closes:` names it or the log already cites
   the hash: write nothing for it, list it, and leave it to a judgement (yours, or the repo's
   tier-1 housekeeping agent's, FRAMEWORK §6.11). Commit bodies are not read for ids.
3. Otherwise add `- <your key> (<actor>): <short sha>: <subject>` to the item's `## Log` (§
   context-engine task comment) and log `comment` with `"mechanical": true`. One item by hand is
   the same with that commit, which must be in HEAD's history (`git merge-base --is-ancestor <sha>
   HEAD`); with words of your own (`--text`) the title need not lead with the id, and the line is
   `<short sha>: <words>`, unless the words already cite a hash the commit's full hash starts with.
4. List what you commented and what you skipped, with why.

**Why:** the item's log should show its history without reading git, and a title that leads with
the id (the commit rule) parses without a model, so it is tier 0; any other mention of an item is
a judgement (tier 1). A commit that closes the item is `hk close`'s evidence, and a log citing it
would make `hk close` refuse it as not new, so the comment leaves it alone. The range partitions
HEAD's history, so no commit is commented twice by two sessions, and the cited-hash check makes a
second run write nothing.

## The residual-risk register

### context-engine register show

Read the register file named by `[working] register` (the kit's default is `docs/invariants.md`).
Print `register <path>: <N> residual risks`, N counting every entry, retired ones too, and then
`, <M> retired` when M entries are retired. Then, for each `## <id>: <heading>` entry (`RISK-n`,
or a kept register's own prefix), print a blank line, `<id>: <heading>` with the heading as written
(a retired entry's `(retired <session>: <why>)` included), and its `affects`, `why not
eliminated`, `control`, `from`, `merged` and `retired` lines, those it has, each indented two
spaces. A register file that does not exist yet is empty: print `register <path>: 0
residual risks` and say the first `task close` that files a residual creates it (a missing
`[working] register` is still an error). With `--engine`, read instead the `REGISTER.md` inside the installed
`context_engine` package (`<interpreter> -c "import context_engine, os;
print(os.path.dirname(context_engine.__file__))"` prints the directory) and say `engine
<version>` for the path.

**Why:** the register is handed over with the work (P17: regulations 7 and 11 of Singapore's Design for Safety
Regulations). The engine's own
register ships inside the package, so a repo the engine is pointed at can read what the engine
cannot protect it from, without a checkout of the engine (FRAMEWORK §7.4). A retired entry prints
its marker and its `retired:` line, so it does not read as live.

### context-engine register retire

`register retire <id> --why "<why>"`, or `--into <id>` to merge it into another entry, retires
an entry in place. It is never deleted.

1. Refuse, writing nothing, when:
   - `[working] register` is not set;
   - `--why` holds a line break or a round bracket;
   - `--why` is empty and there is no `--into`;
   - the register has no entry `<id>`, or no file yet;
   - the entry is already retired: its heading says `(retired ...)`, or it has a `retired:` line.
     A retirement is final.
   - With `--into <target>`: the target is `<id>` itself, is not in the register, or is retired.
2. The reason is `<why>`. With `--into <target>` it is `merged into <target>`, then `, <why>`
   when a why is given.
3. Rewrite the entry's heading `## <id>: <risk>` as `## <id>: (retired <session>: <reason>) <risk>`.
4. After the entry's last `- **<field>:**` line, add `- **retired:** session <session> (<actor>):
   <reason>; kept so its id is never reused`.
5. With `--into`, go to the target's entry. If it has a `- **merged:**` line, extend that line with
   `; <id> folded into this entry, session <session>`. If it has none, add `- **merged:** <id> folded
   into this entry, session <session>` after its last field line.
6. Change no other byte. A CRLF file is written back LF.
7. Log one op line: `op: register.retire`, `id` the entry, `path` the register, `fields:
   ["retired"]`. With `--into`, log a second line: `id` the target, `fields: ["merged"]`. This is
   not a milestone: the next commit carries it.

**Why:** an id is never reused (FRAMEWORK P3, §7.4). So an entry that is no longer live, a
duplicate or one folded into another, stays in the file, and it must not read as live:
- the heading says so to a reader of plain git;
- the `retired:` line says who retired it, when and why;
- `register show` prints both.

The marker is parsed off the heading, so the entry's risk is unchanged, and G-W13 and a re-close
still match a task's residual to it. G-W13 fails an entry whose heading and `retired:` line
disagree, so a retirement half done by hand shows. The reason is one line without brackets
because the marker is found by its brackets.

## Docs: freeze and archive

Both work on a hand-written `.md` doc, never on a record: a record's liveness is its state, and a
closed item keeps its file (FRAMEWORK §6.7).

### context-engine doc freeze

1. Find the doc's liveness declaration as G-D4 reads it (FRAMEWORK §8.3): the topmost line in its
   first `liveness_lines` lines (10 by default) that opens a block (the file's first line, or one
   after a blank line, a heading or the frontmatter) and is led, after a blockquote's `>` and
   emphasis, by LIVING, ACTIVE-PLAN or FROZEN, or by `ARCHIVED <date>`; or a frontmatter
   `liveness:` key. Stop if it is already FROZEN: nothing to do. Stop if `[guards]
   liveness_given` names the doc: its opening lines are not this repo's to edit. If it declares
   nothing, "that line" below is the line after its title, when the first non-blank line after
   any frontmatter is a `# ` title (past the blank line that follows the title, when a line
   follows that blank); else the line first after the frontmatter; else its first line. Stop if
   the FROZEN line would then fall outside the first `liveness_lines` lines, where G-D4 does not
   read it (the engine refuses): put it within them instead, or raise `[guards] liveness_lines`.
   If the declaration is a frontmatter key, set it to `FROZEN` and say when and why in the body.
2. Right above that line, insert `FROZEN <today, YYYY-MM-DD> (session <key>): <why>` and one
   blank line, in the doc's own line endings; wrap the FROZEN line at 100 characters. With no
   `--note`, the why of an ACTIVE-PLAN doc that names board items is `<their ids> closed; below,
   the plan as it stood`. Change nothing else.
3. Log one op line (§ What every write does). Not a milestone: the next commit carries it, the
   close's commit when it is done just before `task close`.

**Why:** an ACTIVE-PLAN doc freezes when its item closes (FRAMEWORK §6.7, §7.2), and a frozen doc
is provenance that nobody keeps current. The FROZEN line goes above the old declaration, not over
it: the topmost declaration is the one G-D4 counts, and the old line stays as part of the record
of what the doc was. `task close` refuses to close the last open item a plan names until the plan
is frozen (§ task close), so G-D4 is never red between a close and its freeze.

### context-engine doc archive

For a file git tracks, with no uncommitted change: a whole doc or other file (the steps below),
or a passage of a doc (§ Pruning part of a doc, after them). Neither the redirect table nor,
for a passage, the archive file may have an uncommitted change no op line names (an earlier
archive's own writes, still uncommitted, are fine): a hand edit would ride into the commit.

1. The archive directory is `--to`, else `[working] archive` (the kit's default is
   `docs/archive`). Stop if the file is already in it, if a file of its name is already there
   (choose another directory, or a dated name), if it is named `README.md`, which is the
   redirect table's name, or if it is the engine's own `context.toml` or op log (`[engine]
   oplog`): moved, every later command goes on without it.
2. Move it: `git mv <file> <dir>/<name>`. Never repair it: no fixed links, no updated text. (The
   engine moves the file without staging it, and the next commit records the move; git shows a
   rename either way.)
3. Make a markdown doc's first line
   `ARCHIVED <today, YYYY-MM-DD> (session <key>), from <old path>.`, then one blank line, then
   the doc as it was, in its own line endings. A doc with frontmatter keeps it on top: the
   ARCHIVED line goes first after its closing `---`, where G-D4 reads it too, and a `liveness:`
   key in it, which G-D4 reads first, becomes
   `liveness: FROZEN  # ARCHIVED <today>; was <its old value>`. If the frontmatter has no
   `liveness:` key and the ARCHIVED line would fall below G-D4's window (`[guards]
   liveness_lines`, 10 by default), put `liveness: FROZEN  # ARCHIVED <today>` first in the
   frontmatter; if its `liveness:` key lies outside the window, stop: move the key to the top
   by hand and commit that first. Material that is not markdown moves byte for byte, with no
   ARCHIVED line: its row dates it.
4. Add its row to `<dir>/README.md`, the directory's one redirect table, after the last
   row of the table headed `| Was | Now | Archived | Why |`. Its cells: the old path; a markdown
   link whose text and target are both the archived file's name; `<today> (session <key>)`; the
   why, or `—`. With no README, create it: `# Archive: <dir>`, a
   blank line, a paragraph led by `LIVING:` saying that it is this directory's redirect table and
   that nothing else here is repaired, extended or kept current, a blank line, then the table. A
   README without that table gets a `## Redirects` section at its end holding it.
5. Log three op lines (§ What every write does). Not a milestone: the next commit carries the old
   path's deletion, the archived copy and the table.
6. If `context.toml` names the old path (a route, a `[guards]` list), point the entry at the new
   path or drop it, or lint is red (G-R1, G-D4); the engine's reply names such an entry. A link
   to the old path in a doc lint reads (G-D3) is red too: in a living doc, point it at the
   archived copy by hand. The engine rewrites none of them; its reply names each pointer G-D3
   newly reports (by hand: compare `context-engine lint`'s G-D3 lines before and after the
   move; a pointer already red that only changed line, or moved into the archived copy, is
   not new). The archived copy's own links are never repaired: if G-D3 reads the archive
   directory, take it out of `[guards] links` and `paths`.

**Pruning part of a doc** (an orientation note over its cap): which lines go is a judgement;
the cut is mechanical, `context-engine doc archive <doc> --lines A-B [--to <dir>] [--why]`. The
doc is markdown, tracked, committed (the lines named are the lines in git) and not FROZEN (a
frozen doc is never edited).

1. Cut lines A to B, numbered from 1 as an editor numbers them. Not the whole doc (archive it
   whole), not into its frontmatter, and not so that G-D4 reads another liveness, or a
   declaration where the doc declared none (whatever `[guards] liveness_given` gives it):
   keep the declaration, and the blank line, heading or frontmatter that opens its block, out
   of the range, and keep the line above a word in passing. A blank passage is nothing to prune.
2. Append the passage unchanged to `<dir>/<stem>-pruned.md` (for `HANDOFF.md`,
   `HANDOFF-pruned.md`): a blank line, `## Pruned from <doc> on <today>`, a blank line, the
   passage. The headings are the file's index. A new archive file opens with
   `ARCHIVED <today> (session <key>), passages pruned from <doc>, each under its dated heading.`
3. Add a row to `<dir>/README.md` as in step 4 above, one per prune: its first cell
   `<doc> lines A-B`, its link to the archive file.
4. Log three op lines (§ What every write does): the doc, the archive file, the table. Then step
   6 above: a link elsewhere to a heading that left the doc is red under G-D3.

**Why:** an archive answers "where did X go?" (FRAMEWORK §7.3). `git mv` keeps the history with
the material (`git log --follow`). Never repairing it keeps it evidence of what was true when it
was archived. The `ARCHIVED <date>` first line is a FROZEN declaration to G-D4, so an archived
LIVING doc no longer claims to be kept current; in frontmatter, G-D4 reads the `liveness:` key
before it, so that key says FROZEN too. One redirect table per directory is where anyone holding
the old path finds the new one. Only committed material, so the move carries nobody's half-made
edit. The engine names pointers rather than rewriting them: a pointer can sit in a record whose
body is hashed or append-only, or in a frozen doc, which no write may change.

## The banner

### context-engine banner show

Print the banner's key, its hash (above) and its text. Keep the hash: `banner set` needs it.
If the orientation note (`HANDOFF.md`) has no `## Banner (<key>)` heading yet (or there is no
orientation note), the hash is `none`.

**Why:** reading and hashing together is what makes the later write safe.

### context-engine banner set

1. Re-read `HANDOFF.md` and recompute the banner hash. If it differs from the one you saw, stop.
   Another session wrote it: read theirs, merge, and start again.
2. Replace the heading and text up to the next `## ` heading with `## Banner (<your key>)`, a
   blank line, and your text; add a blank line after it only if something follows. Keep the
   orientation note's line endings.
3. **The first banner** (the hash you saw was `none`): insert the same block just before the
   orientation note's first `## ` heading, so that heading still ends the banner; with no `## `
   heading, append it after a blank line. Create the orientation note if it does not exist.
4. Before writing, run lint's own G-D1 (the caps of every route that covers the file), G-D5
   (when `[guards] numbers` lists it and it is neither FROZEN nor ACTIVE-PLAN) and G-D8 (the
   orientation note) on the orientation note as it would be. If the new text adds a failure the stored note does not
   already have (match them by guard and message, numbers aside), stop: write nothing, and fix the
   text as the failure's remedy says. A failure already there is lint's to report, not this
   write's.
5. Say how many lines the old banner had and the new one has. When the old one had 10 or more and
   the new one has fewer than half as many, warn: the lost lines may hold durable facts, which
   belong in a tracked doc. When orient would cut the new banner at its cap, say so.

**Why:** the banner carries exactly one session key, and a later close must not silently erase
another session's banner (FRAMEWORK §5.5 rule 2), nor the durable facts in a long one. The
check before writing is the same one lint runs, so a banner is never stored only for lint to
reject it, a round of fixes later.

## Criteria

Criteria are rows of the `[working] criteria` file (FRAMEWORK §3.3), not records: lint reads them
in place (G-T1, G-T2, G-T6). Both operations write one row and leave every other byte of the file
as it was. Never reflow, re-pad or re-sort the table: a hand-edited table stays as its editor left
it. Neither is a milestone: the next commit carries the file and the op line.

### context-engine criterion new

1. Choose the table: a table of the criteria file whose header's first column is `id` and which has
   a `status` column and a source column (`from`, `story` or `source`). With several, the one under
   the heading you name; the engine refuses to guess. With none there, write the heading, the
   table's header row and its `|---|` line yourself first: the operation writes rows only.
2. Choose the id: the next number of the rows' series (`AC-n`; `AC` when there are no rows), above
   every row of the file and every id a `criterion.new` op line created, so an id is never handed
   out twice (P3), even after a row was deleted. Sessions drafting at once in separate worktrees do
   not see each other's rows: take ids from a block handed out for the purpose and pass `--id`.
   A given id keeps the rows' prefix (any prefix while there are no rows): a second prefix leaves
   the next id unclear, so the engine refuses one. G-D2 and the duplicate-row check (G-T1, G-T2,
   G-T6) red a clash once the work is merged.
3. Check what the row will say, as the guards will read it. Its source names at least one story or
   decision that exists (G-T6). Its status is `planned`, `tested`, `manual`, `deviates` or `retired`.
   `tested` or `manual` only once a test or driver claims the id under `[guards] tests` (the
   `[guards] claim` pattern; G-T2), and that test has been seen red with the behaviour broken (P7).
   Each cell is one line, with no `|` unless escaped as `\|`, and no trailing backslash.
4. Add the row directly after the table's last row, in that row's line ending, as
   `| <id> | <criterion> | <from> | <status> |` in the table's own column order, every other cell
   empty; a `Claimed by` column gets the claiming file when the row is `tested` or `manual`.
5. Log one op line (§ What every write does).

**Why:** intake step 3 drafts the criteria (FRAMEWORK §6.2); a draft written by hand into a table
others edit by hand is where a reflowed row, a reused id or an unread row creeps in. One row only,
and each check the same one lint makes at rest, so a write never stores what the gate then fails.
Rows only, because where a new section goes and what it is called are choices for the writer.

### context-engine criterion set

1. Find the one row the id leads, in a criteria table (step 1 above). An id leading two rows is
   resolved first: the later row takes the next free id.
2. Rewrite only the cells you change: status, criterion or from. Each changed cell becomes
   ` <value> `, but the last cell of a row with no closing pipe becomes ` <value>` (the row's
   trailing spaces stay after it); every other cell keeps its bytes (spacing, emphasis, an escaped `\|`, a missing
   closing pipe), and the row keeps its line ending. A status is compared as lint reads it
   (`**Planned**` is already `planned`), so nothing is rewritten just to drop formatting.
3. Make the checks of § criterion new step 3 on the values that change. Moving a row to `tested`
   or `manual` fills a `Claimed by` column with the claiming file.
4. Log one op line naming the columns changed.

**Why:** a criteria census (FRAMEWORK §6.2 step 6) rewrites and retires criteria, and a criterion
becomes `tested` once its claiming test was seen red (P7). Writing those cells and nothing else keeps
the diff to the change. `tested` is refused without a claim because lint would red it at once.
Not checked by either operation, nor by lint: that a `retired` row names the decision that
retired it, a `deviates` row its deviation entry, and that a story marked `sign-off: required`
keeps its criteria `planned` (FRAMEWORK §3.3).

## Append-only records (decisions and stories)

### context-engine record new

1. Take the next free id in the kind's series (as `context-engine task new`), or `--id`, one this
   session reserved: `DEC-n` in `docs/decisions`, `STORY-n` in `docs/stories`, file `<id>-<slug>.md`.
2. Frontmatter: `type`, `id`, `title`, then the keys you were given. Any **required** key you
   were not given is filled only if its only possible value is the present: a date-typed key (`decided`,
   `date`) with today, a key-typed one (`session`) with your session key. Everything else
   required, `actor` above all, must be given; if it is missing, stop. If you were given
   `stale_after`, stop unless it is a timestamp with its offset that has not yet passed (G-D4 of
   § context-engine lint; off with it). For a story (the kind
   `[guards] stories` names), stop too unless it passes G-I2 and G-I1 of § context-engine lint:
   `actor` is `owner`, `human:<id>` or `agent (reconstructed from ...)`, `story_status` is from the
   vocabulary but not `retired`, and `unconfirmed` when the actor is an agent's (§ context-engine
   record confirm moves it on), there is no `status: deprecated`, `supersedes` or
   `superseded_by` (§ context-engine record retire writes those), and `goals` names at least one
   goal of the charter, each spelled as the charter spells it (`GOAL-n`, or `G-n`). A record carried over from a repo's own log (an entry
   added after its import) gets `aliases: [<old id>]` here, under G-D2's alias rule; a decision's
   aliases are fixed from then on (`decision set` refuses them). For a decision, stop unless it
   passes G-I3 of § context-engine lint: a ruling made on the owner's delegation (the owner said
   "you decide", and an agent or another human made the call) has `ruled_by: <who ruled>` (never
   `owner`, in any case) and an `actor` that is not the owner alone (`owner asked`), and its body
   is the owner's delegating words verbatim, then a `## Ruling` heading with the ruling under it;
   any other decision has no `## Ruling` heading (both checks off while `[guards] off` names
   G-I3). Stop too, whatever `[guards] off` names, if a decision was given a `status` key,
   `decision_status: SUPERSEDED`, `supersedes` or `superseded_by`: only § context-engine
   decision supersede writes those, with both links (G-W4 and G-D3 check them at rest, but this
   stop is not behind their off switches). A
   decision's `decision_status`, when you were not given one, is the status it is born with:
   `ACCEPTED` if it is the owner's own (`actor` is `owner` or `human:<id>`, in any case, and it has
   no `ruled_by`), else `PROVISIONAL` (an agent's call, or a ruling on the owner's delegation); a
   decision kind whose enum lacks `ACCEPTED` (a `context.toml` from before it) gives the owner's
   own `PROVISIONAL` too. Stop if the owner's own is given `PROVISIONAL` (that means a call
   awaiting review, and the owner's word awaits nothing) while the enum has `ACCEPTED`, or if an
   agent's call is given `ACCEPTED` (only § context-engine decision accept marks a call reviewed,
   and logs it); `BUILT`, `DEFERRED` and `REJECTED` are taken as given. For every
   kind, whatever `[guards] off` names, stop if the text holds a `## Amendments` line: the text
   below it would fall outside `body_sha` and read as a later amendment (a later note goes in with
   § context-engine record amend, once the record exists). Stop too, unless `[guards] off` names
   G-D3, if the text, outside inline code, fenced code, blockquotes and `*HISTORY —*` lines,
   cites an id of a configured series that no record or alias has (other than the id being
   written, or one this session reserved: § context-engine id reserve), or a `<id> § <heading>`
   whose target has no such heading: lint G-D3 would fail the stamped text for good. An id of
   another repo goes in backticks, with the repo named in the words around it; in an owner's
   quote that is markup, not a change of their words (or quote the words as a blockquote).
3. Body: the text, verbatim (owner words keep their line breaks), one final newline.
4. Stamp it at once (`context-engine record stamp`): `body_sha` is the last frontmatter key.

**Why:** owner intent is recorded verbatim, now (§6.2), and a record stamped when it is created can
never be quietly rewritten. Who decided is never guessed: an `actor: owner` on the agent's own call is
the worst error this file can hold, and a call the owner delegated is still the agent's (`ruled_by`).

### context-engine record stamp

On each named decision or story with no `body_sha`, once its body is final: add
`body_sha: sha256:<hex>` (the body hash above) as the last frontmatter key. Never replace an
existing stamp. Check every named record first; if any is not append-only or is already
stamped, stamp none of them.

**Why:** stamping freezes the owner's words (G-D10). Stamping too early freezes a draft, and
every later fix then has to be an amendment.

### context-engine record amend

The only body change an append-only record takes (the engine takes the text as an argument, or
written to a file and passed with `--file`). If the record is stamped, its hash must match
first. If there is no `## Amendments` heading, add one at the end. Then append a blank line and
`- **<key> (<actor>):** <first line>`, with further lines indented two spaces (blank lines stay
blank). Stop first, as § context-engine record new step 2 does, if the entry, read as it will
stand, cites an id no record has: its first line follows the `- **<key> (<actor>):**` prefix, so
a `>` or a fence opened there hides nothing, and its other lines are indented, so a `## ` line
there is no heading.

**Why:** owner intent changes by addition, so the original words and every change stay readable
in order, and the hash above the heading still holds.

### context-engine record correct

A bracketed insertion into a stamped decision or story (any append-only record): a typo, a
missing word, a `[sic]`. It only inserts.
1. The record must be stamped and its hash must match (as `context-engine record amend`).
2. Take the original text: the body above `## Amendments`, in the body hash's canonical form
   (LF, outer newlines stripped, one final `\n`; § The two hashes). Find `after` in it: it must
   occur exactly once (count overlapping matches); if it is missing or occurs twice, stop and
   pick a longer anchor. The anchor and the text are each one line, and stop as § context-engine
   record new step 2 does if the body as it would read, `[text]` in place and step 4's
   `corrected:` entry (which repeats the anchor) included, cites an id no record has that it
   did not cite before.
3. Insert `[text]` right after it. Note the offset: the number of characters (not bytes) of the
   canonical text before the `[`.
4. Append an amendment as in § context-engine record amend, with the text
   `corrected: at <offset> (+<length of [text]>) inserted "[text]" after "<after>"`.
5. Leave `body_sha` alone: it pins the words as typed, and a stamp is never replaced.

Undoing the corrections newest first (remove `<length>` characters at `<offset>`) gives back
the original text exactly; lint G-D10 does that, fails a record whose `corrected:` lines do not
match, and hashes the result against `body_sha`, so a bracket whose `corrected:` line was
deleted fails too.

**Why:** a strict body hash would forbid fixing a typo, and a free edit would let anything
change. An insertion that can always be undone keeps the words as typed readable under the fix
(FRAMEWORK §6.5); a change of meaning is an amendment, never a correction.

### context-engine record retire

Retire a story (the kind `[guards] stories` names): the owner withdrew it, or it was recorded
again to replace a broken record (the wrong goal, actor or status when it was created: lint G-I1 and G-I2
name this step). `record retire <id> --by <who> --why "<words>" [--superseded-by <new id>]`.
1. Say whose call it is. `--by owner` (or `human:<id>`): the owner withdrew it, and `--why` is
   their words, verbatim (or written to a file and passed with `--file`). `--by agent`: allowed
   only with `--superseded-by`, which names the replacement story (a new record with the same
   words, made by § context-engine record new, with the story's body as it is and the right fields; redo a `corrected:` insertion with § context-engine
   record correct); `--why` says what was wrong. An agent never withdraws owner intent on its
   own: stop, and ask the owner.
2. Check before any change. The story's hash must match (as § context-engine record amend). A
   successor must be a story, not the same one, and not retired; for `--by agent`, its words as
   typed must be the story's: the body hash of both (§ The two hashes, corrections undone) is
   the same. A story that an earlier `retire` line of the op log retired is final, even if a hand
   edit has since undone it: stop; restore it from git, and to bring its intent back, record a
   new story. A story already `retired` with no such line was retired by hand: retire it here,
   so who and why are recorded. Stop as § context-engine record amend does if step 3's amendment
   cites an id no record has.
3. On the successor: add the story to `supersedes` (a list; a new key goes last). On the story:
   set `story_status: retired`, set `status: deprecated`, and with a successor add it to
   `superseded_by` (a list), each new key going last in that order. Then append an amendment
   as in § context-engine record amend, with the text `retired (by <who>): <why>`, or
   `retired (by <who>; superseded by <new id>): <why>`.
4. Log one `retire` line for the successor, then one for the story (table above), and commit
   them together. The successor goes first because its change is safe to repeat: if the story's
   write fails, the step can be run again.
5. With a successor, the criteria that came from the story still name it in their `from` cells,
   and lint G-T1 says a `live` successor has none: add the successor's id to those cells. A
   successor that is an agent's reconstruction is created `unconfirmed` (§ context-engine record
   new), so a confirmation of the story does not carry over: it waits for § context-engine record
   confirm, and G-T1 does not read it until then.

**Why:** a story is the owner's words, so retiring one is owner intent, as recording it is: who
decided and why are kept with the story, under its amendments, and the op log line is what lint
reads. G-I2 fails a story marked retired that no logged `retire` retired (a hand edit of
`story_status`), and a story that a logged `retire` retired but that is no longer `retired` with
`status: deprecated`: a retirement is never undone, so the history stays readable. An agent's
retirement needs the story's own
words on the successor, so it moves a record and never drops owner intent. `status: deprecated`
is OKF's word for "kept for history, no longer current" (FRAMEWORK §9.3); a retired story
serves no goal (G-I1) and needs no criterion (G-T1).

### context-engine record confirm

Confirm a story an agent reconstructed (`actor: agent (reconstructed from ...)`, `story_status:
unconfirmed`): the owner read it and said it is what they meant. `record confirm <id> --file
<words.md> [--by <who>] [--status unspecified]`.
1. Say whose words they are. `--by owner` (the default) or `human:<id>`; the words are theirs,
   verbatim, written to a file and passed with `--file` (or given as the argument). An agent
   never confirms a story on its own: stop, and ask the owner (FRAMEWORK §11.4 asks once, for
   all of them, in one batched question). A successor that replaces a confirmed reconstruction
   (§ context-engine record retire, step 5) is confirmed by asking the owner again, in the words
   of that new answer: their earlier confirmation is never quoted for it.
2. Check before any change. The record must be a story whose `actor` is an agent's
   reconstruction and which is not `retired` (to bring a retired story's intent back, record a
   new story); its hash must match (as § context-engine record amend). A story that an earlier
   `confirm` line of the op log confirmed is not confirmed again, even if a hand edit has since
   made it `unconfirmed`: stop, and restore it from git. A reconstruction already `live` or
   `unspecified` with no such line was confirmed by hand: confirm it here, so whose words
   confirmed it are recorded. Follow the story's `supersedes` links, and theirs in turn: if the
   words are those of a `confirmed (by <who>): <words>` amendment on any story found there
   (whitespace aside), stop, and ask the owner again; if they answer in the very same words,
   quote them with when and where they said them. Stop as § context-engine record amend does
   if step 3's amendment cites an id no record has.
3. Set `story_status: live`, or `unspecified` with `--status unspecified` when no criterion comes
   from the story yet (lint G-T1 fails a `live` story with none). Leave `actor` as it is. Then
   append an amendment as in § context-engine record amend, with the text `confirmed (by
   <who>): <words>`, or `confirmed (by <who>; unspecified): <words>`.
4. Log one `confirm` line (table above) and commit it with the story.

**Why:** a reconstruction is the agent's reading of the owner's intent, never their words; the
owner's confirmation is what makes it intent, so it is recorded as owner words are, under the
story's amendments, and the op log line is what lint reads. G-I2 fails a reconstruction that is
`live` or `unspecified` with no logged `confirm` (a hand edit of `story_status`, which lint cannot
otherwise tell from any other, since the op log records keys, not values), and a story that a
logged `confirm` confirmed but that is `unconfirmed` again. The `actor` stays the agent's because
provenance is never rewritten: the words above `## Amendments` are still the agent's. A
successor is asked about afresh because the owner confirmed the record it replaces, not this
one: their earlier words, quoted here, would record a confirmation they never gave for it.

## Decisions

The decision kind is the one `[working] decision` names. Its link fields are lists:
`supersedes` / `superseded_by`, `relaxes` / `narrowed_by` (each pair written together),
`creates`, `amends`, `retires`, `tasks`, `code` (`file` or `file::symbol`) and `refs`.
No decision write stores an index: the listing is printed on demand (§ context-engine
decision index) and published (§ context-engine site build).

### context-engine decision supersede

`<new>` replaces `<old>`. Both must be decisions; `<new>` must not be SUPERSEDED or REJECTED
itself, and must not already supersede `<old>`. Re-read both files, then in one change:
on `<old>` set `decision_status: SUPERSEDED`, `status: deprecated` and add `<new>` to
`superseded_by`; on `<new>` add `<old>` to `supersedes`. Log one `supersede` line per file.

**Why:** a link written in one direction only is a half-truth, and the engine writes both under
one lock. `status: deprecated` is OKF's word for "kept for history, no longer current" (§9.3).

### context-engine decision set

Set or unset a decision's own frontmatter keys: `decision_status`, `reopen_if` (a REJECTED
decision only: what would justify reopening it), `tags`, `title`, `description`, and any key no
operation owns. Values take the kind's declared types, as in § context-engine task set
(`tags=a, b` is a list; an empty value means null). Refuse, and write nothing, when:
- the key is `type`, `id` or `body_sha` (engine-owned, never changed);
- the key is `status`, `supersedes` or `superseded_by`: use `decision supersede`, which writes
  both directions and the derived OKF status;
- the key is a link field (`relaxes`, `creates`, `amends`, `retires`, `tasks`, `code`, `refs`):
  use `decision link`, or `decision relink` to move a `code` link; no operation removes an id
  link (ids are never reused, so one never goes stale);
- the key is `narrowed_by`: it is the backlink `relaxes` writes, so run `decision link <other>
  <id> --rel relaxes` on the decision that narrows this one;
- the key is `answers` (written once by `ask answer`), or provenance: `actor`, `decided`,
  `session`, `attribution`, `imported_from`, `generated`, `ruled_by` or `aliases` (who decided, who
  ruled on the owner's delegation, when, in which session, from what source and under which earlier
  ids are fixed when the record is created; note a correction with `record amend`);
- the key is `verified` (the owner endorsed it): an endorsement is the owner's words, so record
  them with `record amend <id> "<the owner's words>"`;
- the key is `reopen_if` and the decision would not be REJECTED after the change, or the change
  moves a decision off REJECTED and keeps its `reopen_if` (unset it in the same call);
- the new `decision_status` is `SUPERSEDED`: use `decision supersede <id> <new>`;
- the new `decision_status` is `ACCEPTED`: use `decision accept <id>`, which checks the decision
  is PROVISIONAL and logs the review;
- the key is `stale_after` and its value is empty, not a timestamp with its offset, or already
  passed, as in § context-engine task set (G-D4; off with it): unsetting it is taken;
- the decision is already SUPERSEDED and the change touches `decision_status`: a superseded
  decision stays superseded; to bring it back, record a new decision and supersede the one that
  replaced it.

Re-read the file, then change only the named keys in place (a new key goes last). The body and
`body_sha` do not change. Log one `set` line with the changed keys. **Move an owner's decision's status (to BUILT, REJECTED or
DEFERRED) only on the owner's words, recorded first** (`record amend <id>
"<words>"`): the engine cannot tell whether the owner said so, so this rule is the only control.

**Why:** a decision's status, tags and `reopen_if` are not covered by `body_sha`, so they can be
kept current without touching the words; but the keys other operations own carry a rule
(both directions of a link, the derived OKF status, who decided), and a raw edit would skip it.

### context-engine decision accept

`decision accept <ids>` marks PROVISIONAL decisions reviewed: each becomes `ACCEPTED`. Run it once
the owner has reviewed the agent's calls named (the close digest flags each one), never before.
`<ids>` are ids, aliases or ranges (`DEC-n..m`, or `DEC-n..DEC-m`: one prefix, low to high, at most 500),
comma- or space-separated, in any number. `--owners` adds every decision still PROVISIONAL that is
the owner's own (`actor` is `owner` or `human:<id>`, in any case, and no `ruled_by`): the born-ACCEPTED
rule of § context-engine record new applied to decisions recorded before it, once, after the
upgrade that added `ACCEPTED`. Spell out the ranges, then refuse, and write nothing, when:
- no id is given and no `--owners`;
- the decision kind's `decision_status` enum in `context.toml` has no `ACCEPTED`: add it, as the
  kit's `context.toml.template` has it;
- an id names no decision (a range must not span a number no decision has), or a record that is
  not a decision;
- a decision named is not PROVISIONAL: say which, with its status (a SUPERSEDED one is history,
  with the decision that replaced it; a BUILT one a task carried out; an ACCEPTED one already
  reviewed);
- `--owners` alone finds no owner's decision still PROVISIONAL.

Check every decision first; then, on each, replace `decision_status: PROVISIONAL` with
`decision_status: ACCEPTED` in place. The body and `body_sha` do not change. Log ONE `accept` line
for all of them, not one per decision: `id` the ids folded into ranges, `path` the first file,
`ids` and `paths` every id and file in order, `fields` `["decision_status"]`. It is not a milestone:
the line rides along to the next commit, which carries every file in `paths`.

**Why:** PROVISIONAL means one thing, an agent's call awaiting the owner's review:
an owner's own word is born ACCEPTED, so the review queue holds only calls someone
must still look at. One op for a whole review, in one line, keeps a review of sixty calls from
being sixty `decision set` lines. `decision set` refuses ACCEPTED so that every acceptance is in
the op log; lint (G-W4) fails an agent's call that is ACCEPTED with no `accept` line naming it.
Whether the owner really reviewed the calls is the caller's word: the line records who said so.

### context-engine decision link

Add `<target>` to one link field of a decision. `--rel` names the field; without it, a board
item's id goes to `tasks` and anything with `::` or `/` to `code`; a decision needs
`--rel relaxes` for one this decision narrows (which also adds this decision to the other's
`narrowed_by`) or `--rel refs` for one it only relates to; one it replaces is § context-engine
decision supersede. Check the target
first: a record id must exist; a criterion id must be a row (`| AC-n |`) of the criteria file
`[working] criteria` names; any other id is refused; a `code` path must be written from the
repo root with `/` (no `\`, no `..`, no leading `/` or drive), name a file (not a directory or
the root) that exists with its exact case, and every dotted part of its symbol must be
non-empty and appear in the file. A target already in the
field is a no-op: stop.

**Why:** `why()` answers from these links, so a link to something that does not exist answers
wrongly. Existence is checked when the link is written; G-T3 fails a current decision's link
that later stops resolving: move it with § context-engine decision relink, and do not
hand-edit `code:`.

### context-engine decision relink

`decision relink <id> <old> <new>` replaces one `code:` link of a decision when the code it
names moves. Refuse, and write nothing, when:
- `<old>` and `<new>` are the same: nothing to do;
- `<new>` does not name code, as § context-engine decision link checks it: a blank, a directory
  or a path outside the repo is refused, so a dead link is never replaced by nothing;
- the id names no decision, or the decision is `status: deprecated` (SUPERSEDED): its links are
  history, and G-T3 does not check them; link the code to the decision that replaced it (one
  `decision link <successor> <new>` per successor);
- `<old>` is not in its `code:`: if it is in another link field, that link is an id, and ids
  never move; otherwise name one of its `code:` links, or add one with `decision link`.

Re-read the file, then replace `<old>` with `<new>` in place in `code:` (if `<new>` is already
there, drop `<old>` and keep `<new>` once). The body and `body_sha` do not change. Log one
`relink` line with fields `["code"]`. **Relink only to where the same behaviour now lives**, in the commit that moves it: if
what the code did is gone, the decision no longer describes the code, so supersede it.

**Why:** a moved file or renamed symbol leaves the link dead, G-T3 red and `why()` answering
from nothing. `code:` is written by an operation, so a hand edit would skip the existence check
and leave no op-log line. Whether the new code carries out the decision is a judgement only a
reader can make; no operation can make it (a residual in the engine's register).

### context-engine decision find

A read. List the decisions matching every filter given: `--status` (case-insensitive),
`--tag`, `--links-to <target>` (as § context-engine why matches it) and `--text` (in the id,
title, description or body, case-insensitive). One line each by id:
`<id> <status> <title> (<actor>, <decided>)  [<tags>]`.

**Why:** "what did we decide about X" is answered by one-line rows, not by opening every file.

### context-engine decision show

A read. Print `<id> · <decision_status> · <title>`, then `actor`, `decided`, `session` and
every non-empty link field and `tags`, one per line, then the body, or only the
`## <section>` asked for (e.g. `Amendments`).

**Why:** one decision's status and links come first, so a reader rarely needs the whole file.

### context-engine decision index

A read (MCP `decision_index`): every decision, from its frontmatter alone, never its prose. Write
nothing; print:
1. `# Decisions`, a blank line, then the line
   ``Printed on demand from each decision's frontmatter by `context-engine decision index`; never stored.``
2. For each `decision_status` in the kind's enum order (then any other value, alphabetically)
   that has decisions: a blank line, `## <status>`.
3. Under it, group by each decision's first tag (`untagged` if none): tags alphabetical,
   `untagged` last. Per group: a blank line, `### <tag>`, a blank line, then one line per
   decision by id number:
   `- [<id>](<the record's path from the repo root>): <description, else title> *(<actor>, ruled by <ruled_by> where set, <decided>)*`,
   followed by `; superseded by <ids, comma-separated>` when `superseded_by` is set.

**Why:** the listing is what a stored `index.md` used to hold, and printing it on demand means it
can never be stale and no two sessions collide on it. OKF v0.2 §8 lets a consumer synthesize an
index on the fly, and the published site (§ context-engine site build) carries it for a reader
without the CLI (FRAMEWORK §6.5). A repo that still sets `[working] decision_index` loads as
before; orient names the key and the stored file to remove.

### context-engine decision import

Turn a single-file decision log (FRAMEWORK §11.4) into one record per entry. Do it on a copy and
on a branch; nothing is written if any check fails.
1. Each entry's id becomes the kind's `prefix` and the entry's number (`D-27` becomes `<prefix>-27`);
   an entry whose id changed keeps its old id in `aliases: [D-27]`, under G-D2's alias rule (§
   lint). No two entries may end with one id, an entry in the kind's own prefix is written as the
   engine writes it (`<prefix>-1`, not `<prefix>-01`), no new id's number may already be in the series, no
   old id may already be a record's alias, and no rejected row may already be a REJECTED record
   with an `imported_from` and the same title (each was imported before): import only the
   entries added since, without the rejected table, and record one whose number is taken as
   `context-engine record new` does, with `aliases: [<old id>]` and the next free number.
2. Each entry starts at a heading `### <id> -- <title> *(<attribution>)*` at the start of its
   line, optionally followed by ` -- <note>` (`—` and `–` work as the dash), and runs to the next
   entry heading; the last entry runs to the end of the log. A `#` or `## ` heading inside it
   stays in its text, except headings with nothing under them just before the next entry or the
   end (a log's date groups, `## 2026-08-29`). A heading inside a code fence or an HTML comment
   block (`<!--` to `-->`) is text. Its text is copied verbatim; an entry with no text stops the
   import. Text no record holds (before the first entry, bar the rejected table of step 5, and
   those empty headings) is not imported, and the output names it by line; the dry run's id map
   also lists every `#`/`## ` heading kept inside an entry, by line, so check that each belongs
   where it landed (a group heading with an intro under it, such as `## Earlier` followed by a
   paragraph, lands in the entry above, and an appendix after the last entry lands in that
   entry: move it above the first entry in the copy), and, in a list of its own, every `###` to
   `######` heading kept inside one (sub-headings, kept as text). A log with no entry heading
   stops the import, naming its `###` headings. Line numbers in the output are those of the file
   given: after normalising a copy, they are the copy's.
   The import stops, naming every such line at once, on a line it cannot read:
   - a `### ` heading that starts with an id but is not an entry heading (`### D-107 FIX`,
     `### D-4 (amended)`, a letter id such as `### D-159b -- ...`, and an entry heading not at
     the line's start or with its id dressed, ` ### D-2 -- ...`, `###  D-2 -- ...`,
     `### **D-2** -- ...`, `### [D-2] -- ...`). It is never folded into the entry above: move
     it, in the copy, under a `## Amendments` heading at the end of its parent's text
     (`D-107`'s, `D-159`'s), where an id-like heading naming that same entry is part of the
     entry; or write it as an entry heading of its own; or make it `####`. An amendment written
     as a full entry heading of its parent's id under that `## Amendments` stops the import too:
     write it `### D-107 FIX`, with no dash or attribution;
   - an attribution with no actor, a date where the actor goes (`*(2026-07-25)*`), a date inside
     the actor with no comma before it (`*(made normative 2026-07-24)*`), or no date; write
     `*(unrecorded, <date>)*` when the log does not say who decided, never a guessed actor (the
     date from git history if the log has none);
   - a `#` or `## ` heading after an entry's own `## Amendments` (its text would be an
     amendment, not stamped), and a `## Amendments` line inside a code fence or comment in an
     entry (the record's stamp would end there);
   - a code fence or an HTML comment that is never closed.
   An entry whose text has its own `## Amendments` heading keeps it, the last entry too: the text
   above it is stamped (`body_sha`), the text under it becomes the record's first amendments,
   verbatim, and a heading note (step 4) is appended after them.
3. Frontmatter: `title`; `actor`, the attribution up to its first comma or semicolon (`unrecorded`
   when the log names no one, kept as written); `decided`,
   the last `YYYY-MM-DD` in it; `session`, the importing session; `decision_status: BUILT`
   unless the note says `SUPERSEDED by <id>`; `attribution`, the whole parenthetical, when it
   says more than `<actor>, <date>`; `imported_from`, the log's path.
4. The note: `SUPERSEDED by <id>`, `supersedes <id>`, `relaxes <id>`, `relaxed by <id>` and
   `narrowed by <id>` become links to the new ids, written on both records (a link to an id not
   in the log stops the import); a SUPERSEDED record also gets `status: deprecated`. The note
   itself is kept, old ids and all, as an amendment: `imported heading note: <note>`.
5. The first `## ` section before the first entry whose heading says "rejected" and holds a
   table outside a fence or comment (one inside an entry is that entry's text; a second table in
   that section stops the import): each row becomes a
   `decision_status: REJECTED` record with the next free id above the highest imported one, in
   row order, and no alias. Title: the first
   cell; body: the line *Imported from the table under*, the heading as `` `## <heading>` `` and
   a colon, then a blank line, the header line, the separator line and the row, verbatim;
   `actor: agent`; `decided`: the last date in the row's middle cells, else the date of the
   first entry its last cell names; `refs`: the new ids of the entries its last cell names;
   `attribution`: which table, and where the reasoning is.
6. Stamp every record and log one `import` line each.
7. The id map, one table row per record (`| id in the log | record | status | title |`, a
   rejected row's first cell `(rejected table row <n>)`), under a title and a FROZEN line, saved
   as an evidence doc, with a `## Not imported`, a `## Headings kept inside entries` and a
   `## Sub-headings kept inside entries` list (step 2); `--dry-run` prints it, ready to save, after its summary, and writes nothing to the
   repo, not even the lock file. Every entry the import set BUILT because the log gave no status is
   reviewed, and `decision set <id> --set decision_status=<status>` corrects any that was parked,
   deferred or not built.

**Why:** the import gives a homogeneous series and keeps each number, and each old id stays a
resolvable alias, so a commit message or a docstring that cites `D-27` still finds its record;
verbatim text and a stamp make the move checkable line by line; all or nothing means a
half-import never lands. A rejected row becomes a record so "what was rejected that touches X" is
a query.

### context-engine why

A read. The decisions whose link fields (and `answers`) name `<target>`: an id, a file, or a
`file::symbol`. A file matches its own `file::symbol` links, and a `file::symbol` matches a link
to the whole file. One line each by id, as § context-engine decision find, ending
`via <field, ...>`.

**Why:** "why is this code like this?" is answered by the decisions that link it, in one call,
instead of grepping a log.

## The session's own commits

### context-engine commit

`context-engine commit [<paths> ...] -m "<title>" [-m "<paragraph>" ...] [--sweep] [--co-author
"<name> <email>" | --model <model-id>]` (or `-F <file>`; over MCP `commit(paths, message, sweep,
co_author, model)`). It takes your session's uncommitted op lines of your lane (`CE_AGENT`),
not another session's or lane's (§ What every write does, **Commit the write**), and its reply names the lines it
left: `left uncommitted: <n> op lines of <key>` (`<key> lane <lane>` for a subagent's). `--sweep` takes every line, whoever wrote it:
use it once the owner, or housekeeping's stale-stub step, says that session is dead. **Run the gate first:** it runs none. When `[gate] require` names
a scope, it refuses a path in that scope (a directory: any file under it) unless a green gate run
covers the scope's id (§ context-engine gate status); the paths the op log's lines name are not
counted. By hand: check that first, then:

1. Re-read `git status` and `git log -3`: another session may have committed since.
2. Take your session's uncommitted op lines of your lane (with `--sweep`, every line) and the paths they
   name, as § What every write does, **Commit the write**, says.
3. `git add -- <each named file git has never seen>` that you take, and every new file under a
   directory you name (with lines left behind, add them to the temporary index of step 5 instead). Do not name a path git ignores (the engine refuses one; `git add
   -f` it first if it belongs in git), the whole repo (`.`), a path outside the root, or
   pathspec magic (`:/`). A record the op lines name must parse (as § What every write does
   says).
4. The message is your text, then the engine's lines, then your closing trailer paragraphs: `Ops:
   <op> <id>, ...` naming the op lines the commit carries, and the `Also carries ...` paragraph
   when another session's lines go in. Then `Session: <key>` and the `Co-Authored-By:` line (below), added to your last paragraph when
   that is a closing trailer paragraph (every line `Token: value`, or indented to continue the
   one before, and one of them `Session:`, `Closes:` or a `<who>-by:` such as `Signed-off-by:`),
   else as a paragraph of its own, so a last paragraph of `Sabotage:` lines stays body: git reads
   trailers only from the last paragraph. A closing paragraph that has the `Session:` trailer
   already keeps it, as it keeps a `Co-Authored-By:` naming the same address, and one naming
   another session is refused.
5. With no line left behind: `git commit --only -F <message file> -- <your paths>
   docs/ledger/ops.jsonl <the paths the lines name>`; if it fails, `git rm -q --cached` what step 3
   added. With lines left behind, never `--only` on the op log (it would carry their lines
   without their files): use the temporary-index steps of § What every write does, **Commit the
   write**, and write the op log blob's file from Git Bash (PowerShell's `>` writes UTF-16).

Before committing it runs lint and refuses when a failure sits on a path the commit would
carry (a given path, a file under a given directory, the op log, a path a taken op line names),
naming the failure and its remedy; a failure on any other path is not this commit's, and the
gate still reports it. By hand: run `context-engine lint` and check no failure names one of your
paths.

It writes no op line: it changes no record. It commits whatever `[engine] commit` says, and
refuses (exit 2) whenever it commits nothing, naming why (among them: nothing changed among
the paths given). Its reply counts the files git committed.

**Every agent commit's message ends the same way,** through the engine or by hand (a builder in
a worktree, a plain `git commit -o`): one last paragraph holding every trailer, `Session: <key>`
with the others the harness or the task adds (`Co-Authored-By:`; `Closes: <id>` only where
FRAMEWORK §6.10 item 1 puts it), one per line, with no blank line among them and nothing after
them. Git reads trailers only from a message's last paragraph, so a `Session:` line followed by
a blank line and another trailer is no trailer to git, and § Reconciling foreign commits
(procedures.md) reads the commit as foreign. The template, whose `Sabotage:` lines
([sabotage.md](sabotage.md) step 5) are a body paragraph of their own:

<!-- commit-message-template -->
```text
<id>: <what changed>

<why, and what the commit carries, in as many paragraphs as it needs>

Sabotage: <what was broken> -> <test> red

Closes: <id>
Session: <key>
Co-Authored-By: <name> <email>
```

Through `context-engine commit`, leave `Session:` and `Co-Authored-By:` out: the engine adds
both to that paragraph, and so does every commit it makes of its own writes. A work commit has
no `Closes:`, so its message ends in the `Sabotage:` lines; the engine reads a last paragraph
with no `Session:`, `Closes:` or `<who>-by:` trailer (`Co-Authored-By:`, `Signed-off-by:`) as
body, and puts its two lines in a paragraph of their own after it, so the `Sabotage:` lines
stay body. The
`Co-Authored-By:` value is, first hit wins: `--co-author "<name> <email>"` (`none` for no line);
the name `--model <model-id>` reads; `CE_CO_AUTHOR` in the environment (`<name> <email>` or
`none`); else the actor's model: `CE_ACTOR=claude-code/claude-opus-5-5` gives `Claude Opus 5.5
<noreply@anthropic.com>`, and a `claude...` harness whose model it cannot read gives `Claude
<noreply@anthropic.com>`. An actor with no `/` (`human:<id>`, `process:<id>`) gets none, so a
person's commit never says an agent co-wrote it. A `Co-Authored-By:` line you leave in the
message's last paragraph stands, and one for the same address is not added twice. By hand, write
the line yourself, as the template does.

**Why:** a milestone policy leaves board writes uncommitted in the tree, so a plain
`git commit -o` of a record would commit it without its op line, and G-W7 fails that commit on a
clean checkout. Committing through the engine takes the lines and the files they name together,
under the lock, so a session's own commit never splits them (FRAMEWORK §5.5 rule 4). It runs no
gate, as `git commit` runs none: the gate is the session's step before it, so a gate is never
skipped silently by a tool that was thought to run it.

## The scoped gate

The gate's result is recorded against a content id of each scope's files (`[gate]` in
`context.toml`; FRAMEWORK §8.5 G-V1), so a scope whose files have not changed since a green run
needs no re-run. Each id is computed the same way:

1. List the files: `git ls-files -z --cached --others --exclude-standard` from the root, each
   once, sorted by its UTF-8 bytes. Keep those a scope glob in `include` matches and none in
   `exclude` matches, as whole `/`-separated paths: `**` spans directories (`docs/**`,
   `**/*.md`), `*` and `?` stay inside one part. Skip a listed file the work tree lacks, the
   engine's lock file (`[engine] lockfile`) and everything in the run ledger's directory.
2. Hash, with SHA-256: the header `context-engine-gate/g1/<scope>/` plus the scope as
   JSON with sorted keys and sorted lists (`{"exclude": [...], "include": [...], "readers":
   [...]}`, `", "` and `": "` separators) and a NUL byte; then per file its path, NUL,
   `<size>:<sha256 hex of its bytes>`, NUL.
3. The id is `g1/<scope>:` and the first 16 hex characters.

A scope with `readers` (the tests that read files it leaves out) has a second id,
`<scope>+readers`: the same steps over every file (`include = ["**"]`, `exclude = []`, its
`readers`), named `<scope>+readers`.

*Why: content, never a commit id, so `git add` and `git commit` leave a green run green; a file
gone from the tree is skipped as it is once its deletion is committed, so that commit leaves it
green too; the scope's own definition is in the hash, so a changed glob matches no old run.*

### context-engine gate run

`context-engine gate run [--readers]` (CLI only). By hand: compute every id; run `[gate]
command` from the root (`{python}` is the engine's interpreter; with `--readers`, every scope's
readers appended to it, and only the `+readers` ids), its output to a log file; compute the ids
again. Append one JSON line to `[gate] ledger` (default `.context/gate/runs.jsonl`): `started`,
`seconds`, `status` (`PASSED` on exit 0, else `FAILED`; `INCONSISTENT`, with empty `scopes`, when
every id moved during the run), `exit`, `part` (`all` or `readers`), `command`, `args` (the
readers appended), `scopes` (id name -> id, only the ids that did not move), `moved` (the names
of those that did, when any did), `log`, `head`, `session`, `actor`. The ledger must be
gitignored and untracked (a fresh directory gets a `.gitignore` of `*`).

With `[gate.parts]` (`gate run [--part NAME]... [--stale]`): do the above once per part, in
declared order, with the part's `command` (run with `CE_GATE_PART` set to the part's name, so
the scope guard checks it against the part's scope) and every id; `part` is the part's name and
`args` empty. Go on past a part that fails, and append each line as its part ends. `--part` runs the
parts named; `--stale` runs only the stale ones (as § gate status), and none when none is stale.

**Why:** a recorded result is what lets the next commit skip a re-run, and a run whose files
moved under it tested nothing nameable for those ids; the ids that did not move were tested
whole, so another session's write during a long run costs only the ids it touched. An unignored ledger would sit inside the ids it records,
so every row would be born stale. `--readers`: a change outside a scope cannot change what its
other tests read, so only the tests that read it need to run again. A part records every id, so
a part keyed on every file still counts toward the code scope; and a long gate wants every
part's verdict from one pass. A gate run is out of the MCP's scope (FRAMEWORK §8.0.2).

### context-engine gate status

`context-engine gate status [--require <scope>[+readers|+parts]]` (over MCP `gate_status`). By
hand: compute each id; it is covered when the newest ledger line whose `scopes` holds that id has
`status` `PASSED` and the `command` configured now. With `--require <scope>`, exit 1 unless that
scope's id is covered: what `context-engine commit` checks for a path in it. With `--require
<scope>+readers`, its `+readers` id too: what a hand edit to a file the scope leaves out needs.

With `[gate.parts]`, read only each part's own lines (`part` its name; `PASSED` or `FAILED`):
a scope is covered when, for every part, the newest such line holding the scope's id passed
with the part's `command` now. A part is fresh when it covers its own `scope`, and stale when it
is not fresh or does not cover the `require` scope. `--require <scope>+parts`: the scope
covered and every part fresh, what the close and the push need.

**Why:** "did the gate pass on these files?" is answered from the ledger, not from memory. The
newest line decides, so a green run then a red one on the same files is red; and a run of
another command proves nothing about this one. The commit checks coverage, never freshness: a
part keyed on every file goes stale on every op-log line, any session's.

## Bootstrap

### context-engine init

Give a repo the framework's files (FRAMEWORK §6.9). The kit is `src/context_engine/kit/` in the
engine repo at the pinned ref (or `context_engine/kit/` in the installed package), and its
`README.md` has the same steps.
1. If `context.toml` exists, keep it: check it loads (`context-engine lint` names what it
   refuses; stop and fix that first), and in steps 3 and 5 create only what it points at (a kind's
   `dir`, a `[working]` path, a route's `path`, the lock's directory).
2. Otherwise copy `context.toml.template` to `context.toml`, and set `[kinds.task]` `prefix` to
   the one the owner chose: 2 to 5 capitals naming the board (the kit's `TASK` is a placeholder;
   without the owner's choice, stop and ask, as the engine refuses). To change another kind's id
   prefix, edit `prefix` in its `[kinds.<name>]` table. A prefix is 2 to 5 capitals; none may be
   reserved (`EXP`, `HAZ`) or another kind's standard one (FRAMEWORK P3: `DEC`, `STORY`, `AC`,
   `ASK`, `RISK`, `GOAL`, `BATON`, `PAUSE`), no two kinds may share one, and none may collide with
   ids the repo already uses: search the files git tracks (`git ls-files`, skipping binary files
   and any over 1 MB) for each prefix as an id of its own, `<PREFIX>-n` or `<PREFIX>n`, and warn
   for each prefix found, with how often and where first.
3. Copy `HANDOFF.md`, `docs/charter.md`, `docs/criteria.md`, `docs/ledger/session-log.md` and
   every runbook under the kit's `docs/runbooks/` (`manual-mode.md`, `procedures.md`, `gate.md`,
   `sabotage.md`, `fan-out.md` and `release.md`) from the kit to the same paths, skipping any that
   exist.
4. `CLAUDE.md`: if it holds no `<!-- context-engine kit <version> begin -->` marker, append the
   kit's `bootloader.md` after a blank line (create the file with a `# <repo>: contract` title if
   it is absent), keeping the file's line endings. If it already has its own Gate (a level 1 or 2
   heading reading `Gate`, alone or followed by a parenthesis, a colon or a dash, outside code
   fences and HTML comments), append the kit block only, up to its end marker, so that gate stays
   the one gate (it should end with `&& context-engine lint`). If it holds an older version's block, or a
   block with no `@` import line (the inline block of a kit from before `bootloader.md`), leave it for the
   upgrade to replace (kit README § Upgrade); never add a second block. Copy the kit's
   `contract.md` to `docs/framework/contract.md` if absent. If `.claude/settings.local.json` is
   absent, write it with the `SessionStart` hook from the kit README § Hook (this machine's
   absolute paths) and `.claude/.gitignore` holding `settings.local.json` if that is absent too,
   unless the repo's hook comes from the Claude Code plugin (kit README § Claude Code plugin): then
   write no hook (the engine's `--skip-settings`), so orient does not run twice.
   If it exists and has no such hook, it holds the owner's grants: show the owner the entry and
   ask; with a yes, add it to `hooks.SessionStart`, keeping every other key and its order (the
   engine's `--merge-settings`); with a no, leave it (`--skip-settings`). Git must ignore the file,
   since the hook holds this machine's paths: when you write the hook and neither
   `.claude/.gitignore` nor the repo's `.gitignore` ignores it, write `.claude/.gitignore` holding
   `settings.local.json` if that is absent; on every later run that finds the hook and nothing
   ignoring it, say so.
5. For each of `docs/decisions`, `docs/stories`, `docs/working`, `docs/evidence` and `tasks`
   that does not exist, create it with an empty `.gitkeep`. Create
   `.context/.gitignore` and `.scratch/.gitignore`, each holding the line `*`, if absent. If no
   line of the repo's `.gitignore` ignores `.context-engine/` (the engine's environment, kit
   README § 1), append `# context-engine: the engine's pinned env (kit README § 1)` and
   `.context-engine/` after a blank line, keeping the file's line endings (create the file if it
   is absent). If no line of the repo's `.gitattributes` gives the op log (`oplog` in
   `context.toml`) `merge=union`, append `# context-engine: two branches' op lines merge line by
   line (FRAMEWORK §5.5 rule 4)` and `<op log> merge=union` the same way; a path with a space or a
   wildcard needs quoting there, so write that line with care, or warn and leave it.
6. Report in one line what you created, what you skipped, and a warning for each thing the
   owner must act on (the settings file's ignore line; a prefix already in use; a checkout that
   converts line endings, where `git config --get core.autocrlf` prints `true` and
   `git check-attr text eol -- context.toml` shows neither `eol: lf` nor `text: unset`, whose
   fix is the kit README § 2a; on a dry run, the
   settings question and a task prefix not yet chosen), and,
   when you wrote `context.toml` or the kit block, the next steps: fill `docs/charter.md` with the owner, replace the Gate stub's
   placeholder in `CLAUDE.md` with the repo's test command (or, with no stub, check the repo's own
   Gate ends with `&& context-engine lint`), run lint, commit the kit's files by path, register
   the MCP server with `claude mcp add --scope local context-engine -- "<interpreter>" -m
   context_engine.mcp` and ask the owner to restart Claude Code (kit README § 1), then
   `session start`. Never overwrite an existing file, and
   never install anything into the repo's environment (the engine's own is `.context-engine/env/`,
   made before init, kit README § 1).

**Why:** a new repo should get every file the procedures expect in one step, the runbooks
included, so it can follow every procedure by hand from its own tree, and an existing repo
must not lose a byte of what it has: it maps its components onto the routes instead of moving
them (FRAMEWORK §11.4). The engine also refuses a file that appears while it runs (it creates
each one only if it is still absent) and a `CLAUDE.md` that changed since it read it; by hand,
`git status` before committing shows the same.
