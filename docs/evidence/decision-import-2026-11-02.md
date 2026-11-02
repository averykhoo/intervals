# Decision log import: how `docs/decisions.md` became 87 records (2026-11-02)

FROZEN 2026-11-02 (session 2026-11-02a, context-engine CE-11 adoption): how the frozen log was normalised and imported,
and whose call each transform was; never edited.

The pre-engine log `docs/decisions.md` (at `fa4cf12`) stays FROZEN in place, because code, tests and the README cite
its `D<n>` ids and its heading titles. The engine's importer reads only `### <id> -- <title> *(<actor>, <date>)*`
entries, and this log had none (a D table of rows D1-D30, then a newest-first log of date-led `###` entries), so a
normalised copy, `decisions-normalised-2026-11-02.md`, was written by a script and imported. Its id map is
`decision-import-idmap-2026-11-02.md` and the line-conservation check of the result is
`decision-import-check-2026-11-02.md`, both beside this doc.

## The rules (an agent's calls, PROVISIONAL until the owner confirms the merge)

1. Every line of the log that carries a decision is in exactly one record, verbatim and in its original order; the
   preamble (how to add an entry, the next free ids), the table's header and separator, and the two `##` section
   headings are not imported and stay in the frozen log.
2. Ids: each D-table row `D<n>` is record `D-<n>` (D-1..D-30), its row the record's body; each log entry naming its
   own id (D31..D36) keeps it; every other log entry, and the two paragraphs under the table that state decisions,
   take a new id, D-37..D-87, oldest first (the log's own "D37 is next" rule is superseded by the engine's series).
3. Titles: a log entry's heading without its date; a table row's from its question cell (synthesised: 30 rows and the
   two paragraphs). The original heading line is the first line of its record's body, so a citation of a heading
   title still finds its text.
4. Actor: `owner` only where the text says so mechanically (a row "decided <date> by owner", an entry opening
   "the owner, <date>"); otherwise `unrecorded`, never guessed. Dates: the heading's, a row's "decided <date>", or
   for three undated entries the date git first shows their text (`date from git` in the attribution).
5. Status: every record is BUILT as imported; none was marked SUPERSEDED at import. The entries whose text carries a
   status marker (superseded, amended, awaiting the owner, deferred, confirmed) are listed on the board for review.
6. With the prefix `D` the records keep the log's numbers, so no record has an alias: `D17` in code is `D-17`.

## Checks

The importer's dry run said 87 entries before the import; the conservation check (`check_conservation.py`, which
reads only the original log and the records) passed, and was seen red on 15 kinds of break (a dropped, duplicated,
foreign, swapped or moved line, a deleted or stray record, a wrong title, actor or date) before it was believed.
The scripts are kept with the CE-11 evidence in the context-engine coordinator's backup, not in this repo.
