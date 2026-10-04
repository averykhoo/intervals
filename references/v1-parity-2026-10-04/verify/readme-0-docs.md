# verify readme-0 (docs lens): illustration to-do truncated in references/todo-from-v1-readme.md

claim: MISSING. verdict after check: **MISSING (survives)**, refuted=false.

## evidence
* `grep -n 'redo illustrations' -A12 archive/v1/README.md references/todo-from-v1-readme.md`:
  v1 README:66-73 has 7 sub-lines; todo file:9-13 has only the first 4. absent from the todo file:
  - `    * or use a different aspect ratio? 600x800?` (v1:71)
  - `  * use better colors` (v1:72)
  - `  * zoom into x axis a bit to show there are infinite lines near there` (v1:73)
* `git show 2f3a895:archive/v1/README.md | sed -n 64,75p`: the three lines were already in v1's README
  at the commit that created the todo file (2f3a895, 2026-09-26); archive/v1/README.md has one commit
  only (6d4851f). so the omission was at copy time, not a later v1 edit.
* `git log -- references/todo-from-v1-readme.md`: only 2f3a895; never amended.
* `git grep -n -i -E 'aspect ratio|600x800|better colors|infinite lines|zoom into' -- ':!archive'`: no hits.

## records searched
* todo-from-v1-readme.md:4 "kept here so they survive the deletion of `archive/v1/` ... copied verbatim." (contradicted)
* v2-plan.md:1956-1957 "**H5**: the v1 README's reading list and illustration to-do are kept, moved to
  `references/todo-from-v1-readme.md` so they outlive `archive/v1/`" (requires the whole to-do; no trimming recorded)
* v2-plan.md:1955 "**H4**: delete `archive/v1/` after v2 is stable" -> loss becomes permanent from the working tree.
* HANDOFF.md:416 "the v1 README's leftovers moved to `references/todo-from-v1-readme.md` (H5 done)" (no trimming noted).
* no record in v2-implementation-plan.md, README.md, HANDOFF Q21, references/ or docstrings drops these three lines.

## note
not a library capability; a doc-preservation gap. git history keeps the lines after H4, but the
recorded intent (H5, "verbatim") is a tracked copy. fix: append the three lines at todo:13.
