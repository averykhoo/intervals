# verify readme-0: illustration to-do truncated in references/todo-from-v1-readme.md

verdict: claim SURVIVES (not refuted). corrected_verdict: MISSING (a documentation-preservation gap, not a code capability).

## reproduction (own probe)
probe: `.scratch/v1-parity/verify/readme-0-probe.py` (checks each line of v1 README's "* redo illustrations" subtree
against the todo file, line-exact; controls assert one known-kept line and one known-absent line).
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/readme-0-probe.py`
output:
    KEPT v1:66-70 (5 lines)
    LOST v1:71:     * or use a different aspect ratio? 600x800?
    LOST v1:72:   * use better colors
    LOST v1:73:   * zoom into x axis a bit to show there are infinite lines near there

## search for a refutation
* `git grep -i -e 'aspect ratio' -e 'better colors' -e 'infinite lines' -e '600x800' -- . ':!archive/v1'` -> rc=1, nowhere else tracked.
* not a later edit of v1: `git show 2f3a895:archive/v1/README.md` (the commit that created the todo file) already has
  lines 71-73; archive/v1/README.md last changed in 6d4851f (before). so the lines were dropped at copy time.
* no doc records dropping them: the todo file header says "copied verbatim"; v2-plan.md:1956 H5 "the v1 README's reading
  list and illustration to-do are kept, moved to `references/todo-from-v1-readme.md` so they outlive `archive/v1/`";
  HANDOFF.md:416 "(H5 done)". H4 (v2-plan.md "delete `archive/v1/` after v2 is stable") would lose them.

## minimal fix (for the session, not done here: read-only)
append under line 13 of references/todo-from-v1-readme.md the three lines v1 README.md:71-73, verbatim.
