# verify (docs lens): bool(DateTimeInterval) / bool(TimeDeltaInterval)

claim: UNDOCUMENTED_DIFFERENCE (v1 time classes: object default, always True; v2: bool == not is_empty).
result: NOT refuted. corrected verdict: UNDOCUMENTED_DIFFERENCE (low severity).

searched (grep -i `__bool__|truthi|bool(|falsy|truthy` over every *.md, plus the time sections by hand):
* v2-plan.md:111 "`__bool__` = non-empty (set precedent)" -- section "set operations and size" of the core
  MultiInterval. adjacent: does not name the time classes.
* v2-plan.md:2688 "`__bool__` = non-empty, explicitly (set precedent)" -- "more v1 retirements", core class. adjacent.
* v2-plan.md:1197-1210 "the time layer (M8)": "every set operation, relation, comparison and arithmetic op is the
  numeric class's on those seconds" ... "immutable, hashable ... and pickle". truthiness is not a set operation,
  relation, comparison or arithmetic op; not named. adjacent at best.
* v2-implementation-plan.md:143 (M1 dunders list, `__bool__`, `__len__` = piece count) -- core class only.
* v2-implementation-plan.md:409-438 M8 decisions + done record: lists comparisons -> TruthSet ("a v1 caller's
  `if a < b:` can raise; documented"), hash, pickle, "added from v2: before after ... sort_key, hash, pickle,
  from_seconds" -- no `__bool__`, no `__len__` (also new vs v1: v1 time classes had no __len__ either).
* v2-implementation-plan.md §4 time_interval.py row (line 4885+23): renames and drops; nothing on bool/len.
* HANDOFF.md:123 Q21 (a)-(i): nothing on truthiness (so not even pending with the owner).
* references/m8-choices-2026-10-04/report.md: only `TruthSet.__bool__` (line 238) and `end or _end` (247).
* references/todo-from-v1-readme.md: no hit.
* README.md: bool only for relations / TruthSet (106-110, 255); time section silent.
* intervals/time_interval.py: `__bool__` (535) and `__len__` (539, "the number of pieces") have no docstring on
  truthiness; module docstring line 3 "every set operation, relation and arithmetic op is the numeric class's".
* tests/test_time_interval.py: no `bool(<empty time interval>)` / `assert not D()` pin (only bool(a < ...) at 474).

reasoning: the behaviour follows from the documented core rule plus the "thin wrapper" design, and it is the
defensible choice (set precedent, v1's own MultiInterval.__bool__), but no record names it for the time classes
or records the change from v1's always-True. A v1 `if spans:` on an empty time result now takes the other branch.
fix: one clause in plan §4's time_interval.py row (and/or the README time section) plus a pin.
