---
type: Decision
id: D-58
title: M13g, decorations, constructors and signals, done
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:8fddbac598d62deda2aa00083814e7888d26a25b9eb7b6a4a983beb68f38747e
---

### 2026-09-26 revision: M13g, decorations, constructors and signals, done

M13g is done on branch `m13g` (2026-09-26); its record is v2-implementation-plan.md M13g, "done",
and parts 1 to 3 below are its steps. **D16 is now current design** ("ieee 1788" above): the core
`MultiInterval` stays undecorated, and 1788's decorations live in the wrapper `DecoratedInterval`,
with `Decoration` com/dac/def/trv, 1788's constructors (`text_to_interval`, `nums_to_interval` and
their decorated twins, `set_dec`) and propagation through every point function and set operation of
the core. 1788's signals are python's: `UndefinedOperationError` (a `ValueError`) raises,
`PossiblyUndefinedOperationWarning` (an `IntervalWarning`) would warn and is never emitted, since
the exact parser decides validity. no NaI and no `ill` (Q8): every statement that needs one is a
row under "no NaI: invalid input raises". every itf1788 statement of an M13g op is a vector (matching,
or a row), every decorated vector runs through the decorated type with its decoration checked, and
the only statements still skipped are the 19 reverse ops' (M13e), pinned by
`tests/itf1788/test_itf1788.py::test_only_the_reverse_ops_are_skipped`. **still open for the owner**
(each built as the conservative reading, flagged in the part entries): the PROPOSED category "exact
parsing decides validity" (7 rows); whether the 15 rows where an exact value past the doubles keeps
com (`_BOUNDED_EXACTLY`, 3; `PLAIN_ONLY`, 12) stay under "decoration expectations" or get a category
of their own; and `set_dec` demoting as 1788's `setDec` does rather than raising. the reverse ops
join the decorated type with M13e, all trv (`decorated.py::_trivial`)
