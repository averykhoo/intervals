---
type: Decision
id: D-82
title: M8's review round (three reviews, one fixer)
actor: unrecorded
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:fb300f60ec570913835d9c7e50a73d4fedc7400bdd5c759414db8f7ef6142d61
---

### 2026-10-04 revision: M8's review round (three reviews, one fixer)

three read-only reviews of the build (soundness and D30, spec and API, sabotage); the session decided the
fixes, a fixer built them on branch `m8` (the record, with each finding's pin and the sabotage re-run,
`v2-implementation-plan.md` §2 M8 "review round"). behaviour changed, all in "the time layer (M8)" above:
* numpy's `timedelta64` (a `numbers.Integral` to numpy) and `datetime64` are no numbers: refused
  (TypeError) by the numeric class wherever it takes a number or an int argument, and by every scalar
  slot of the time layer (`MultiInterval(np.timedelta64(3, 'ns'))` was `[3]`, `td_iv * np.timedelta64(3,
  'ns')` was `td_iv * 3`)
* the order of two bounds is checked on their readings by `MultiInterval`'s rule, not on the values as
  written; so `D(tue, mon)` (from Tuesday through Monday) is now empty, not a `ValueError`, and
  `D(mon, noon_mon, start_closed=False)` a `ValueError`, not empty
* `degenerate_points` is a tuple sorted by instant (a set lost one instant of a DST fold)
* aware read-outs no longer pass through the UTC datetime (near the range's ends they overflowed) and
  read a tzinfo whose `dst()` is None
* `NaT` is a `ValueError` in every arithmetic operator; `to_pandas()` past pandas' range a `ValueError`
* documented, not changed: aware `dt + td` adds elapsed time; `pd.Timedelta % A` is pandas' own, wider
  remainder (no hook of pandas 3 lets the interval answer)
