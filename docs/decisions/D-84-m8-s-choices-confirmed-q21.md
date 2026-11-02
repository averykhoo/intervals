---
type: Decision
id: D-84
title: M8's choices confirmed (Q21)
actor: unrecorded
decided: 2026-10-05
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:d7e4c1e96b981cbb1020c32a6a44d5ddcf451d4fdf6dfe50a86ad7da547c27c9
---

### 2026-10-05 revision: M8's choices confirmed (Q21)

the owner accepted the session's recommendation on each choice the time layer's build made beyond D30
(`v2-implementation-plan.md` §2 M8, its done-record and review round), all as built, nothing changes:
* (a) the sentinels are `NEG_INF`/`POS_INF`; (b) `td / td` and `td // td` a `MultiInterval`, `td % td` a
  `TimeDeltaInterval` (python's `timedelta` types), `td // real` refused (`MultiInterval // n` floors to whole seconds,
  python's `timedelta // n` to microseconds); (c) `repr` the constructor call, no time `parse`; (d) `tz=` for dates
* (e) a float factor is its exact value (`td * 0.1` is exact and its read-out raises; python rounds to the
  microsecond): the only inexact step would be a silent rounding; `Fraction(1, 10)` reads out
* (f) bounds ordered on their readings by `MultiInterval`'s rule (`D(tue, mon)` empty, `D(wed, mon)` raises), one rule
  over a "looks reversed" special case
* (g) aware `dt - dt` and `dt + td` are elapsed time between instants, consistent with `==` and `<` (and with pandas'
  aware `Timestamp`), not python's same-tzinfo wall clock
* (h) `pd.Timedelta % A` and `divmod(pd.Timedelta, A)` sound but wider than exact, `TimeDeltaInterval(x) % A` the
  documented workaround (no pandas hook defers it); (i) `degenerate_points` a tuple (a DST fold's two instants compare
  equal)
