---
type: Decision
id: D-80
title: the time layer's choices (M8, D30)
actor: unrecorded
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:7ec303563f224b80ce1fce69e04bda8c736274dccbbcb7ac3aa5658e0ef5dff1
---

### 2026-10-04 revision: the time layer's choices (M8, D30)

the owner accepted the recommendations of `references/m8-choices-2026-10-04/` (a read-only agent's report;
the session re-ran its claims). to be built in M8; the design goes into "current design" when it is:
* **time as exact seconds, wall clock for naive**: `DateTimeInterval` and `TimeDeltaInterval` wrap a numeric
  `MultiInterval` of exact Fraction seconds (D4's (a)). a naive datetime counts from naive 1970-01-01 by
  subtraction, never `timestamp()` (machine-dependent, DST-dependent, and `OSError` on windows before
  1970-01-02 and for `datetime.min`/`max`); this is pandas' reading of a naive `Timestamp`. an aware
  datetime is its UTC instant; naive and aware do not mix (`TypeError`, as python and pandas); aware ends
  in different zones do, the left operand's zone kept for display, never part of `==`
* **infinite ends**: two sentinels, ordered below / above every datetime, date, timedelta and pandas type
  and taken back by the constructors, are what an infinite end reads out as; the numeric value underneath
  is ±inf, closed or open as written, as everywhere in v2. `math.inf` does not order against a datetime
  and `datetime.max` is a finite instant, so neither can stand in
* **no end-of-day snap**: a `date` is the half-open day `[d 00:00, d+1 00:00)`, its flag saying whether
  the day is in; a datetime is an exact instant (v1 also stretched `10:00` to `10:59:59.999999`). with
  exact seconds the snap leaves a gap between adjacent days, misses `23:59:59.9999995` and makes a day
  shorter than 86400 s; the half-open day tiles, as `[0, 1) | [1, 2)` does
* with them: an end that is no whole number of microseconds raises when read out as a datetime (a raw
  Fraction accessor beside it; never rounded silently); comparisons return `TruthSet`; `==` with a
  foreign type is `NotImplemented`; the wrappers are immutable and hashable; `NaT` and nan are refused
