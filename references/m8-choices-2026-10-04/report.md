# M8 three choices — advisory report (fable, 2026-10-04)

Read-only advisory. Probes under `.scratch/m8-choices/`. Status: COMPLETE (2026-10-04). No tracked file
was touched.

**Summary.** (1) Read naive datetimes as wall-clock seconds by subtraction from a naive epoch, aware ones
as UTC seconds; mixing naive and aware raises; different zones among aware ends are fine, one zone kept
for display. (2) An infinite end reads as one of two sentinel objects that order below/above every
datetime, timedelta and Timestamp and are accepted back by the constructor. (3) Drop the snap: a `date`
is the half-open day `[d 00:00, d+1 00:00)`, the flag says whether the day is in; a datetime is an exact
instant.

## 1. Timezones

**Recommendation.** Read a naive datetime as wall-clock time: exact `Fraction` seconds since the naive
epoch `1970-01-01T00:00` by *subtraction* (`d - datetime(1970, 1, 1)` -> `days*86400 + seconds +
microseconds/10**6`), never `timestamp()`. Read an aware datetime as exact UTC seconds the same way
(`d - datetime(1970, 1, 1, tzinfo=utc)`). An interval is either all-naive or all-aware; mixing the two
in one constructor, one set op, one comparison or one arithmetic op raises `TypeError` (`==`/`!=`
return False/True, as python's and pandas' do). Aware datetimes in *different* zones are allowed in
one interval and one op: the value is the instant (UTC seconds); the wrapper remembers one `tzinfo`
for display only (the first end's, or an explicit `tz=`), and gives it back on `inf`/`sup`/iteration.

**Why not v1's `timestamp()` (local-time reading).**
* It is machine-dependent: the same naive datetime is a different number on a laptop in another zone,
  so a printed or pickled interval means something else elsewhere. Round trip only worked on the same
  machine (`fromtimestamp` is local too).
* On this Windows box it does not even cover the epoch. Probe (`probe_limits.py`, 2026-10-04, local zone
  "Malay Peninsula Standard Time", UTC+8): `datetime(1970,1,1).timestamp()`, `(1970,1,2)`, `(1900,1,1)`,
  `datetime.min`, `datetime.max` all raise `OSError [Errno 22]`; `datetime(3001,1,1)` works but
  `fromtimestamp(32536850400)` (3001-01-20) raises; `fromtimestamp(-1)` raises. So v1 could hold no date
  before 1970-01-02 08:00 local and nothing past year 3001; v1's own `infimum`/`__str__` call
  `fromtimestamp(float(...))` (`archive/v1/time_interval.py::DateTimeInterval.infimum`, `::__str__`).
* The subtraction reading is pure integer arithmetic and covers the whole `datetime` range exactly:
  probe `wall_seconds(datetime.min) = -62135596800`, `wall_seconds(datetime.max) =
  253402300799999999/1000000`, both round-trip `== d` through `epoch + timedelta(microseconds=int(s*10**6))`.
  v1 also lost sub-microsecond exactness to float (`float(self.interval.infimum)`); `Fraction` keeps it.
* v1 even disagrees with pandas on the same object: v1 turns a `pd.Timestamp` into a python datetime and
  then calls `timestamp()` (`::__init__` lines 73-80), i.e. local time, while pandas itself reads a naive
  `Timestamp` as UTC. Probe: `pd.Timestamp('2024-01-01').timestamp()` = 1704067200.0, python local
  `.timestamp()` = 1704038400.0 (8 h apart), wall-clock seconds = 1704067200.0. **The wall-clock reading
  gives exactly pandas' numbers** (`Fraction(ts.value, 10**9)` for a naive Timestamp; for an aware one
  `.value` is already UTC ns: probe `Timestamp('08:00', tz='Asia/Singapore').value == Timestamp('00:00',
  tz='UTC').value` is True). So the M8 pandas round-trips are exact and need no zone logic.

**DST folds and gaps.** With a wall-clock reading every naive day is exactly 86400 s and every wall time is
one point; the physical fold/gap is invisible. Under v1's local reading in a DST zone a naive spring-forward
day was 82800 s and a fold day 90000 s, and two naive readings of the same wall time differed by an hour
(probe, America/New_York: `1:30 fold=0` -> ts 1730611800, `fold=1` -> 1730615400; the 1:30->3:30 gap pair
is 2 h apart by wall clock, 1 h by UTC). The owner's machine has no DST, which is why this never showed
and why it is dangerous: it is a hidden dependency. Does it matter for an interval library? Only for
users who want *physical* durations across a transition, and they have the aware path, where UTC seconds
give the physics (NY 1:30->3:30 on gap day = 1 h). Wall-clock is what calendar/availability use (the
`references/owner-questions-2026-10-03/allen.md` domain for `DateTimeInterval`) expects: "Monday 9-17"
is 8 h whatever the clocks did.

**Mixing aware and naive raises.** Both references do: python `naive < aware` and `naive - aware` raise
`TypeError` (`==` is False); pandas `Timestamp` the same (`Cannot compare tz-naive and tz-aware
timestamps`), and `pd.Interval(naive, aware)` raises (probe `probe_pandas.py`). Treating naive as UTC
silently would make `DateTimeInterval(naive) == DateTimeInterval(aware_utc)` True where pandas says
False. v2's own style is the loud TypeError for a foreign type (`intervals/multi_interval.py::MultiInterval.__contains__`,
`::_coerce` returns NotImplemented; probe: `M(datetime(...))` raises TypeError). The empty interval has
no zone and combines with either, as `EMPTY` does with any class.

**Different zones among aware ends.** Allow, normalize to the instant. python compares and subtracts
across zones freely; pandas compares across zones (`UTC 02:30 == SGT 10:30` True, equal hashes) but
`pd.Interval` refuses two zones in one interval (`ValueError: left and right must have the same time
zone`). pandas' rule is about its dtype (`datetime64[ns, tz]` carries one zone per column), not about the
set of instants; `[09:00 Singapore, 09:00 New York]` is a legitimate set of instants. The alternative
(pandas-strict, raise on two zones) loses that for no gain, since the zone is display only. Round-trip to
`pd.Interval` then converts both ends to the display zone (`ts.tz_convert(tz)`), which pandas accepts.

**Alternative that loses:** keep `timestamp()` but with `fold` handling. Still machine-dependent, still
OS-range-bound on Windows, still disagrees with pandas on naive `Timestamp`s.

**Sub-decisions left to the owner.**
* The display zone's rule: the first non-empty operand's `tzinfo` (left operand in a binary op), or
  always UTC for a mixed-zone result. I suggest the former plus an `astimezone(tz)` method returning a
  copy with another display zone (cheap: the numbers do not change). Note `datetime.timezone.utc` and
  `ZoneInfo('UTC')` are different objects with the same offset; comparing display zones should not be
  part of `__eq__` (equality is of the set of instants, as pandas' `Timestamp.__eq__`).
* A `date` is naive. `DateTimeInterval(date(...))` next to aware ends raises as any naive would; a `tz=`
  kwarg on the constructor ("this date, in this zone") is the one escape hatch worth adding, since a
  date has no zone of its own. Owner's call whether to add it in M8 or later.
* `TimeDeltaInterval` has no zone; read `timedelta` exactly as `days*86400 + seconds + microseconds/10**6`,
  not `total_seconds()` (probe: `timedelta(days=10**6, microseconds=1).total_seconds()` = 86400000000.0,
  the microsecond lost); `pd.Timedelta` as `Fraction(td.value, 10**9)`.

**Interactions.** With Q3: under the wall-clock reading `date + 1 day` is always `start + 86400` exactly,
so a half-open day is clean; under v1's reading a DST day was 82800 or 90000 s. With Q2: infinite ends
have no zone; the display zone applies to finite ends only.

## 2. What an infinite end reads as

**Recommendation.** Two module-level sentinel objects (one pair serving both classes; names the owner's,
say `NEG_INF` / `POS_INF`), ordered below / above every datetime, date, `pd.Timestamp`, timedelta,
`pd.Timedelta` and each other, hashable, `repr` `-inf` / `inf`, accepted by the constructors and by
slicing, so `DateTimeInterval(NEG_INF, t).inf is NEG_INF` and `DateTimeInterval(iv.inf, iv.sup, ...)`
rebuilds a piece. The wrapper maps them to `±math.inf` on the way in and back on the way out; the numeric
class never sees them. Infinite ends are closed or open as the user wrote them, as v2 takes
`MultiInterval(1, inf)` literally, and a missing slice bound is the closed infinity, as
`MultiInterval.__getitem__` has it.

**Why.** v2 answers this for numbers by making ±inf *points of the domain*: `M(-inf, 5).inf` is `-inf`
(float), `inf_closed` True, `size` `Size(rays=1, length=5, points=1)`, `REALS` holds both (probe
`probe_v2.py`; `intervals/multi_interval.py::MultiInterval.inf`, `::is_finite`; `v2-plan.md` "domain and
semantics": "both infinities as points"). The time wrapper should do the same with a value of *its*
domain, and the test for a read-out is that it behaves as a datetime would where it is used: ordering
against datetimes, round trip through the constructor, hashing, repr. Probe `probe_inf.py` (2026-10-04)
on each candidate:

| candidate | `x > datetime` | `x > pd.Timestamp` | rebuilds the same interval | verdict |
|---|---|---|---|---|
| `math.inf` (as v2 returns) | TypeError | TypeError | only if the constructor takes floats | loses: a latent TypeError the day an interval is unbounded |
| `datetime.max` / `min` | True | True | **no**: a finite instant (`253402300799999999/1000000` s), so the rebuilt interval is bounded and `is_finite` flips | loses: silently changes the set |
| `pd.Timestamp.max` / `min` | True | True | no, and narrower than datetime (1677 / 2262); pandas 3 parses `'9999-12-31'` at unit `us`, so it is not even pandas' bound any more | loses |
| `pd.NaT` | False both ways (nan semantics; `NaT == NaT` False) | same | no (`pd.Interval(NaT, t)` raises) | loses: unordered |
| `None` | TypeError | TypeError | no (`None` is "no end" in v1's constructor) | loses |
| raise | n/a | n/a | n/a | loses: `[-inf, t].sup` must be checkable; forces every caller through `is_finite` first |
| sentinel pair | True | True (pandas returns NotImplemented, python asks the sentinel) | yes | **wins**; also orders against `date`, `timedelta`, `pd.Timedelta`, `NaT`, floats; `sorted([ts, POS, d, NEG])` and `max(d, POS)` work |

The sentinel is what D4 option (b) already envisaged ("two sentinel objects that compare below/above
everything", `v2-implementation-plan.md` line 22); choosing (a) for *storage* does not forbid them as
*read-outs*, and the read-out is the only place they are needed. One pair serves both classes: `-inf`
below every datetime and every timedelta is consistent, and it spares the user two names.

**Main alternative and why it loses.** Return `±math.inf` exactly as the numeric class does (no new
symbols, maximally thin). It fails ordering: `math.inf > datetime(...)` raises TypeError, so
`if iv.sup > deadline` works on every bounded interval and blows up on the first unbounded one, the case
the M8 spec's "`[-inf, t]` style open-ended ranges" tests exist for. A number is also the wrong type for
an accessor documented as returning a datetime.

**pandas round trip.** pandas has no infinite Timestamp (`pd.Timestamp(inf)` raises
`OutOfBoundsDatetime`; `pd.Interval(NaT, t)` raises). A `to_pandas()` of an unbounded interval should
raise `ValueError` by default; clamping to `Timestamp.min/max` changes the set and should be opt-in if
offered at all (probe: `pd.Interval(Timestamp(datetime.min), Timestamp(datetime.max))` works at unit
`us`, so a clamp could use datetime's bounds, not `Timestamp.min/max`).

**repr.** With a literal-based repr, `DateTimeInterval(-inf, datetime.datetime(2024, 1, 1, 0, 0))` falls
out of the sentinels' `repr` for free; with a text grammar (`DateTimeInterval.parse('[-inf, 2024-01-01T00:00:00]')`)
v2's `inf`/`infinity`/`∞` tokens already exist (`intervals/fmt.py::_NUMBER`).

**Sub-decisions left to the owner.**
* Names and home of the pair. The package exports `EMPTY` and `REALS` as module constants
  (`intervals/__init__.py`), so a module-level pair fits the house style.
* Whether to take a closed infinity literally in the time domain. I suggest yes, as v2 does, with no
  "time is always open at infinity" rule: thinness, and `size.rays` is the same either way. v1's slice
  opened the infinite bound (`archive/v1/time_interval.py::DateTimeInterval.__getitem__`,
  `start_closed=not math.isinf(start)`); v2's slice closes it and the wrapper should follow v2.
* The same accessor has a second hard case the probes surfaced: an end that is not a whole number of
  microseconds (a `pd.Timestamp` with nanoseconds, `TimeDeltaInterval / 3`, Q3's half microsecond)
  cannot become a `datetime` or `timedelta`: `timedelta(seconds=Fraction(1, 3))` and
  `timedelta(microseconds=Fraction(1, 2))` raise TypeError (probe), while `pd.Timestamp` takes
  nanoseconds. Options: round `inf`/`sup` to the microsecond (silent, lossy), raise, or return a
  `pd.Timestamp` when nanosecond-exact and raise otherwise; plus a raw accessor (`inf_seconds`, the
  Fraction) that never fails. I lean to raise, naming the raw accessor, never round silently (v2 never
  rounds an exact value unasked). See "Other decisions", item 1.

**Interactions.** With Q1: the sentinels carry no zone; only finite ends render in the display zone.
With Q3: none.

## 3. The end-of-day snap

**Recommendation.** Drop the snap. A `date` `d` denotes the day, the half-open set `[d 00:00, d+1 00:00)`,
and the closed/open flag says whether that day is in: as a closed end, `d` means "through the end of day
d" (`d+1 00:00`, open); as an open end, "before day d" (`d 00:00`, open); as a closed start, `d 00:00`
closed (unchanged from v1); as an open start, "after day d" (`d+1 00:00`, closed). A `datetime` (and a
`pd.Timestamp`, which is one) is an exact instant whatever its clock reads; the hour/minute/second/
microsecond snapping v1 applied to *datetime* ends goes too. Document both.

**Why.** Under exact `Fraction` seconds the snap is wrong in three measurable ways (probe `probe_v2.py`,
2026-10-04, on the v2 class with seconds as values):
* **adjacent days do not merge**: v1-style closed days `[0, 86399.999999] | [86400, 172799.999999]` stay
  `{ [0, 86399999999/1000000] , [86400, 172799999999/1000000] }`, two pieces, because the gap
  `(23:59:59.999999, 00:00:00)` is a non-empty set of exact seconds. Half-open days
  `[0, 86400) | [86400, 172800)` give `[0, 172800)`, one piece. With the snap `(mon | tue).is_contiguous`
  is False and `len(week)` is 7; every calendar use trips on it.
* **membership**: `23:59:59.9999995` (a Fraction, reachable from a `pd.Timestamp` with nanoseconds or from
  `TimeDeltaInterval` arithmetic) is `False` in the snapped day and `True` in the half-open one. It is in
  the day by any reading.
* **size**: a half-open day is `Size(rays=0, length=86400, points=0)`, `wid` 86400; a snapped day is
  `Size(0, Fraction(86399999999, 1000000), 1)`. Seven snapped days do not add up to a week
  (`v2-plan.md` "set operations and size": "tiling forces this convention: `[0,1) + [1,2)` must equal
  `[0,2)`" is the same argument).

v1's author saw it: `archive/v1/time_interval.py` lines 98-101 say the open-end flag "doesn't really
make sense with this method because that only excludes the last microsecond of the day" and wish
`[Tuesday to Saturday) == [Tuesday to Friday]`. The half-open rule makes exactly that true: `[Tue, Sat)`
is `[Tue 00:00, Sat 00:00)`, which is `[Tue, Fri]` closed. The snap existed because v1 stored float
seconds and had no clean "up to midnight, excluded"; v2 has open cuts (`intervals/cuts.py` module
docstring: `[1, 2) | [2, 3]` "tiles exactly because the end `(2, BELOW)` equals the start `(2, BELOW)`"),
so the reason is gone.

**A date in each position; the one-date interval.** `DateTimeInterval(d)` is the whole day
`[d 00:00, d+1 00:00)`, size 86400, `is_degenerate` False, as v1's full day was (v1 `_end` = end of day).
`date in iv` keeps v1's meaning through v2's subset alias
(`intervals/multi_interval.py::MultiInterval.__contains__`: a MultiInterval argument is subset): the
whole day is inside. A slice `iv[d1:d2]` with dates reads `[d1 00:00, d2+1 00:00)` (v1's slice already
took the stop date's supremum for this reason, `::__getitem__` line 237).

**A datetime at midnight is an instant.** v1 turned a datetime *end* of `2024-01-02 00:00` into
`2024-01-02 23:59:59.999999` (the whole next day) and a datetime end of `10:00` into `10:59:59.999999`
(lines 102-109; probe confirms). That is value-driven and surprising: the type says what the user meant.
The M8 spec's "keep the end-of-day snapping for `date` inputs" is about `date` only; I read the datetime
snaps as not covered and recommend dropping them whatever the owner decides for dates.

**Backward compatibility.** The *meaning* of `[d1, d2]` in dates is unchanged: all of both days are in,
so `datetime in iv` answers the same for every microsecond-exact datetime. Observable changes: (a) `sup`
of a date-ended interval is the next midnight with `sup_closed` False instead of `23:59:59.999999`
closed; (b) unions of adjacent days merge; (c) sizes are whole days; (d) a datetime end is no longer
stretched. (a) breaks a caller testing `supremum == 23:59:59.999999`; M8 breaks every v1 caller anyway
(`.interval.endpoints`, in-place mutation, `cardinality`; HANDOFF session log 2026-10-04), so this is the
moment. A `snap=True` shim is possible but I would not build it: it reintroduces the three bugs.

**Main alternative and why it loses.** Keep the snap "because v1 users expect 23:59:59.999999". They
expected it as a *display*, not a set boundary; under exact arithmetic the set it defines is wrong, and
the first user who unions two days finds out.

**Sub-decisions left to the owner.**
* `__str__` sugar: print a piece whose ends are both midnights as a date range `[d, e-1]` (v1's `__str__`
  printed `[2018-09-01]` for a full day), or always the literal `[2024-01-01 00:00, 2024-01-02 00:00)`.
  `repr` should be the literal, round-trippable form either way (as `MultiInterval.__repr__` is
  `.parse(...)` text); sugar, if any, belongs in `__str__`.
* A `date` beside aware ends (Q1): a date is naive; "this day in this zone" needs the `tz=` kwarg.

**Interactions.** With Q1: a day is exactly 86400 s only under the wall-clock reading; v1's local
reading would make the half-open day 82800 or 90000 s on a DST transition in a DST zone. With Q2: none.

## Other decisions found

Not among the three, but each must be settled before or during the build:

1. **A non-microsecond end** (see Q2's last sub-decision): `inf`/`sup`/`degenerate_points`/iteration
   must return a datetime for a Fraction that may not be µs-exact (`timedelta(seconds=Fraction(1, 3))`
   raises TypeError, probe `probe_inf.py`). Raise, round, or `pd.Timestamp` at ns; plus a raw
   Fraction accessor. Affects `total_duration` too (`timedelta(seconds=size.length)`).
2. **Comparison operators.** v1's `<` etc. returned `bool` (`archive/v1/multi_interval.py::__lt__` via
   `__compare`); v2's return a `TruthSet` (`intervals/relations.py::lt`, `TruthSet.__bool__` raises on
   `{T, F}` and `{}`). The thin wrapper should return v2's `TruthSet`; say so, since a v1 caller's
   `if a < b:` can now raise.
3. **`__eq__` with a foreign type.** v1 raised TypeError (`_datetime_interval(other)`); v2 returns
   NotImplemented, so `==` is False. Follow v2; also give the wrapper `__hash__` (v1 defined `__eq__`
   without one, so a v1 time interval was unhashable; v2 is immutable and hashable).
4. **`pd.NaT`/`nan` in the constructor.** v1 silently dropped them (`pd.isna(end)` -> `None`, so
   `DateTimeInterval(NaT, t)` became the point `t`, lines 59-63). v2 rejects nan with ValueError
   (`intervals/cuts.py::normalize_value`). Recommend raising.
5. **`end or _end`** (line 126) is a bug to not port: a falsy end (`Fraction(0)`, the epoch under the
   wall-clock reading) is replaced by `_end`. Use `is None`.
6. **Cross-type table gaps.** v1 had `TimeDeltaInterval * Real`, `/ Real`, no `TimeDeltaInterval /
   TimeDeltaInterval` (a dimensionless `MultiInterval`), no `Real * DateTimeInterval` (correctly a
   TypeError). Decide whether `td / td -> MultiInterval` and `td // td`, `td % td` join the table.
7. **Membership of a bare `datetime`** stays scalar (`contains_point`); of a `date`, subset of the day
   (Q3). Of a `pd.Timestamp` with nanoseconds: exact, so a ns-offset instant can be in or out of a
   µs-bounded interval; fine, but a doctest should show it.
8. **The time grammar** (`parse`/`repr`): whether `DateTimeInterval.parse` exists at all in M8 or the
   repr is the constructor call. The numeric class's repr is `.parse(...)`; a time grammar needs ISO
   8601 datetimes, dates, `inf`, and open/closed brackets, which is a day of work on its own.
9. **Display zone on `__eq__`**: equality is of the set of instants, not of the display zone (Q1), as
   pandas compares Timestamps across zones (probe: `UTC 02:30 == SGT 10:30` True, equal hashes).

## Probe files

`probe_pandas.py` (tz mixing, pandas limits, DST), `probe_limits.py` (Windows `timestamp()` range,
exact Fraction round trips, pandas naive-as-UTC), `probe_v2.py` (v2's infinite ends, adjacency,
foreign types; run with `PYTHONPATH` = repo root), `probe_inf.py` (read-out candidates, Fraction to
datetime). All run with `C:/Users/user/anaconda3/envs/intervals/python.exe` on 2026-10-04, pandas
3.0.6, numpy 2.5.2, python 3.13.15, local zone UTC+8 without DST.
