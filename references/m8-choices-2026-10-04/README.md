# M8's three choices: recommendations (2026-10-04)

the owner asked for an educated recommendation on M8's three open choices (timezones, what an infinite end
reads as, the end-of-day snap). a read-only fable agent wrote `report.md` from v1's `time_interval.py`, the v2
class and probes (`probes/probe_*.py`; pandas 3.0.6, python 3.13.15, local zone UTC+8 without DST). the
summary and what is left to decide are in `v2-implementation-plan.md` §2 "M8".

the session re-ran the claims the recommendations rest on, independently (`probes/verify_session.py`, run
with `PYTHONPATH=.`, 2026-10-04), and each held:

* `datetime.timestamp()` raises `OSError` on this laptop for 1970-01-01, 1970-01-02 00:00, 1900-01-01,
  `datetime.min` and `datetime.max`; `fromtimestamp(-1)` raises
* pandas reads a naive `Timestamp` as UTC (`1704067200.0` for 2024-01-01, python's local `timestamp()`
  `1704038400.0`); the wall-clock subtraction and `Fraction(ts.value, 10 ** 9)` both give `1704067200`;
  `datetime.min`/`max` read exactly
* naive against aware: python's and pandas' `<` raise `TypeError`, `==` is `False`; `pd.Interval` refuses a
  naive/aware pair and two zones
* `timedelta(days=10 ** 6, microseconds=1).total_seconds()` loses the microsecond
* `math.inf > datetime` and `math.inf > Timestamp` raise `TypeError`; `pd.Timestamp(inf)` raises; a
  sentinel object orders above `datetime`, `date` and `Timestamp`, and `sorted` takes it
* `timedelta(seconds=Fraction(1, 3))` and `timedelta(microseconds=Fraction(1, 2))` raise `TypeError`
* v2: `M(-inf, 5)` keeps `-inf` closed (`Size(rays=1, length=5, points=1)`); comparisons return `TruthSet`;
  a missing slice bound is closed; `nan` and a `datetime` are refused
* the snap on the v2 class: v1-style closed days union to two pieces, half-open days to `[0, 172800)`;
  `23:59:59.9999995` is out of the snapped day and in the half-open one; sizes as the report states
* v1's constructor snaps a datetime end to the end of its hour, minute or second
  (`archive/v1/time_interval.py::DateTimeInterval.__init__`) and has `end or _end`; its slice opens an
  infinite bound
