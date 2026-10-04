# verify timedelta-5: T(start_closed=False) with no bounds (lens: reproduce and find a way)

claim: v1 raises ValueError, v2 returns the empty set; UNDOCUMENTED_DIFFERENCE.

## reproduction (own probe)

probe: `.scratch/v1-parity/verify/timedelta5_probe_empty_flags.py`
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/timedelta5_probe_empty_flags.py`

all four flag combos, no bounds, for v1/v2 MultiInterval, TimeDeltaInterval, DateTimeInterval:

| flags (start, end) | v1 (MI, TD, DT) | v2 (MI, TD, DT) |
|---|---|---|
| True, True   | empty | empty |
| True, False  | ValueError (no message) | empty |
| False, True  | ValueError (no message) | empty |
| False, False | empty | empty |

key line: `mismatching flag combos (v1 TD vs v2 TD): 2` (the probe's "v1 always equals v2" expectation is caught).
`T() == T(start_closed=False) == T(start_closed=False, end_closed=False)` is True in v2.
the neighbouring checks agree: half-open point `T(h, start_closed=False)` raises in both; end without start raises in both.

## code

* v1: `archive/v1/multi_interval.py:151-155` (`if start is None: ... if start_closed != end_closed: raise ValueError`);
  v1 `time_interval.py:441` passes the flags straight through to it, so both v1 time wrappers inherit it.
* v2: `intervals/multi_interval.py::MultiInterval.__init__` and `intervals/time_interval.py::TimeDeltaInterval.__init__`
  (and `DateTimeInterval.__init__`): `if start is None:` builds the empty set and never reads the flags.

## finding a way / who is right

* every input v1 accepted gives the same set in v2 (the empty set); no v1 result is lost. the only v1 behaviour v2
  lacks is *rejecting* unequal flags on an empty constructor. nothing to compose: v2 cannot be made to raise there.
* v1 is not wrong in the soundness sense, but its check is incoherent: the empty set has no ends, so flags mean nothing
  for it either way, yet v1 accepts (False, False) and refuses (True, False). v2's "ignore flags when there is no
  start" is the consistent reading. neither is unsound.
* docs: no line in v2-implementation-plan.md §4, v2-plan.md, README.md, HANDOFF.md, references/todo-from-v1-readme.md
  records it (grepped "flags", "start_closed", "MultiInterval()", constructor rows). the v2 docstring says only
  "`MultiInterval()` is empty".

## verdict

kept as UNDOCUMENTED_DIFFERENCE, sharpened: not timedelta-specific (v2 MultiInterval, DateTimeInterval identical);
input-validation only, no capability or set lost; cosmetic. minimal repro:
`MultiInterval(start_closed=True, end_closed=False)` -> v1 ValueError(''), v2 empty.
