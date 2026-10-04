# r2-18 reproduce: open/closed flag type checking (claimed UNDOCUMENTED_DIFFERENCE)

probes: .scratch/v1-parity/verify/r2-18/probe_flags.py (out_flags.txt), probe_mi_agree.py (out_mi_agree.txt), probe_none_point.py
command: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/r2-18/<probe>.py (from repo root)

## reproduced
* v1 `interval.Interval(0, f, 1, True)` raises TypeError for every non-bool f (incl. np.bool_, 0/1): archive/v1/interval.py:61-64.
* v2 `MultiInterval(0, 1, start_closed='no')` is `[0, 1]`, `0 in it` True; from_pieces, kernel.Builder.add_piece,
  OutwardMultiInterval, DateTimeInterval all read flags by truthiness. no v2 spelling raises TypeError for a non-bool flag.

## why it is not a v1->v2 gap
* the type check lived ONLY in v1's `interval.Interval` dataclass. v1's own main class `multi_interval.MultiInterval`
  (archive/v1/multi_interval.py:135-200, annotated `Optional[bool]`) never checked flag types and read them by
  truthiness exactly like v2: `v1.MultiInterval(0, 1, start_closed='no').endpoints == [(0,0),(1,0)]` (closed), same as v2.
* seeded sweep, 400 ranged cases over 16 flag values (str, None, 0/1/2, [], [0], np.bool_, 0.0, Fraction(0), bools):
  v1 MultiInterval vs v2 MultiInterval disagree 0. sabotage (expect start_closed='no' -> open) caught: True.
* `interval.Interval` as a whole is documented dropped: v2-implementation-plan.md:4906 surface map
  "| `interval.py` (`Interval`, `MultipleInterval`) | archived in `archive/v1/`; `tests/oracles.py` does its job |".
* workaround for the validation: caller-side `isinstance(flag, bool)`; nothing in the library.

## residual (side note, not this claim's gap)
* v2-plan.md:622 says exceptions are "for malformed construction (`[2,1]`, `nan`, bad types)", yet non-bool flags are
  accepted silently: an internal v2 doc-vs-code looseness, shared with v1 MultiInterval.
* point constructor compares flags with `!=` (intervals/multi_interval.py:100) but builds by truthiness: with v1's
  annotated `Optional[bool]`, `MI(5, start_closed=None, end_closed=False)` -> v2 ValueError 'half-open degenerate',
  v1 MultiInterval -> [] (empty). `MI(5, start_closed='a', end_closed='b')` v2 ValueError, v1 [5]. only non-bool
  inputs; with bools v1 and v2 agree everywhere probed.

## verdict
refuted. corrected: DROPPED_DOCUMENTED (the checking class `interval.Interval` is archived per the surface map);
for the MultiInterval constructor the flag semantics are EQUAL to v1's.
