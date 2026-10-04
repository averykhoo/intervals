# verify r2-18 (docs lens): "open/closed flag type checking (v1 TypeError for non-bool flags)"

claim: v1 archive/v1/interval.py:61-64 (Interval.__post_init__) raises TypeError for non-bool flags; v2 reads
flags by truthiness; UNDOCUMENTED_DIFFERENCE.

## verdict: REFUTED as a gap. corrected verdict: DROPPED_DOCUMENTED (for interval.py's Interval); the
## MultiInterval constructor's flag handling is EQUAL to v1's.

1. the TypeError lives only in `archive/v1/interval.py::Interval` (a dataclass, positional
   `Interval(start, start_open, end, end_closed)`). that whole module is recorded as dropped from the API:
   v2-implementation-plan.md:4906 (§4 surface map):
   "| `interval.py` (`Interval`, `MultipleInterval`) | archived in `archive/v1/`; `tests/oracles.py` does its job |"
   v1's README never uses `Interval(`; its public class is `MultiInterval`.
2. v1's own `MultiInterval` (the class v2's `MultiInterval` replaces) reads flags by truthiness exactly as v2
   does: archive/v1/multi_interval.py:201-202 `_start = (start, 0 if start_closed else 1)`,
   `_end = (end, 0 if end_closed else -1)`; no isinstance check on the flags anywhere in its __init__ (136-202).
3. probe `.scratch/v1-parity/verify/probe_r2_18_flags.py`, run
   `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/probe_r2_18_flags.py`:
       'no'    v1.MultiInterval: ('ok', '[0, 1]')  v1i.Interval: ('TypeError', 'no')    v2: ('ok', '[0, 1]')
       None    v1.MultiInterval: ('ok', '(0, 1]')  v1i.Interval: ('TypeError', 'None')  v2: ('ok', '(0, 1]')
       0 / 1 / [] / 'False' / 'x': v1.MultiInterval == v2 on every row; only v1i.Interval raises
       sabotage caught (expecting TypeError from v1.MultiInterval fails)
4. docs searched for a record naming flag truthiness: none. v2-plan.md:621-622 "exceptions are for malformed
   construction (`[2,1]`, `nan`, bad types)" is ADJACENT only (does not name flags, and v2 does not apply it to
   flags). v2-implementation-plan.md:137 types the signature `start_closed=True, end_closed=True` (bool by
   annotation only). nothing in HANDOFF Q21, README, references/todo-from-v1-readme.md names it.

## residual (not a v1->v2 gap)
v2 inherits v1 MultiInterval's lax flag reading: `MI(0, 1, start_closed='no')` silently builds `[0, 1]`. that is
a possible hardening item for v2 (v2-plan.md:622's "bad types" spirit), but v1's MultiInterval behaved the same,
so it is not something v1 could do that v2 cannot.
