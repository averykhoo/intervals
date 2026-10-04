# verify readme-1 (reproduce and find a way): v1 README "notes:" + "Geminis feedback"

claim under test: UNCLEAR gap, "v1 README prose (mod by perspective transform, Gemini critique) not kept in v2".

verdict: REFUTED as a v1->v2 capability gap. corrected verdict: V1_BUG_FIXED (the critique's behavioural items are
v1 bugs v2 fixes) with the modulo and regex items EQUIVALENT_RENAMED. the prose is design commentary, not a v1
capability: v1 never implemented a perspective-transform mod (its `__modulo` is the case sweep the critique attacks).
what is left is an archival question (keep the prose for H4 or not), not a parity gap.

probe: `.scratch/v1-parity/verify/readme-1/probe_readme_notes.py`
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/readme-1/probe_readme_notes.py`

| item (archive/v1/README.md) | v1 | v2 | result |
|---|---|---|---|
| reciprocal of [-2,2] (critique #2) | `(-inf, inf)`; says 0, 1/4, 49/100 in it | `1 / MultiInterval(-2,2)` = `{ [-inf, -1/2] , [1/2, inf] }` | V1_BUG_FIXED: x in 1/A iff |x| >= 1/2 (exact Fraction check, 9 points, v2 all right, v1 wrong on 0, +-1/4, 49/100). doc v2-plan.md:37 "reciprocal splits at zero into sign-pure pieces" |
| regex in merge (critique 4B) | compiled inside `merge` (multi_interval.py:307-309) | `intervals/fmt.py::_TOKEN` module-level; `MultiInterval.parse('[1,2]')` -> `[1, 2]` | EQUIVALENT_RENAMED; doc v2-plan.md:2683 "parsing moves out of `merge()` into `fmt.py`, regexes compiled at module level" |
| modulo (notes + critique #3) | `%` | `A % m` (far-edge, intervals/modulo.py:17) | 300 random (lo,hi,m) Fraction cases x 10 sample y = 3000 points: v2 `%`, Gemini's slice-and-shift built from v2 `&`/`-`/`|`, and v1 `%` all agree with brute membership. sabotage (`A % (m+1)`, probe_sabotage_mod.py) -> 426 mismatches, FAIL: probe can fail |
| or-trick (critique 4A) | `[0,1]*(2,3)` = `(0, 3)` | `[0, 3)` | V1_BUG_FIXED: 0 = 0*2.5 is attained. doc v2-plan.md:2674 "v1's epsilon propagation through corners is UNSOUND" |
| complex / negative-base pow (critique 4C) | `[-2,-1]**0.5` TypeError | `[-2,-1] ** Fraction(1,2)` = `{}` (1788 pow, outside domain) | no complex in either; v2 real-only (D11) |
| "perspective transform" mod note | prose only, never code | no tracked mention (git grep trapezoid/projective/perspective: nothing outside archive/v1) | not a capability; the idea's goal (correct mod with closure) is met by the far-edge algorithm |

sabotage line in the probe (`0 in 1/[-2,2]` expected True) prints FAIL, as intended: `fails 1 (1 expected)`.

residual (not a parity gap): H5 (v2-plan.md:1956) keeps only the reading list and the illustration to-do; nothing
records that the "notes:" prose and the Gemini critique are superseded. if the owner wants a trail before H4 deletes
archive/v1/, one line in references/todo-from-v1-readme.md or the H4 row would do.
