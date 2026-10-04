# verify readme-1 (docs lens): v1 README "notes:" prose + "Geminis feedback" critique

claim: UNCLEAR (prose has no recorded disposition; owner to decide copy-to-references vs superseded before H4).

verdict: refuted as a capability gap. corrected_verdict: DIFFERS_DOCUMENTED (prose superseded; every
actionable point has its own record; the reciprocal point is a v1 bug fixed). residual: a
housekeeping question only (whether to keep the prose text), adjacent to H5/H4, not a v1 capability.

## each part of the prose, its record

| prose item | record | v2 run |
|---|---|---|
| notes: mod by perspective transform; "in practice only need 2 points" | references/modulo-derivations/claude-fable/proof-two-edge-reduction.md:11 "Every value of x mod y on a rectangle is attained on its top or right edge" (the rectangle geometry, reduced to far edges); intervals/modulo.py:17 "the algorithm is the far-edge one derived in references/modulo-derivations/claude-fable/"; references/gemini-conversation-recap.md:89 "Do not use geometric projection. Use **Quotient Analysis**." | `[12,18.7] % 7.5` = `{[0, 3.7], [4.5, 7.5)}` |
| critique 2: reciprocal returns whole line when 0 in A | v2-plan.md:37-42 "reciprocal splits at zero into sign-pure ... `1/[-1, 1]` = `[-inf, -1] ∪ [1, inf]`"; v2-plan.md:2460 "v1's reciprocal returns the whole line for anything touching zero → fix by splitting at zero" | v1 `(-inf, inf)`; v2 `{[-inf,-1/2],[1/2,inf]}`, 1/4 excluded (V1_BUG_FIXED) |
| critique 3: `__modulo` sweep is fragile | references/modulo-derivations/claude-fable/v3-modulo-design-notes.md:18-19 "`__modulo` ... the ~270-line geometric sweep engine is **dead code** ... and **substantially broken**" | as row 1 |
| critique 4A: `or` trick on epsilons | v2-plan.md (generic applicator) "v1's epsilon propagation through corners is UNSOUND, not just inelegant: `[0,1] * (2,3)` ..." (~line 2674) | not re-run (other slices) |
| critique 4B: regex compiled in `merge` | v2-plan.md:2683 "parsing moves out of `merge()` into `fmt.py`, regexes compiled at module level (hot-path compile in v1)"; v2-plan.md:1305 | `intervals/fmt.py::_TOKEN` is an `re.Pattern` |
| critique 4C: complex roots of negatives | v2-implementation-plan.md:29 D11 "negative bases are dropped with `DomainClippedWarning`" (real-only; the word complex is not used) | `[-4,4] ** 0.5` = `[0, 2]` + DomainClippedWarning |
| prose as a whole (keep or not) | adjacent only: v2-implementation-plan.md:752 (M11 backlog swept "the old README ... v1's public surface") and :779 "**smaller** (c): the old README's leftovers, `HANDOFF.md` H5"; v2-plan.md:1956 "**H5**: the v1 README's reading list and illustration to-do are kept"; references/todo-from-v1-readme.md:3 "the two items of `archive/v1/README.md` "TODO" that v2 has not done". none names the notes or the critique; the git history keeps the file after H4 anyway | - |

## probe
file: .scratch/v1-parity/verify/readme_1_docs_probe.py
cmd: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/readme_1_docs_probe.py
key output:
    v1 1/[-2,2] = (-inf, inf)
    v2 1/[-2,2] = MultiInterval.parse('{ [-inf, -1/2] , [1/2, inf] }')
    FAIL SABOTAGE expect v2 contains 0.25 (should FAIL)   <- the probe can fail
    v2 [12,18.7] % 7.5 = MultiInterval.parse('{ [0.0, 3.6999999999999993] , [4.5, 7.5) }')
    PASS fmt._TOKEN is compiled at module level
    v2 [-4,4] ** 0.5 = MultiInterval.parse('[0.0, 2.0]') ['DomainClippedWarning']
(the probe's last line wording is inverted: ok is False only because of the sabotage row; every real check passed.)

## not awaiting Q21
nothing here is an M8 (time layer) choice; Q21 does not cover it.
