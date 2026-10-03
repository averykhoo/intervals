# recommendations on the open owner questions (2026-10-03)

the owner asked for each open question in `HANDOFF.md` to be thought through: every option, its pros
and cons, when each would be the better choice, and a recommendation. nine read-only subagents wrote
one report each (the files beside this one); each walks through every option and gives a confidence and
the cost of deciding later (before vs after 2.0). nothing was changed in the library. these are advice,
not decisions: a question closes when the owner answers it.

the probe scripts the reports cite were in `.scratch/owner-questions/` (gitignored, deleted with the
run); their outputs are transcribed in the reports. every number in them was measured 2026-10-03 on the
shared, loaded laptop.

## summary

"change" means the recommendation differs from what is built; "before 2.0" means deciding after the
release would change results or types under users.

| question | recommendation | change? | conf. | before 2.0? | report |
|---|---|---|---|---|---|
| Q20 crossed piece, exact class | keep `[1e-06, 1/1000000)`; one sentence in `v2-plan.md` "flags at rounded ends" | keep | 0.85 | yes | `q20.md` |
| Q19 outward isotone across types | keep; document "isotone within one grid, across grids within the double cover of f(B)"; add a public "every end onto doubles, outward" method | keep + add | 0.8 | yes | `q19.md` |
| Q17 pown of exact operands | (c): past a limit the tightest open float enclosure (nearest: rounded), one exact-result limit of about 2**22 bits shared by pown, pow and exp2/exp10; `EXACT_POWER_LIMIT` stays the float-corner threshold | change | 0.85 (limit 0.7) | yes | `pown.md` |
| m14b-open: repr past 4300 digits | decide the principle now (repr must not raise once Q17 makes such values cheap); spelling later, hex with `parse` reading `0x` | change, later | 0.65 | no | `pown.md` |
| Q18 correctly rounded nearest pown | yes, a promise: exact power rounded once, `rounded_pow` past the threshold; drop libm's `float ** int` | change | 0.85 | yes | `pown.md` |
| Q11(a)-(e) M15 | keep all; document `tol` absolute and `max_steps` counting boxes; `rtol` additive later | keep | med-high | (d) yes | `solver.md` |
| Q12(a)-(d) M16a | keep all; if an interval linear `solve(A, b)` is ever wanted, rename the nonlinear one `roots` now | keep | high | (b) yes | `solver.md` |
| Q13(a), (c)-(e) 1788 layer; Q9; Q10 | keep; close Q9 and Q10 as built; pay the two owed adapter clauses | keep | 0.65-0.9 | (a) yes | `ieee1788.md` |
| Q13(b) empty-set numbers in the layer | return `nan` in the layer only (library keeps D9) | change | 0.7 | yes | `ieee1788.md` |
| layer-numpy (open item 6) | keep `__array_ufunc__ = None` for 2.0, one README line | keep | 0.65 | no | `ieee1788.md` |
| the four unasked 1788 departures | confirm all four in one new D row; delete the stale "domain-clipped functions" from `REASONS` and `v2-plan.md` | keep + tidy | 0.85-0.9 | no | `ieee1788.md` |
| Q14(a)-(e) allen | keep all; docstring: the set is extensional; one rule for asserting normalized operands | keep | 0.6-0.9 | (c) yes | `allen.md` |
| Q15(a), (b), (d), (e) numpy | keep | keep | med-high | (b) yes | `numpy.md` |
| Q15(c) `==` against an ndarray | elementwise, as pandas, `np.isin` and every other element type | change | med-high | yes | `numpy.md` |
| Q15(f) `fmin`/`fmax` | map to `minimum`/`maximum` | change | medium | no | `numpy.md` |
| Q15(g) numpy in `[test]` | add it; the README numpy section as doctests | change | med-high | no | `numpy.md` |
| Q15(h) both operands ours | keep the hook's rule and fix the methods (a defect, below) | fix | high | yes | `numpy.md` |
| Q16(a)-(c), (f) backend | keep opt-in, surface as built, non-dyadic pure, ship in 2.0 | keep | 0.7-0.85 | no | `backend.md` |
| Q16(d) pins | pin `[fast]` to `<3` as well, and check it in the pin test | change | 0.7 | no | `backend.md` |
| Q16(e) CI | one gate job with `INTERVALS_BACKEND=gmpy2` and a ledger phase; no gmpy2 fuzz | change | 0.75 | no | `backend.md` |
| vectors-ext (b) rootn/pown/fma | skip glibc's rows (conformance inputs, not hard cases; licence is not the blocker); if Q18 is taken, CORE-MATH's pow rows with integral exponents are a ready pown set | close | high | no | `misc.md`, `pown.md` |
| Q6-rest | port neither `random_multi_interval` nor a public `apply()` for 2.0 | close | high / med-high | no | `misc.md` |
| Q6-shift semantics | `A << n` is `A * 2**n`, `A >> n` is `A * 2**-n`, exact; negative n allowed; `//` stays the floor | decide | med-high | yes | `misc.md` |

## checked first-hand by the session (2026-10-03)

delegation does not transfer judgement; these claims were re-run before this file was written:

* Q20: `MultiInterval(1 - Fraction(1, 10**30), 1.0) * Fraction(1, 3)` is
  `[0.3333333333333333, 333333333333333333333333333333/1000000000000000000000000000000]`, a float low
  end beside an exact high end from plain `*`; `M(Fraction(1, 10**30), 1.0000000000000003e-30).rootn(5)`
  is `[1e-06, 1/1000000]`
* Q15(h), a defect: `M(0.1) + O(0.1)` is an `OutwardMultiInterval`, but `M(0.1).hypot(O(0.1))`,
  `.minimum` and `.union` return `MultiInterval`; `M(0.1).hypot(O(0.1))` is the point
  `[0.1414213562373095]` while `np.hypot(M(0.1), O(0.1))` is `(0.1414213562373095, 0.14142135623730953)`.
  README "rounding" says "mixing the two gives an `OutwardMultiInterval`"
* Q15(c): `np.array([A]) == A` is `False`; `np.array([A]) == np.array([A])` is `[True]`
* Q18: README's elementary bullet says "correctly rounded by a pure-python evaluator (no libm)"; python's
  `float ** int` mis-rounded 8 of 20000 random `x ** n` (x in [0.5, 2], n in [2, 60]) against
  `float(Fraction(x) ** n)`, e.g. `1.3811118839148833 ** 26` is `4425.378458811314`, correctly rounded
  `4425.378458811315`
* Q13(c): `reverse.mul_rev(M(1, inf), M(inf))` is `(0, inf]`, `M(inf) / M(1, inf)` is `[inf]`
* Q16(d): `pyproject.toml` has `fast = ["gmpy2>=2.3"]`, unpinned above
* Q11(e): `newton(lambda x: x**2 - 1e-40, M(-1, 1))` ends in two unproved roots
  `(-2.3388402937128627e-14, -5e-41]` and `[5e-41, 2.3388402937128627e-14)`
* the 1788 tidy: `tests/itf1788/test_itf1788.py::REASONS` still lists `'domain-clipped functions'`
* misc: the repo has no `LICENSE`; `tests/itf1788/COPYING.LESSER` sits beside the vendored LGPL files (D15)
* found on the way: `tools/backend_speed.py`'s docstring cites `h3-records/gmpy2.md`, not in the tree

not re-checked: the agents' other probe numbers (Q19's 59% of mixed pairs failing for `%`, the speed
ratios of the backend, the solver's split counts, pown's 23% of CORE-MATH integer-exponent rows).

## how the reports fit together

* Q19 and Q20 are one tension (each end typed on its own); both reports land on keep/keep, and both say
  the alternative, one type per piece, is only right if Q19 is answered "make it isotone", and then as one
  change with Q20
* Q17 (c) and Q18 are one build in `ops._power_descriptor`; the pown report asks that `ops._NotADouble`'s
  proof be re-derived first-hand for general fractions before building it
* `misc.md` says pown vectors would flag non-bugs because nearest pown is libm by design; that holds only
  while Q18 is unanswered. if Q18 is taken, pown gets CORE-MATH's integral-exponent pow rows
* the numpy report's side finding (`repr(M(10**5000))` raises) is the same 4300-digit item as m14b-open
* the misc report's `<<`/`>>` route through `mul`, so they inherit Q15(h)'s fix if the methods are fixed
