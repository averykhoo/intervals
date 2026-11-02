---
type: Decision
id: D-78
title: owner answers to Q9-Q20 and the owner's-call items
actor: unrecorded
decided: 2026-10-03
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:e2993e0980aa116513ebafe4afb8bf24e8236137b77407999a843cfb11d2ae68
---

### 2026-10-03 revision: owner answers to Q9-Q20 and the owner's-call items

the owner accepted every recommendation of `references/owner-questions-2026-10-03/` on 2026-10-03
("i'll accept everything fable said"): nine reports, one per group of questions, each option's pros,
cons and when it is the better choice (`README.md` there is the table). where a report said
"consider", "optional" or "later if asked", nothing was decided beyond that. in short:
* **kept as built, now owner-confirmed**: D19 (Q11, M15) and D20 (Q12, M16a) whole; D21 (Q13, the
  1788 layer) but (b); D22 (Q14, allen) whole; D23 (Q15, numpy) (a), (b), (d), (e); D24 (Q16, the
  backend) (a)-(c), (f). Q9 and Q10 close as built in the 1788 layer (`ieee1788.mul_rev_to_pair`; the
  constructors' binary64 pass); `layer-numpy`: the layer keeps `__array_ufunc__ = None`
* **Q20 (D29)**: a crossed piece in the exact class stays the piece between the two values, each end
  keeping its flag (`[1e-06, 1/1000000)`), as plain `*` already gives a float end beside an exact one
* **Q19 (D29)**: the outward class stays typed per end, so it is isotone within one grid and, across
  grids, `f(A)` lies within the tightest double cover of `f(B)`; documented, and a public method puts
  every end on the double grid, outward, for a user who needs isotonicity. rejected: doubles only in
  the outward class (drops exact points such as the solver's `1/6`), and rounding whenever a float is
  mixed in (not isotone either: a purely exact `B` stays exact)
* **Q17 and Q18 (D28)**: pown of exact operands past one exact-result limit of about 2 ** 22 bits,
  shared by pown, pow and exp2/exp10, is the tightest open float enclosure (outward) or the value
  rounded to nearest, with a default-ignored warning; `EXACT_POWER_LIMIT` stays the float-corner
  threshold. pown to nearest is correctly rounded for every n (libm's `float ** int` was off by an ulp
  on this laptop; it was the one libm value in the library). `repr` must not raise past python's
  4300-digit limit: hex past it, which `parse` reads
* **Q13(b)**: the 1788 layer's numbers of the empty set are `nan`, 1788's answer; the library keeps
  D9's `ValueError`
* **Q15(c), (f), (g), (h)**: `==`/`!=` against an ndarray is elementwise; `fmin`/`fmax` are
  `minimum`/`maximum`; numpy joins `[test]` and the README numpy section becomes doctests; a method
  mixing the two classes returns the class the operators do (a defect: `M(0.1).hypot(O(0.1))` was a
  `MultiInterval`, against README "rounding")
* **Q16(d), (e)**: `[fast]` pinned to the window `auto` takes, as `[test]`; one CI gate job on the
  forced gmpy2 backend, no gmpy2 fuzz
* **the four 1788 departures never asked (D27)**: confirmed deliberate; the stale "domain-clipped
  functions" category goes
* **Q6-shift**: `A << n` is `A * 2 ** n` and `A >> n` is `A * 2 ** -n`, exact, through `*`; negative n
  allowed; `//` stays the floor. **Q6-rest closed**: neither `random_multi_interval` nor a public
  `apply()` in 2.0. **vectors-ext (b) closed**: glibc's rows are conformance inputs, not hard cases;
  pown takes CORE-MATH's pow rows with integral exponents instead
* small, from the same reports: `allen_relations`' set is documented as extensional and every
  cut-tuple relation asserts normalized operands; the solver documents `tol` as absolute and
  `max_steps` as boxes, splits symmetrically inside (0, 1], and `newton`'s width test cannot overflow
  (open item `newton-width`)
* left open, as the reports said: `Root`/`RootBox` as dataclasses if a third state appears; an `rtol`;
  the outward fma, `%`, hypot and `cancel_minus` per corner (tighter, optional); rootn run on cbrt's
  worst-case inputs (optional); a public strategies module after 2.0 if asked
