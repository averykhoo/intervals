# Q11 (D19, M15) and Q12 (D20, M16a): the solver stack's choices — advice for the owner

written 2026-10-03 by a read-only advisory agent. sources: `intervals/autodiff.py`, `intervals/solver.py`,
`intervals/__init__.py`; `v2-plan.md` "the solver stack", "2026-09-27 revision: M15", "2026-09-28 revision:
M16a"; `v2-implementation-plan.md` §0 D19, D20, §2 M15, M16a; `HANDOFF.md` Q11, Q12, open item 4
`newton-width`, "still owed" (M16a bullet). comparisons with other libraries are from the agent's
knowledge, flagged "(knowledge)"; anything measured by probe is flagged "(probe, 2026-10-03)" with its
script under `.scratch/owner-questions/probes-solver/`.

status: complete, 2026-10-03 (every section written; summary in §11).

## 0. what is built today (verified against the tree, 2026-10-03)

* `intervals/autodiff.py`: `Dual(value, derivative)` (two `MultiInterval`s or two `DecoratedInterval`s,
  `Dual.variable`, `Dual.constant`, 42 ops with chain rules), `derivative(f, x)`; below it (M16a)
  `gradient(f, xs)`, `jacobian(F, xs)` as n passes through the private `_passes`.
* `intervals/solver.py`: `Root(interval, unique)` NamedTuple, `newton(f, x, *, tol=1e-10, max_steps=10_000)`;
  below it (M16a) `RootBox(box, unique)` NamedTuple, `solve(F, xs, *, tol=1e-10, max_steps=10_000)`, which at
  `n == 1` delegates to `newton` and rewraps (`solver.py::solve`, the `if n == 1:` branch).
* `intervals/__init__.py`: the eight names imported and listed in `__all__` under two "M15"/"M16" comments;
  `tests/test_applicator.py::test_package_exports_unchanged` pins `__all__`.
* the step: `solver.py::_newton_step` is `piece & (point + mul_rev(slope, -value))`; uniqueness there needs a
  non-empty image inside `piece.interior` and `0 not in slope`.
* the C¹ gate: `solver.py::_evaluate` (`value.decoration >= DAC and slope.decoration >= DAC`), and for n
  variables `solver.py::_jacobian` on the closed hull.
* `tol` is used in two places per solver: the "newton goes on while it halves" test
  (`width > tol or past < _PAST_TOL`) and the output test (`piece.wid() <= tol` / `_width(box) <= tol`,
  `_width` = the widest component). it is an absolute width in the units of `x`. `max_steps` counts
  entries popped from the work stack (boxes), not calls of `f` (a box costs n + 2 calls plus up to n + 2
  when output unproved: `v2-plan.md` "the solver stack", known-limits bullet).
* `_PAST_TOL = 8`, `_SPAN = 16` (`solver.py` module constants); `_magnitude_split` returns `None` for any
  piece with `hi <= 1` in magnitude, i.e. a piece inside `[-1, 1]` is never split by exponent.
* README leads with `newton` and `solve` as the demonstration (README.md lines 69-82, the doctest block).
* tests: `tests/test_solver.py` (newton), `tests/test_solve.py` (54 tests, 2026-09-28), `tests/test_autodiff.py`,
  `tests/test_gradient.py`; `tests/test_solver.py::test_tol_stops_refining` and
  `tests/test_solve.py::test_constant_and_continuum_systems` pin `tol` as an absolute width on a continuum.

### how the comparable libraries name and shape this (knowledge, not verified against their current docs)

| library | global verified zeros | result | tolerance | derivative |
|---|---|---|---|---|
| Julia IntervalRootFinding.jl | `roots(f, X; contractor=Newton/Krawczyk/Bisection, abstol=1e-7, reltol=0)` (older releases: one `tol`, renamed) | `Root(interval, status)`, status `:unique` / `:unknown`; same `roots` and `Root` for 1-D and `IntervalBox` | absolute by default, relative optional | ForwardDiff automatically |
| INTLAB (MATLAB) | `verifynlss(f, xs)` verifies one zero near an approximation (Krawczyk + epsilon inflation); `verifynlssall(f, X)` finds all: `[X, XS] =` verified-unique boxes and possibly-containing boxes | two lists, exactly our unique / unproved split | absolute-ish, internal | gradient type (forward AD) |
| pyinterval (taschini) | `interval(...).newton(f, fprime, maxiter=...)` as a method on the interval | an interval (union of components) | none; runs to fixed point | caller supplies `fprime` |
| mpmath | `findroot(f, x0, solver=..., tol=...)`: approximate, one root, not verified; `polyroots` | an mpf | on the residual | finite differences / caller |
| scipy | `root_scalar(f, bracket, xtol, rtol)`, `brentq(xtol=2e-12, rtol=8.9e-16)`, `root(F, x0)` / `fsolve` | one approximate root | `xtol + rtol * |x|` (the Python-ecosystem convention; also `math.isclose(rel_tol, abs_tol)`, `np.allclose(rtol, atol)`) | caller or finite differences |
| ibex (C++) | `DefaultSolver` / `Solver(sys, eps_x_min, eps_x_max)` | `Solution` with status INNER / BOUNDARY / UNKNOWN, a cell limit and a timeout | absolute per-variable, min and max | symbolic |
| arb / flint | `arb_calc_isolate_roots`, `arb_calc_refine_root_newton` | isolating intervals, flags for proved | absolute | caller's function with Taylor order |

reading: the task-name (`roots`, `solve`, `verifynlss`) is the norm for "all zeros in a box"; the method name
(`newton`, pyinterval's method, arb's `refine_root_newton`) is used where the function IS one step of one
method. a two-state status (unique / unknown) is universal; ibex has three. an absolute tolerance is the
default everywhere; relative is an optional second knob where it exists at all.

## 1. Q11(a) public surface: `Dual`, `derivative`, `newton`, `Root` exported from `intervals`

**the question.** D19(a): the four names (and M16a's four, Q12(b)) are public and re-exported from the
package root, not "newton as a test only" (the 2026-09-26 revision's minimal reading). what is public at
2.0 is what cannot be renamed or dropped without a deprecation cycle.

**option 1 — all eight exported from `intervals` (built).**
meaning: `from intervals import newton, solve, Dual, derivative, gradient, jacobian, Root, RootBox`.
pros: the README's headline demo works in one import; H3 was commissioned as "the demonstration of what
multi-intervals are for" (`v2-plan.md` "the solver stack", first paragraph), and a demonstration hidden in a
submodule demonstrates less; `tests/test_applicator.py::test_package_exports_unchanged` already pins the
list, so growth is deliberate. cons: eight names in the root namespace for one feature (the root has about
50); any later change of their signatures or result shape is a top-level break; `gradient` shadows
`numpy.gradient` (a different thing: finite differences of an array) and `solve` reads as a *linear* solve
to numpy/scipy users (`np.linalg.solve`, `scipy.linalg.solve`) under `from intervals import *`.
when better: when the solver is a first-class feature the owner intends to keep and extend; when the
README demo matters.

**option 2 — public in the submodules only (`intervals.solver`, `intervals.autodiff`), not re-exported.**
meaning: `from intervals.solver import newton`; drop the eight from `__init__.py` and `__all__`. the 1788
layer already takes this shape (`from intervals import ieee1788`, Q13(a)).
pros: the root namespace stays "the set type and its ops"; the solver can change shape in a 2.x with less
fallout (a submodule's API is still public, but users expect more churn there); no star-import collisions.
cons: the README demo needs two import lines; two tiers of "public" need explaining; the precedent (the
1788 layer) is kept out because it has *different semantics for the same ops*, which is not the solver's
case — the solver is new functionality, no conflict.
when better: when the owner is unsure the solver's API is settled (see §5 on `tol` and §10 on `Root`'s
shape) and wants 2.0 out without freezing it.

**option 3 — a narrower root export: `newton`, `solve`, `derivative`, `gradient`, `jacobian` at the root;
`Dual`, `Root`, `RootBox` from their modules.**
meaning: the verbs at the root, the types where they live. pros: five names, the ones a user types.
cons: a user writing `f` for `newton` does meet `Dual` (an `isinstance` check, `Dual.constant` for a
set-valued constant; `autodiff.py::Dual._coerce` accepts bare sets, so this is rare), and must import the
result types to annotate; splitting a feature's names across two import paths is the worst of both.
when better: rarely; only if the root namespace size itself is the concern.

**option 4 — a sub-package namespace exported once: `from intervals import solver` then `solver.newton`.**
meaning: export the module objects, not the names (what `ieee1788` does). pros/cons as option 2 with one
import line. a cosmetic variant of 2.

**recommendation: option 1, keep as built. confidence: medium-high.** the solver is the point of H3 and
of multi-intervals (one newton step splits at the gap), and the README is written around it. the two
things that would make me switch to option 2: (i) the owner expects to change `tol`'s meaning or
`Root`/`RootBox`'s shape after 2.0 (§5, §10) — then do not freeze them at the root; (ii) the owner sees
the package as "a set type" with the solver as an add-on, in which case the 1788-layer shape is the
consistent one. the numpy name collisions are tolerable because nobody should `from intervals import *`
alongside `from numpy import *`; mention them in the README if option 1 stands.

**cost of changing later.** before 2.0: one edit to `__init__.py` and the pinned `__all__` test, the
README's import lines. after 2.0: removing a root export is a break (a deprecation shim in `__init__`
for a release); adding one is free. so "narrow now, widen later" is the cheap direction, which argues
mildly for option 2 only if there is real doubt.

## 2. Q11(b) the newton step runs only where decorations prove C¹

**the question.** D19(b): `f` is evaluated on a `Dual` of two `DecoratedInterval`s and the step runs
only where both the value and the derivative are dac or com (`solver.py::_evaluate`, and `_jacobian` in
n variables); elsewhere the piece is only pruned by range and bisected. this is a correctness rule with a
cost, not an API choice: no name or signature depends on it.

**option 1 — the gate as built (value and derivative both dac or better).**
pros: `unique` is a theorem, not a hope: `tests/test_solver.py::test_not_c1_would_lose_a_zero` shows the
silent failure without it (the derivative's *values* of `abs(x) + x/2 - 1/4` lose the zero at 0);
the sabotage table (plan §2 M15) shows the value's decoration is needed separately (a jump:
`::test_a_jump_is_caught_by_the_value_decoration`). the gate is per piece, so it costs only on pieces
touching a non-C¹ point; measured cost in n variables: the kink 102 calls against 58 with the gate on the
box rather than the closed hull (plan §2 M16a, "the rules measured on the prototype"), and factors not C¹
at their zeros cost 500 to 1500 calls unbudgeted (the box across the kink line is bisected to `tol`).
cons: conservative where the decoration is weaker than the truth — `abs(x) ** 3` is C² at 0 but
`abs`'s derivative is `sign`, def at 0, so pieces across 0 are bisected; `x ** 1.5` at 0 likewise. these
are the multiple or degenerate zeros that newton proves nothing about anyway, so the loss is cost, not
proofs.
when better: always, for a library whose selling point is that `unique=True` is proved.

**option 2 — trust the caller (`assume_c1=True` keyword, or no gate).**
meaning: the step runs everywhere the derivative's set is computable. pros: fewer bisections on non-C¹
functions; the only way to step across a point where the formula's decoration is pessimistic.
cons: `unique` becomes "unique if your `f` is C¹ on the piece", which the user cannot check per piece;
the kind of silent soundness hole the whole repo is built to refuse. a keyword is *additive*: it can be
added in any 2.x without breaking anything, so there is no reason to decide it now.
when better: never as the default; as an opt-in only if a user measures the gate as their bottleneck.

**option 3 — gate on the derivative's decoration only.**
meaning: drop the value check. cons: a hand-made `Dual` with a dac derivative and a trv value (a jump)
would be stepped; the sabotage row "C1 gate: value decoration ignored" was green until the test was
written, which is why the test exists. no measurable gain (every built op's derivative formula already
carries its op's domain). reject.

**recommendation: option 1, keep. confidence: high.** what would change my mind: nothing about the
default; an opt-in `assume_c1` is the only reasonable relaxation and it can come later, additively.

**cost of changing later.** zero API cost in either direction (internal rule; a keyword is additive).

## 3. Q11(c) the step is `mul_rev`, never `/`

**the question.** D19(c): the newton set is `m + mul_rev(F', -f(m))` (`solver.py::_newton_step`), and
in n variables the gauss-seidel row is `m[i] + mul_rev(Mx[i][i], c)` (`solver.py::_gauss_seidel`). the
library's `/` has `[0] / [0] = ∅` (D7), where the mean value theorem needs every `t` with `t * 0 = 0`.

**option 1 — `mul_rev` (built).** pros: the only sound choice under D7; it is also exactly the
multi-interval advantage: `mul_rev` of a slope set holding 0 gives two pieces and the step cuts the gap
in one go (1788's `mulRevToPair` as one set); the sabotage row "division instead of mul_rev" is red by
`tests/test_solver.py::test_every_zero_is_enclosed`, and on the prototype gauss-seidel with `/` lost a zero
of the cusp (plan §2 M16a, 5 calls and 1 box). cons: none for correctness; `mul_rev` costs a little more
than `/` where 0 is not in the slope (not measured; the arithmetic dominates anyway, Q12(a)).

**option 2 — `/` with a special case when `0 ∈ F'`.** meaning: re-derive the two-piece result by
hand. this *is* `mul_rev`, written twice. reject.

**option 3 — `/` and the hull where the slope holds 0 (what a connected-interval library does).**
sound but loses the split: the cusp cost 391 calls against 359 on the prototype with the hull. reject.

**recommendation: keep, confidence: very high.** this is not an owner's choice but a consequence of D7
and the mean value theorem; it needs no decision, only the record it already has. internal: no API cost
either way.

## 4. Q11(d) one variable only (and: fold `newton` into `solve`?)

**the question.** D19(d) said `newton` is one variable; M16a superseded the "only" by adding `solve`,
which at `n == 1` delegates to `newton` and rewraps each `Root` as `RootBox((interval,), unique)`
(`solver.py::solve`). so today there are two entry points and two result types. the live question is
whether 2.0 should ship both, or one.

**option 1 — two functions, two result types (built).**
meaning: `newton(f, x) -> tuple[Root]` with `Root.interval` a set; `solve(F, xs) -> tuple[RootBox]` with
`RootBox.box` a tuple of sets; `F` returns a sequence even at n == 1.
pros: the one-variable call is the headline and reads naturally (`for root in newton(f, X): root.interval`);
the 1-D loop is cheaper than the n-D loop run at n == 1 (1.4x to 2x fewer calls measured 2026-09-28, plan
§2 M16a) and that is why the delegation exists; `Root` and `RootBox` each have one obvious field name.
cons: two names for one task; the n-D name is a task name (`solve`) and the 1-D name an algorithm name
(`newton`), an inconsistency a user notices; generic code must branch on which type it got.
when better: when 1-D use dominates (it will, for a library whose users are mostly not solving systems)
and the README demo matters.

**option 2 — one function, `solve`, list-of-sets always.**
meaning: drop `newton`; `solve(lambda x: (x ** 2 - 2,), [M(-10, 10)])` and `root.box[0]`.
pros: one name, one type, one doc. cons: the common case gets uglier (a 1-tuple return, a 1-list input,
`.box[0]`); the 1-D fast path still exists inside, so nothing is simpler in the code; the demo loses its
punch. when better: if the owner wants the smallest possible public surface and expects system-solving to
be the main use. I do not think it is.

**option 3 — one name with two shapes (the Julia shape): `roots(f, X)` for a set, `roots(F, [X, Y])` for a
box, one result type `Root(box, unique)` with a `.interval` convenience for n == 1.**
meaning: dispatch on whether the second argument is a set or a sequence; one NamedTuple whose first field
is always a tuple (or a class with both views). pros: one public verb, the n == 1 fast path kept inside;
matches IntervalRootFinding's `roots`/`Root`; generic code never branches. cons: a function whose return
*shape* depends on argument shape is the numpy pattern users complain about; `Root.interval` for n == 1
and `Root.box` for n > 1 means one of the two is a property that raises or wraps; a set vs a sequence of
sets is an easy dispatch (sets are not sequences here), but a user passing a one-element list and
expecting `.interval` gets `.box` — the kind of thing a NamedTuple cannot paper over. when better: if the
owner wants a single verb and is willing to pay with a slightly odd result type.

**option 4 — keep two functions but let `solve` also accept a bare set for `xs` (and `F` returning a bare
value at n == 1), returning `RootBox`es.**
meaning: `solve(lambda x: x ** 2 - 2, M(-10, 10))` works and gives `RootBox((interval,), unique)`.
pros: additive — accepting a wider input is not a break, so it can come in any 2.x; it gives generic code
one entry point without removing the natural 1-D one. cons: two ways to do it. when better: as a later
addition if users ask; not a 2.0 decision.

**recommendation: option 1 for 2.0, with option 4 noted as free to add later. confidence: medium-high.**
the thing that is hard to take back is *removing* `newton` from the public surface; keeping it costs
nothing later. the name inconsistency (`newton` vs `solve`) is real but defensible: `newton` IS the
interval newton operator with the extended division, which is the one method that will ever be wanted in
one variable over multi-intervals (krawczyk in 1-D is strictly weaker: it has no division, so no split),
so the algorithm name is stable; `solve` is a composite (krawczyk proves, gauss-seidel narrows,
bisection) with no honest single method name. what would change my mind: if the owner prefers the Julia
shape (one verb), choose option 3 *now* — the rename of `newton` → `roots` and the type merge are the
kind of change that must happen before 2.0 or not at all.

**cost of changing later.** before 2.0: a rename and a NamedTuple merge touch `solver.py`, `__init__.py`,
the README doctests and about 80 test call sites (`newton(` appears throughout `tests/test_solver.py` and
`tests/test_solve.py::test_n_equals_one_is_newton`); an afternoon. after 2.0: option 4 is free; options 2
and 3 need a deprecation cycle with aliases and a result type whose fields change — the expensive kind.

## 5. Q11(e) `tol=1e-10` absolute, `max_steps=10_000`

**the question.** both keywords are public API on `newton` and `solve` (keyword-only). `tol` is an
absolute width in the units of `x`; it governs only *unproved* pieces (a proved zero is narrowed while
newton narrows it, whatever `tol`: `solver.py::newton`, the `if unique:` branch), plus the 8-step
allowance past `tol` (`_PAST_TOL`). `max_steps` counts boxes popped from the stack. the owner asks
whether `tol` should be relative.

**what the probe shows (2026-10-03, `.scratch/owner-questions/probes-solver/tol_scale.py`, `small_scale.py`;
loaded laptop, times indicative):**

| case | as built (`tol=1e-10`) | `tol=1e-30` or `tol=0` |
|---|---|---|
| `x - 1e20` on `[0, 1e21]` | 1 root, unique, `[1e20]`, 13 calls | — |
| `x**2 - 1e40` on `[-1e21, 1e21]` | 2 roots, both unique, width 1.6e4 (adjacent doubles), 40 calls | — |
| `x**2 - 1e-40` on `[-1, 1]` | 2 roots, **both unproved**, width 2.3e-14 (relative width ~1e6), 132 calls | 2 roots, both **unique**, width 1.5e-36, 216 calls |
| `(x - 1e-20)**2` (double zero) | 1 root, unproved, `(5e-41, 2.3e-14)`, 67 calls | — |
| `(x - 1/3)**2` on `[0, 1]` | 1 root, unproved, width 1.1e-16, 122 calls | `tol=0`: width 5.6e-17, 124 calls |
| `(x - 1e20)**2` on `[0, 1e21]` | unique `[1e20]` (exact end hit), 145 calls | — |
| `abs(x - 1e-20)` (kink) | 1 root, unproved, `(0, 5.8e-11]`, 72 calls | — |
| `solve(lambda x: (x**2 - 2,), [M(-10, 10)])` | two `RootBox((interval,), True)` | — |

reading: at large scale an absolute `tol` is unreachable and harmless (a piece that cannot be split is
output; a proved zero narrows anyway). at small scale it is harmful: the piece around ±1e-20 is "narrow
enough" at 2.3e-14 before newton can contract it (the slope `2x` over `[5e-41, 2.3e-14]` spans 26 decades,
so the newton set is never inside the piece), and a simple zero is output unproved and 1e6 times too wide.
`tol=0` fixes it in one variable (216 calls) because a proved zero stops itself and an unsplittable piece
stops itself; `tol=0` is however a footgun on a continuum or a multiple zero in n variables (bisection to
adjacent doubles everywhere, `max_steps` the only brake).

**option 1 — `tol` absolute, default 1e-10 (built).**
pros: the convention of every interval root finder I know (IntervalRootFinding's `abstol`, ibex's
`eps_x`, arb's); one number with one meaning; it is the only criterion that bounds work on a continuum
(`tests/test_solver.py::test_tol_stops_refining`, `tests/test_solve.py::test_constant_and_continuum_systems`).
cons: scale-blind, as the probe shows; the default 1e-10 is arbitrary (IntervalRootFinding 1e-7 by
default; scipy's `xtol` 2e-12). when better: when zeros live at scale 1e-3..1e6, which is most users.

**option 2 — `tol` relative (width over magnitude).**
meaning: stop when `wid <= tol * mag(piece)` (or `* mig`). pros: scale-free away from 0. cons: a piece
holding 0 has `mig = 0` and `mag` ~ its width, so a relative criterion either never stops (mig) or stops
at once (mag) there; every library that offers it pairs it with an absolute floor; a pure relative
default would bisect a continuum through 0 down to the subnormals. reject as the *only* knob.

**option 3 — both: `tol` (absolute) and `rtol` (relative), stop when `wid <= tol + rtol * mig`, default
`rtol=0`.**
meaning: scipy's `xtol + rtol*|x|`, numpy's `atol + rtol*|b|`, IntervalRootFinding 0.6's `abstol`/`reltol`.
pros: the Python convention users expect; `rtol` is additive (a new keyword with a default that reproduces
today's behaviour), so it can land in any 2.x; the names `tol` + `rtol` read fine together (scipy's
`root_scalar` uses `xtol`/`rtol`). cons: two knobs to document; the small-zero case still needs the user
to set `tol=0, rtol=1e-10` (the absolute floor must be below the zero's scale), so it is a tool for a user
who knows, not a better default. when better: when a user with zeros across many decades asks.

**option 4 — hybrid default: `tol * max(1, mag)`.**
meaning: relative above 1, absolute below (like `math.isclose` with `abs_tol=tol`). pros: one knob, sane
at 1e20. cons: does nothing for the small-scale case, which is the only one that hurts; changes the
meaning of `tol` for every user with large values (fewer bisections, wider unproved output). not worth
the semantic shift.

**option 5 — no `tol` at all (IntervalRootFinding's old `roots` ran to a fixed `tol` of 1e-15 and had no
budget).** reject: a continuum would then be `max_steps` boxes always.

**recommendation: option 1 for 2.0, with option 3 (`rtol`) as the additive path later; plus two
non-API fixes before 2.0 (§10): the magnitude split inside `(0, 1]` (measured 12x fewer calls on the
small-zero case, same output) and a docstring sentence that `tol` is absolute, so a zero at a scale below
`tol` is output unproved unless `tol` is lowered. confidence: medium.** what would change my mind: if the
owner expects users with zeros near 0 at tiny scale (chemistry, probabilities), ship `rtol` *now* so the
docstring can say "set `tol=0, rtol=1e-10`" — the keyword itself is cheap. if the owner prefers one
knob, keep absolute and document.

**`max_steps=10_000`.** options: (a) as built, a count of boxes (built; pinned by
`tests/test_solver.py::test_evaluation_budgets` indirectly through call counts); (b) `max_evals`, a count of
calls of `f`/`F`, the cost a user feels (a box costs n + 2 calls plus up to n + 2 at output, so the two are
proportional, not equal); (c) a wall-clock `timeout` (ibex has one) — non-deterministic, reject for a
library whose results are compared in tests; (d) no budget (IntervalRootFinding) — reject: soundness on a
budget is one of the repo's proved properties (`::test_every_zero_is_enclosed_on_a_budget`). recommend (a):
keep the name, say in the docstring that it counts boxes and what a box costs. confidence: medium-high.
the name `max_steps` reads as "newton steps" to a numerical-methods user; `max_boxes` would be more exact,
but a keyword rename after 2.0 is a break and before 2.0 is a grep; low stakes either way.

**cost of changing later.** adding `rtol`: free at any time. changing what `tol` means (relative, hybrid):
a silent behaviour change for every caller — before 2.0 only. renaming `max_steps`: a break after 2.0.

## 6. Q12(a) jacobian as n passes vs vector mode

**the question.** D20(a): `gradient`/`jacobian` and `solver.py::_jacobian` call `F` n times, pass j with
`x_j` seeded `[1]` and the others `Dual.constant` (`autodiff.py::_passes`). vector mode would carry a
tuple of n tangents in one `Dual` and compute the jacobian in one pass. an internal speed choice with one
API consequence: vector mode changes what a `Dual.derivative` *is* (a set today; a tuple of sets then), or
needs a second type.

**the arithmetic of the saving (inferred, not measured).** per pass, `F` computes every value *and* one
derivative; a derivative formula costs about two value-sized ops (product rule: two products and a sum).
so n passes ≈ n values + n derivatives ≈ 3n units; vector mode ≈ 1 value + n derivatives ≈ 2n + 1 units.
saving: n = 2, 6 → 5 (17 %); n = 3, 9 → 7 (22 %); large n, one third. per box the plain call and the point
call (2 more values) stay. the profile (plan §2 M16a: 81 of 95 s in `MultiInterval._binary` →
`applicator._apply`, 54 s of the 95 in the decorated passes) says the arithmetic is the cost, so vector
mode's ceiling is roughly a fifth to a quarter of the solver's time at n = 2 or 3. the n = 3 system that
"did not finish in 120 s" (HANDOFF still-owed) would still not finish in 90 s.

**option 1 — n passes, `Dual` untouched (built).** pros: `Dual` stays the simple, pinned type (42 chain
rules against arb in `tests/test_autodiff.py`); `gradient` at n == 1 is literally `derivative`; the code is
small (`_passes` is 10 lines). cons: the (n − 1) redundant value computations above. when better: now,
while nothing has been measured as too slow.

**option 2 — vector mode inside `Dual` (`derivative` becomes a tuple of n sets).** pros: one pass.
cons: edits every chain rule (`_chain` must map `factor *` over a tuple; the product and quotient rules
and `_pow` need per-component sums); `Dual.derivative`'s type changes, which is a public break if done
after 2.0 and a re-pinning of 42 rows if done before; one third at best. when better: only after a
measured solve is too slow and the arithmetic itself has been made faster (`evaluate-box`, HANDOFF item
8; the gmpy2 backend), when the solver's bookkeeping becomes the remaining cost.

**option 3 — a separate `Jet`/`Tangent` type for vector mode, `Dual` unchanged.** pros: no break; `Dual`
stays as the 1-D teaching type. cons: two autodiff types, the 42 rules written twice (or `Dual` becomes a
`Jet` of length 1 internally, which is option 2 with a facade). when better: if vector mode is ever
wanted, this is the way to add it after 2.0 without a break.

**option 4 (not in the plan) — cheaper redundancy cuts that need no `Dual` change.** (i) skip the plain
`_values` call on a box that will be stepped: the decorated passes already return the value over the
closed hull, and `0 ∉ value(hull)` implies `0 ∉ value(box)`, so the range prune can read the first pass.
but the prune must stay *first* and cheap — most boxes in a branch and prune die at the prune, where one
plain call beats n passes — so this helps only boxes that survive; (ii) `evaluate-box` (HANDOFF item 8): a
float corner's exact value computed three times under `OUTWARD` — a saving in every op of every pass,
larger than vector mode's and independent of it. neither measured.

**recommendation: option 1, keep; do not schedule vector mode; if speed is ever the task, measure
`evaluate-box` first. confidence: high.** what would change my mind: a user workload at n ≥ 4, where
vector mode's third starts to matter and the n passes also multiply the decorated overhead.

**cost of changing later.** internal if done as option 3; a public break if `Dual.derivative`'s type
changes after 2.0. so the one thing to decide *now* is only: is `Dual.derivative` a set, forever? yes.

## 7. Q12(b) names `gradient`, `jacobian`, `solve`, `RootBox`

**the question.** D20(b): the four names, public and exported (§1); the alternative named in the plan is
`newton_system` / `krawczyk` and a widened `Root`. this is the part of Q11/Q12 that is hardest to take
back: a public name is forever or a deprecation cycle.

**`gradient`, `jacobian`.** standard (ForwardDiff.gradient/jacobian, jax.jacobian, sympy's `Matrix.jacobian`,
autograd). the collision: `numpy.gradient` is finite differences of an array — only a star-import problem.
alternative `grad` (jax) is terser but less clear. **keep, confidence high.** one wrinkle: `jacobian`
returns rows as nested tuples, no matrix type (D20 text: "rows as tuples, no matrix type"); that is right
for a library without an array type and keeps numpy optional.

**`solve` for the n-variable solver.** options:
* `solve` (built). pros: the task name, short, what sympy calls its equation solver. cons: to numpy/scipy
  users `solve` is the *linear* solve (`np.linalg.solve`, `scipy.linalg.solve`); scipy's nonlinear system
  solver is `root`/`fsolve`. a user may guess `intervals.solve(A, b)`. mild.
* `roots` (IntervalRootFinding's name, 1-D and n-D alike). pros: says "all the zeros", the thing this does
  that scipy does not; sets up option 3 of §4 (one verb for both). cons: `numpy.roots` is polynomial roots
  from coefficients, a closer false friend than `solve`'s; and `roots` for n-D alone, with `newton` for
  1-D, is still two names.
* `zeros` / `find_zeros` (Roots.jl's `find_zeros`). `zeros` collides with `numpy.zeros` under star-import,
  the one collision that would actually bite (`zeros((3, 3))` returning a TypeError from an interval solver).
  reject `zeros`; `find_zeros` is honest but long and unlike the rest of the API (no other verb has `find_`).
* `newton_system`: wrong — the method is krawczyk + gauss-seidel + bisection, and only the narrowing is
  newton-like. `krawczyk`: names one third of the method, and the third that does not narrow. reject both.
* `verify_zeros` / `enclose_zeros`: says what makes it different from scipy (verified, enclosing). unusual
  in Python; closest to INTLAB's `verifynlssall`. defensible but nobody will guess it.

**`RootBox` vs a widened `Root`.** options:
* two NamedTuples, `Root(interval, unique)` and `RootBox(box, unique)` (built). pros: each field has the
  obvious name; `solve` at n == 1 returns `RootBox((interval,), unique)`, so `solve`'s type never varies.
  cons: two types for one concept; generic code branches.
* one `Root(box, unique)` with `box` always a tuple, `newton` returning `Root((interval,), unique)`. pros:
  one type. cons: the 1-D demo reads `root.box[0]`; the headline loses. unless `Root` grows an `.interval`
  property that returns `box[0]` for n == 1 and raises otherwise — workable, slightly odd.
* a class (not a NamedTuple) with `box`, `interval` (n == 1 only) and `unique`, no tuple unpacking. see
  §10 on why unpacking is a liability for `unique`'s future.

**recommendation: keep `gradient`, `jacobian` (high); keep `solve` (medium: the linear-solve false
friend is the only mark against it and the alternatives each have a worse one; if the owner wants the
one-verb Julia shape, `roots` for both is the name to pick, and §4 option 3 is the shape); keep `RootBox`
beside `Root` (medium-high) unless §4 option 3 is taken, in which case merge them now.** what would change
my mind on `solve`: an intended later `solve(A, b)` for interval linear systems (a natural addition to an
interval library — gauss-seidel is already in `solver.py`), which `solve` the nonlinear name would block.
if the owner can imagine wanting that, name the nonlinear one `roots` today.

**cost of changing later.** before 2.0: grep-and-rename in `solver.py`, `__init__.py`, README, the test
files. after 2.0: aliases and a deprecation release per name; a type merge changes field names — the
costliest change in these two questions.

## 8. Q12(c) zeros on split faces: unproved slivers vs off-centre bisection

**the question.** a zero lying exactly on a face where the solver split (a float midpoint, a
magnitude-split point 0 / ±1 / ±2^k, or a simplest-point cut) is a closed end of one box and an open end
of its neighbour; no krawczyk set fits inside an interior there, and the inflation (`solver.py::_inflate`)
is clipped to the box's region, which `_regions` cuts at the face, so it cannot cross either. the
simplest-point rule (`_simplest_point`) catches the zero when `F` is *exactly* `[0]` at the simplest
rational of the box (0, 1, 1/2, ... — which is what split faces usually are); otherwise the zero is
enclosed by unproved boxes of width `tol`, often with unproved slivers beside it. the plan's measured
alternative: bisect at 63/128 of the width instead of 1/2, so a zero rarely lands on a face.

**how often does it happen?** a split face is a double (float midpoint, `_point_in`) or a power of two.
a zero lands on one only if the zero IS that double exactly: zeros of transcendental systems never; zeros
of systems with float or simple-rational coefficients sometimes (`y - 0.1`: pinned by a *step*, not a
split, so the inflation handles it — `tests/test_solve.py::test_simple_zeros_are_proved_unique[y - 0.1]`;
a zero at 0 or 1 or 1/2: the simplest point handles it — `::test_simple_rational_zeros_are_exact_points`).
what is left: a zero at a non-simple double that is also a bisection midpoint of the particular box
sequence — e.g. a zero at 0.75 on `[0.5, 1]` after a split at 0.5 — where `F` evaluates to exactly `[0]`
(then the simplest point of a tiny box around 0.75 is 3/4 and catches it anyway), or does not (irrational
coefficients, rounding: then nothing can prove a point zero, and only off-centre bisection moves the face
away). so the residue is: float-coefficient systems whose zero is a non-simple dyadic and whose `F` does
not evaluate exactly at it. rare; "noise, not error" (HANDOFF) is the right reading.

**option 1 — centred bisection + simplest point (built).** measured: constructed system 3, all 4 face
zeros proved, 988 calls (plan §2 M16a). pros: deterministic, the simplest point costs one call per
unproved box. cons: the residue above.

**option 2 — off-centre bisection (63/128) alone.** measured *before* the simplest point existed: 3 of 4
proved, 943 calls. now redundant with option 1 for simple rationals; it would still help the residue.
cons: a zero can land on a 63/128 point too (less likely, not impossible); it makes every split
asymmetric, so budgets and pinned call counts (`::test_evaluation_budgets`) move; it does nothing for a
zero at a magnitude-split point (0, ±2^k), which are the common face zeros and which the simplest point
already handles.

**option 3 — both.** marginal gain on the residue, at the price of re-measuring every budget row.

**option 4 (not in the plan) — let the output merge adjacent unproved boxes into one per connected
unproved region.** not a proof, but turns "slivers beside a tol-box" into one box; also what the
still-owed continuum bullet (2n + 1 boxes per tol-box) asks for. cosmetic, post-processing, no soundness
risk (a hull of enclosures encloses), and additive — but it changes the pinned output counts and the
promise "pairwise disjoint in some coordinate" would need "and maximal". after 2.0 it is a behaviour
change users might see as a regression (fewer, wider boxes). if wanted, decide before 2.0.

**recommendation: option 1, keep; confidence: high.** the residue is rare and sound. what would change
my mind: a user system with float coefficients whose zeros are dyadic by construction (grid problems); then
option 2 is a 5-line change in `_bisect`, internal, any time.

**cost of changing later.** internal (no API) for options 2 and 3; option 4 changes observable output
shape and is better decided before 2.0 (see §10).

## 9. Q12(d) the simplest-point rule and the inflated krawczyk test

**the question.** two rules beyond H3's wording ("a gradient, a jacobian, krawczyk"), run before an
unproved box is output (`solver.py::_finish_box`): (1) the simplest rational of the box's closed hull is
evaluated, and if `F` is exactly `[0]` there it is output alone as a unique point, the rest as up to 2n
boxes; (2) else krawczyk once more on the box inflated by twice its width (floor 1e-12 relative, 1e-30
absolute) and clipped to the box's region (`_inflate`, `_inflated_unique`), with its own jacobian and C¹
gate. each measured necessary: without (1) the cusp's `(0, 0)` and `(1, 1)` end unproved; without (2)
`(x**2 - 2, y - 1/4)` proves neither zero (`v2-plan.md` "2026-09-28 revision: M16a").

**option 1 — keep both (built).** pros: the textbook systems (pinned coordinates, zeros at simple
rationals) are proved, which is what a user tries first; the region bookkeeping (`_regions`) was the hard
part and is pinned (`::test_inflation_is_clipped_to_the_region`, `::test_the_simplest_points_rest_is_its_own_region`);
cost 1 call per unproved box for (1), n + 2 for (2) (the cusp: 12 calls of 371). cons: more code than H3
asked for (about 120 lines); the inflation's constants (2x, 1e-12, 1e-30) are tuned, not derived.

**option 2 — drop (1), keep (2).** zeros at simple rationals on split faces go unproved (the cusp's two).
the vertex rule (every closed vertex, up to 2^n calls) was the design's first attempt and did worse
(0 of 2, 1 of 2, 1 of 2 against 2, 2, 2). reject.

**option 3 — drop (2), keep (1).** any system with a row that pins a coordinate (`y - 1/4`, `3y - 1`,
`y - 0.1`) proves nothing: the pinned component is degenerate or an ulp wide at once, and no `K` fits
inside its interior. these are exactly the systems a user writes to try the solver. reject.

**option 4 (not in the plan) — epsilon inflation at every step, not only at output (Rump's practice;
INTLAB's `verifynlss` inflates iteratively).** meaning: run krawczyk on the inflated box in the main loop
too, so proofs land earlier and a proved box is then narrowed instead of bisected. pros: fewer steps to a
proof on near-degenerate boxes; it might remove the need for `_PAST_TOL` steps on pinned systems.
cons: n + 2 extra calls per step on boxes that would have been proved anyway; the region clip is needed at
every step (already carried); not measured. when better: as a later speed experiment; it is internal.

**option 5 (not in the plan) — report existence separately from uniqueness.** krawczyk with `K ⊆ H` (not
the interior) gives existence by brouwer without uniqueness; a third output state ("at least one zero,
maybe several") is what ibex's BOUNDARY/UNKNOWN split and some papers report. not a replacement for (1)
or (2); it bears on `unique: bool` (§10).

**recommendation: option 1, keep; confidence: high.** the two rules are measured, pinned and sound
(the soundness argument is in `solve`'s docstring and `_inflated_unique`'s). option 4 is the one worth a
later measurement. nothing here is API.

**cost of changing later.** zero: internal rules; only output *counts* (budget tests) move.

## 10. not asked but worth deciding before 2.0

1. **`Root`/`RootBox` are NamedTuples with a `bool unique`: a two-state status frozen by tuple
   unpacking.** `interval, unique = root` is public behaviour; so is `Root(piece, True)` positionally.
   a third state is plausible later (existence proved but not uniqueness, §9 option 5; "proved empty" is
   never output; a "singular zero suspected" flag), and so is a third field (the region, the number of
   steps). adding a field to a NamedTuple breaks every unpacking site; widening `unique` from bool to an
   enum breaks `if root.unique:` only if a new truthy state means "not unique" — an enum with
   `UNIQUE`/`UNKNOWN` where only `UNIQUE` is truthy would be compatible, but that is a trick. options:
   (a) keep NamedTuples and accept that any growth is a 3.0 change; (b) make them small frozen classes
   (`@dataclass(frozen=True)`) — attribute access unchanged, unpacking gone, fields addable with defaults;
   (c) keep NamedTuple, name the status field `status` with a two-member enum now. recommend (b) if the
   owner thinks a third state is likely within 2.x, else (a). confidence: medium. cost: before 2.0 one
   class statement each and a handful of test sites that unpack (grep `, unique =` and `Root(` in tests);
   after 2.0 a break.
2. **the magnitude split stops at 1 (`solver.py::_magnitude_split`: `if hi <= 1 ... return None`).** a
   piece inside `(0, 1]` spanning many decades is never split by exponent, so a small zero is reached by
   halving absolute width. probe (2026-10-03, `small_scale.py`, monkeypatched at runtime): extending the
   rule into `(0, 1]` for `lo > 0` cut `x**2 - 1e-40` on `[-1, 1]` from 132 to 11 calls at `tol=1e-10`
   (same unproved outcome, width 1.4e-20 instead of 2.3e-14) and from 216 to 48 calls at `tol=1e-30`
   (both zeros proved either way); `[1e-30, 1]` from 65 to 5 calls. a piece with `lo == 0` and `hi <= 1`
   is left alone by the variant (where to cut `[0, 1]` geometrically is a design choice: a fixed
   `2 ** -27`-ish fraction, or nothing). internal, cheap, re-measure `::test_evaluation_budgets`'s
   rows after (they are upper bounds, so likely still green). recommend doing it with `newton-width`
   (item 3). no API.
3. **`newton-width` (HANDOFF open item 4)**: `width <= piece.wid() / 2` in `solver.py::newton` is a
   true division that overflows on an exact piece wider than the doubles; `solve` already has
   `2 * width <= _width(box)`. a one-token fix; do it before 2.0 so the two loops match.
4. **continuum output (HANDOFF still-owed M16a)**: up to 2n + 1 boxes per `tol`-box, each tol-box of a
   continuum yielding its simplest point plus rest boxes. the solver's promise is "every zero is in some
   box" and "pairwise disjoint in some coordinate", both kept; the *count* is the only complaint. a
   post-pass merging adjacent unproved boxes (§8 option 4) would change observable output and is a
   before-2.0 decision if ever; I would leave the output as it is and document "a continuum is output as
   boxes of width `tol`" (already in `solve`'s docstring, known limits). no change recommended.
5. **n = 3 cost (HANDOFF still-owed)**: not an API matter; the arithmetic is the cost (profile in plan §2
   M16a). vector mode would save at most a fifth to a quarter (§6); the levers are `evaluate-box` and the
   backend. record, do not decide now.
6. **a docstring line each for `tol` (absolute; a zero at a scale below it is output unproved unless
   `tol` is lowered) and `max_steps` (boxes, not calls; what a box costs).** the only two places where a
   user is likely to be surprised by the built behaviour. cheap, before 2.0.
7. **the `solve`-as-linear-solve false friend (§7)**: if an interval linear solver is ever imaginable,
   decide the nonlinear name now.

## 11. summary table

| item | recommendation | confidence | API? | cheap after 2.0? |
|---|---|---|---|---|
| Q11(a) eight names exported from `intervals` | keep (option 1); narrow to submodules only if `tol`/`Root` shape is in doubt | medium-high | yes | removing: no; adding: yes |
| Q11(b) C¹ gate by decorations | keep; an opt-in `assume_c1` is additive later | high | no | yes |
| Q11(c) `mul_rev`, never `/` | keep; a consequence of D7, not a choice | very high | no | n/a |
| Q11(d) `newton` and `solve` both public; fold? | keep both; `solve` accepting a bare set is additive later; if the owner wants one verb, do the Julia shape (`roots`, one `Root(box, unique)`) *now* | medium-high | yes | fold: no; widen: yes |
| Q11(e) `tol` absolute 1e-10 | keep absolute; `rtol=0` additive later; document; fix the magnitude split below 1 (12x on the probe) | medium | yes | `rtol`: yes; meaning change: no |
| Q11(e) `max_steps=10_000` | keep name and meaning (boxes); document the cost per box | medium-high | yes | rename: no |
| Q12(a) n passes vs vector mode | keep n passes; `Dual.derivative` stays a set; speed levers are `evaluate-box` and the backend | high | only if `Dual.derivative`'s type changed | as a new type: yes |
| Q12(b) `gradient`, `jacobian` | keep | high | yes | no |
| Q12(b) `solve` | keep, unless an interval *linear* `solve(A, b)` is imaginable — then `roots` | medium | yes | no |
| Q12(b) `RootBox` beside `Root` | keep, unless Q11(d)'s one-verb shape is taken (then merge now) | medium-high | yes | no |
| Q12(c) centred bisection + simplest point | keep; off-centre is an internal 5-line option any time | high | no | yes |
| Q12(d) simplest point + inflated krawczyk | keep; per-step epsilon inflation is a later speed experiment | high | no | yes |
| §10.1 `Root`/`RootBox` NamedTuple with `bool unique` | consider frozen dataclasses if a third status is plausible in 2.x | medium | yes | no |
| §10.2-3 magnitude split below 1; `newton-width` | do both before 2.0 (internal) | high | no | yes |

status: COMPLETE (2026-10-03). probes: `.scratch/owner-questions/probes-solver/tol_scale.py`, `small_scale.py`
(outputs transcribed in §5 and §10.2; the directory can be deleted once this report's conclusions are
tracked).
