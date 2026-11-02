---
type: Decision
id: D-48
title: M13 and M14 planned (not built)
actor: unrecorded
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:090c740416fa2a717004a3b95e0ddb35e50937b65b6d2c96dfb6e22fab46f63e
---

### 2026-09-25 revision: M13 and M14 planned (not built)

owner requests: vendor the whole itf1788 suite (the maintained fork, oheim/ITF1788 at `b6ee1e2`,
19 files) and implement every op in it, as M13; and fuzzing, as M14. the owner accepted every
recommended default, recorded as D9–D17 in v2-implementation-plan.md section 0 with the details.
none of this is "current design" yet: each point moves up into it when its sub-task is built.
in short:
* **D9** `mid`, `rad`, `wid` are of the hull; `mag`, `mig` of the set (`mig([-3,-2] ∪ [2,3])` = 2)
* **D10** `weakly_less`, `strictly_less` for 1788's `less`, `strictLess` (`<` stays pointwise); a
  `.interior` property, so 1788's `interior(A, B)` is `A.within(B.interior)`
* **D11** `**`: an integral exponent is pown as today; a non-integral or interval exponent is 1788's
  `pow`, negative bases dropped with `DomainClippedWarning` (so `MI(-3,1) ** MI(2)` = `[0, 1]`);
  `b ** A` through `__rpow__`; 3-argument `pow` dropped
* **D12** reverse ops with periodic answers: exact pieces up to 1000, else the hull + `HullWarning`
  (built 2026-09-26 at M13e; now in "current design", elementary and step functions, empties and
  warnings, ieee 1788)
* **D13** `cancel_minus` is the Minkowski difference (the largest `X` with `B + X ⊆ A`); where 1788
  answers "no answer" with entire, ours is a real set, under a new residual category
  "cancellation as a Minkowski difference"
* **D14** `python-flint` (Arb) is the independent oracle for the elementary functions, test-only
  (built 2026-09-26; now in "current design", testing)
* **D15** the fork's LGPL-2.1+ files (`mpfi`, `fi_lib`, `c-xsc`) are vendored unmodified as test
  data with their licence files; the wheel ships only `multiinterval/` (built 2026-09-26; now in
  "current design", ieee 1788. five files are all-permissive, not the two `ieee1788-*` alone)
* **D16** decorations, NaI and 1788's constructors in a separate decorated wrapper type, brought
  forward from "later"; the core stays undecorated. open: 1788's signals as warnings or exceptions
* **D17** M13 does not block 2.0.0, and the release is in no hurry
