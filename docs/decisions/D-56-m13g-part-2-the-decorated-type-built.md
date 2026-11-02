---
type: Decision
id: D-56
title: M13g part 2, the decorated type, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:3c9381e9acf31ced85bd7fb0069aa8e3150d2d71e5427336ff6c37ddf3184042
---

### 2026-09-26 revision: M13g part 2, the decorated type, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 2").
the decorated wrapper of D16 is `DecoratedInterval` (`multiinterval/decorated.py`, "ieee 1788" above),
named after M8's `DateTimeInterval`, with a `Decoration` enum and no `ill`. the choices the plan left
open, each the conservative reading, flagged:
* **the constructor is strict, `set_dec` is 1788's**: `DecoratedInterval(x, d)` raises
  `UndefinedOperationError` for a decoration that does not fit, as the literal `"[1,]_com"` does;
  `set_dec` demotes as 1788 defines it, with no signal (`libieeep1788_class.itl:283`-`288` expect
  `[empty]_trv` and `_dac`), and raises only for `ill`. raising for every unfitting `setDec` would
  have turned six matching vectors into rows needing a new category, for input 1788 calls valid
* **bounded is decided on the exact set**, so `[1.0E+400]_com` stays com: three vectors are rows
  under the existing "decoration expectations" (owner question: or the PROPOSED exact-parsing
  category, widened)
* a decoration name is exact lower case (`'COM'` raises): a python argument is not 1788 text
* **a core bug fixed**: `MultiInterval.is_finite` and `.finite` raised `OverflowError` on an exact end
  past the doubles (`MultiInterval(10**400)`), since `math.isfinite` converts to float; they compare
  with ±inf now (`tests/test_multi_interval.py::test_finiteness_of_an_exact_end_past_the_doubles`)
* not built: decorations propagated through the core's ops (1788's decorated arithmetic); every
  other op's vectors still drop their decorations
