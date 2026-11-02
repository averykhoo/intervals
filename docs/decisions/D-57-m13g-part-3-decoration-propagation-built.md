---
type: Decision
id: D-57
title: M13g part 3, decoration propagation, built
actor: unrecorded
decided: 2026-09-26
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:499f3e8b7a4cd8d41fa58dc6844a9cf45f2fb10fd770ed4bd6d8474a960b8610
---

### 2026-09-26 revision: M13g part 3, decoration propagation, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part
3") and "ieee 1788" above. `DecoratedInterval` now propagates 1788's decorations through every point
function of the core and decorates set operations trv; the adapter checks the decoration of every
decorated vector of those ops. the choices the plan left open, each the conservative reading, flagged:
* **a multi-piece box is decided on the set, not on the hull**: continuity on the set is continuity
  on each piece, the pieces being apart (the owner's task text asked for this reading)
* **an attained ±inf is outside every domain**: 1788's functions are functions of reals, so an
  operand holding ±inf as a point gets trv even where the core takes a limit (`exp([0, inf])` is trv,
  `exp([0, inf))` dac)
* **com needs the result bounded as returned**: the exact result for exact operands (so
  `[1,2]_com + [5,max]_com` stays com here, 12 plain-pass rows under "decoration expectations"), the
  rounded one for float operands (`OutwardMultiInterval` gives `[6.0, inf)`, dac, as 1788;
  `MultiInterval` rounds `2.0 + max` to nearest, max, and keeps com). the owner question of part 2
  (`_BOUNDED_EXACTLY`: this category, or an exact-values one) covers these rows too
* continuity at a point is relative to the op's domain (`sqrt` at 0, `acosh` at 1, `pow` at `x = 0`
  with `y > 0` are com), as the vectors require
* `%`, `//`, `divmod` and `round(ndigits)`, not in 1788, get the same rule from their definitions;
  `sign` and every step function at a jump point is dac at best
* booleans, numbers and relations stay off the wrapper (`.interval` first); a bare `MultiInterval`
  operand is a `TypeError`, a real number newDec's point
* the core's warnings reach the caller once; the decoration's own core calls are silenced
