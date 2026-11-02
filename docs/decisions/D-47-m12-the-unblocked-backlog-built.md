---
type: Decision
id: D-47
title: M12, the unblocked backlog built
actor: unrecorded
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:92d9f53dddb361405e49781f9de798a0948982e3f9ae34e6f54c59c9fe527e2f
---

### 2026-09-25 revision: M12, the unblocked backlog built

the (b) items of the M11 backlog, built in one session by owner request ("build all the things that
are unblocked"); the choices made while building are recorded here and in
v2-implementation-plan.md (M12), and written into "current design" above:

* **functions do not go through the applicator**: an irrational value has no exact `Value` for the
  descriptor's `fn` to return, and sin/cos/tan split at irrational points. `functions.py` has its
  own small evaluator, as `modulo.py` does; it reuses `applicator.split_pieces` and `warn`
* **no libm**: every value comes from a pure-python, correctly rounded evaluator (`elementary.py`),
  so results are the same on every platform and a directed rounding is a true bound. this replaces
  "±1 ulp around trig is pragmatic, not rigorous — document"
* **an irrational value of an exact operand is its tightest float enclosure** (`sqrt([2])` is the
  open one-ulp piece), not a nearest float: an exact operand never loses its true value
* **a moved end is open, outward**: attainment is decided on exact values, so the ends that directed
  rounding moved are open; to nearest, flags stay conservative
* **the end of a function's domain is a point of it** where the one-sided limit exists
  (`log(0)` = -inf, `atanh(±1)` = ±inf); 1788 drops those points, and the 11 vectors where that
  shows are divergence rows (degenerate infinities)
* **atan2 on the negative x axis is pi** (there is no -0); `(0, 0)` and `(±inf, ±inf)` have no value
* **Allen relations stay cut-based** (`[1, 2]` and `[2, 3]` overlap): a new divergence category,
  "cut-based relations", for the 7 overlap vectors where 1788 says meets
* **the step functions share one engine and one cap** (`steps.py`); `modulo.floor` delegates to it
* names: `minimum`/`maximum` (numpy's names for the pointwise min and max; builtin `min` needs a
  bool from `<`), `round_ties_away` (1788's roundTiesToAway), `OutwardMultiInterval`
