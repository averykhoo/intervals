---
type: Decision
id: D-69
title: M16b, the 1788 layer (H3's second part), built
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:91cb9b809cd95a566793f68aa36eae03f4e241e44a74d5ffcb5afc3b3b27800a
---

### 2026-09-28 revision: M16b, the 1788 layer (H3's second part), built

the owner asked (2026-09-27) "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now"; H3's rest was built as M16 in five streams, this one M16b. the
2026-08-16 decision ("if real conformance is ever needed, it's a thin wrapper class in
`ieee1788.py`, never a mode on MultiInterval") is now built, as designed there. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q13; D21 in
`v2-implementation-plan.md`):
* **shape**: `multiinterval/ieee1788.py`, one class `Interval` for both flavours, snake_case 1788 names
  with 1788's camelCase in `NAMES`, not exported from `multiinterval`
* **numbers of the empty set raise** (`ValueError`, D9's answer, as the owner chose for the
  reductions, Q2) where 1788 says NaN; `inf`/`sup` of it are `±inf` either way
* **Q9 and Q10 answered by the layer, pending Q13 (c), (d)**: 1788's pair with its decoration is
  `ieee1788.mul_rev_to_pair` and the library's `mul_rev` stays one op, trv; the constructors' binary64
  run is the layer's pass, no class argument on the library's constructors. the adapter's 52
  `DECORATION_ONLY` rows and its reasons are left as they are (true of the library).
  why the library's `mul_rev` stays trv rather than decorated as the division: in the library ±inf
  are points, so where 0 is not in `b` the two sets differ (`mul_rev([1, inf], [inf])` is `(0,
  inf]`, `[inf] / [1, inf]` is `[inf]`; measured 2026-09-27, re-checked 2026-09-28), and a
  decoration copied from the division would claim com or dac for a set that is not the division's; a
  multi-piece `b` gives more than two pieces, so a pair has no meaning on the core; and 1788
  decorates its own `mulRev`, the hull of the same set, trv (`libieeep1788_rev.itl:988`), so one
  library op cannot match both
* **where 1788 and the library disagree, the layer answers 1788's way** (cancellation's "no answer",
  `meets`, attained infinities dropped), the library its own
* **the sign of zero follows 1788 in the layer**: `inf` gives `-0.0` for a lower end of 0 and `sup`
  `+0.0` for an upper end of 0 (1788-2015's rule; the critique's n7: "recorded, not an owner
  question" did not fit the layer's principle of answering 1788's way). no vector can see it
  (`tests/itf1788/itl.py::parse_number` reads `-0.0` as `Fraction(0)`); pinned by
  `tests/test_ieee1788_layer.py::test_the_sign_of_zero_is_1788s`. the package's one zero (2026-09-22
  revision) is unchanged: the ends of an `Interval` are never `-0.0`, and nothing re-enters the
  library from `inf`/`sup`
