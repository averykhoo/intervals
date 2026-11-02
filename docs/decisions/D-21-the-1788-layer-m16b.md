---
type: Decision
id: D-21
title: the 1788 layer (M16b)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, the row's decided-in-the-build date, 2026-09-28
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:78e61a5067a8fe704c47d0da1fe76c9959b07de8cdd929839d1000adcbcc404f
---

| D21 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q13) but (b): the layer's numbers of the empty set are `nan`, 1788's answer, not the library's `ValueError`; Q9 and Q10 closed as built.** the 1788 layer (M16b): (a) `multiinterval/ieee1788.py`, one `Interval` class for both flavours, 1788's names in snake_case (`NAMES` has the camelCase), not exported from `multiinterval`; (b) `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad` of the empty set raise `ValueError`, where 1788 says NaN; (c) Q9 answered by the layer: `ieee1788.mul_rev_to_pair` is 1788's pair with its decoration, the library's `mul_rev` unchanged; (d) Q10 answered by the layer: its pass runs the constructors in binary64, no class argument on the library's; (e) where 1788 defines another answer than the library's set (cancellation, overlap, attained infinities), the layer gives 1788's and the library keeps its own | as built | M16b |
