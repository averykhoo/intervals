---
type: Decision
id: D-81
title: M8, the time layer, built
actor: unrecorded
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:9700ddae537e7c5821201fb9c695167a273db09b6aba69167ac8ed28d2b06757
---

### 2026-10-04 revision: M8, the time layer, built

D30 built as decided, on branch `m8`; the design is now "the time layer (M8)" in current design, the record
(choices among the defaults, the build's own, the sabotage table) `v2-implementation-plan.md` §2 M8. no
deviation from D30; one consequence written down: `dt - dt` of aware ends is the elapsed time between the
instants also when both share a tzinfo (python's `-` then gives the wall-clock difference). the build's
choices for the owner to confirm: the sentinels' names `NEG_INF`/`POS_INF`; a sentinel is no arithmetic
operand; a float factor is exact (`td * 0.1` is not a whole microsecond); `td // real` refused; `tz=` built
