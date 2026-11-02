---
type: Decision
id: D-79
title: v1's leftovers settled; shifts dropped
actor: unrecorded
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:be1008da3a318c6e477c9afe2caf9472fb12dc8f6514d4efc8a32657894df796
---

### 2026-10-04 revision: v1's leftovers settled; shifts dropped

the owner, going through what v1 had that v2 lacks (the implementation plan's §4 surface map):
* **shifts dropped**, reversing Q6-shift (built 2026-10-03, `0513109`; reverted). no use case: `<<` on a
  set only saves writing `* 2 ** n`, and the one plausible use, running integer or fixed-point code over
  sets (`sample >> 8`), wants python's int floor (`3 >> 1` is 1, `-3 >> 1` is -2), which the exact
  `>>` (`M(3) >> 1` is `[3/2]`) did not give. integer ranges are not supported, and would differ
  from the reals in kind (no open ends); 1788 has no integer interval type either. `A * 2 ** n` and
  `A // 2 ** n` remain
* **v1's `merge`**: `union(*)` and `intersection(*)` cover it. the "exactly / at least k overlaps" mode
  and its mixed-input parsing (numbers, sets, lists, tuples, loose strings) were artifacts of how v1
  was written, not features: gone
* **`random_multi_interval`**: v1's test helper, gone (hypothesis strategies do its job)
* **a public `apply()`**: not now; `applicator` stays internal
