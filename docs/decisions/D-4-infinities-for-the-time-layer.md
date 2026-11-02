---
type: Decision
id: D-4
title: infinities for the time layer
actor: owner
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c8caa723cdf4c66c995bf09d2965f67e561b03e03a602c8b879c0807c5c532ca
---

| D4 | **decided 2026-10-04 by owner: (a)**, with (b)'s two sentinels as read-outs of an infinite end only (D30). was deferred: the time layer is not being rebuilt now; v1's is archived with the rest of v1 at M10 and comes back in M8 on top of the v2 class (see M8, M10). infinities for the time layer: v1 stores float unix seconds (loses sub-µs, dodges the question). v2 options: (a) Fraction seconds in the numeric kernel, thin wrapper; (b) native datetime cuts + two sentinel objects that compare below/above everything | (a) — reuses every kernel test unchanged, when M8 happens | M8 |
