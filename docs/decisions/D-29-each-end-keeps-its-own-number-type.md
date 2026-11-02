---
type: Decision
id: D-29
title: each end keeps its own number type
actor: owner
decided: 2026-10-03
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:9c4efb96fd4f597b1716d6e3fda7c853bb5c072aaf7b70bc408fead6bc17bf90
---

| D29 | **decided 2026-10-03 by owner** (Q19, Q20): each end keeps its own number type. the outward class is isotone within one grid; across grids `f(A)` lies within the tightest double cover of `f(B)` (documented; a public method rounds every end onto the double grid, outward). the exact class's crossed piece is the piece between the two values, each end keeping its flag (`rootn((10 ** -30, 1.0000000000000003e-30], 5)` is `[1e-06, 1/1000000)`) | keep per-end typing | `references/owner-questions-2026-10-03/q19.md`, `q20.md`; §2 "fuzz-steps-isotone", "fuzz-rootn-crossed" |
