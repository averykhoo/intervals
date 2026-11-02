---
type: Decision
id: D-3
title: '`int / int` that is not integral'
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, date of the D1–D6 settled log entry, 2026-09-23
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:01177655f58458c601b72e2e1d4feee34c6d8b4a3423c25ebc247684021a8446
---

| D3 | **decided: recommended default**, plus integral Fractions normalize to int in `Cut`. `int / int` that is not integral: Fraction (exact, per "never rounded") or float (what users expect)? | Fraction; `fmt` prints `1/3`; float only if an operand is float | M6 |
