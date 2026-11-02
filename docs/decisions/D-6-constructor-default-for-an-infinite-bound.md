---
type: Decision
id: D-6
title: constructor default for an infinite bound
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, date of the D1–D6 settled log entry, 2026-09-23
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:ad2095adc0541c405f022abf4a5db69c0878a779c7ec13d28323d6b1e52121fd
---

| D6 | constructor default for an infinite bound: `MI(1, inf)` = `[1, inf]` (literal) or `[1, inf)` (1788 reading)? **settled (v2-plan.md current design): literal** — `[a, inf]` and `[a, inf)` are different sets and "a user-typed `[1, inf]` is taken literally". D2 removes the blow-up footgun that made this look open | literal | — |
