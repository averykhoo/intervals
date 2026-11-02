---
type: Decision
id: D-12
title: a reverse op whose answer has infinitely many or very many pieces
actor: owner
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:0f36e09f5bc6a852c0dba7c27f1ca35931ab5970b8810d4f726fb22f4fe4f449
---

| D12 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13e), now in `v2-plan.md` "elementary and step functions", "empties and warnings" and "ieee 1788".** a reverse op whose answer has infinitely many or very many pieces (`sinRev`, `cosRev`, `tanRev` and their `*Bin` forms over an unbounded or wide `x`): the exact pieces up to the step functions' cap of 1000, past it their hull with a `HullWarning`, the rule `steps.py` already follows. a bounded `x` gets the exact union (`sinRev([0.5, 1], [0, 20])` has 4 pieces), which 1788 cannot give. ends are irrational, so each is its tightest float enclosure, open | exact to 1000 pieces, else hull + warning | M13e |
