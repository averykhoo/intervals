---
type: Decision
id: D-28
title: pown of exact operands past one exact-result limit of about 2 ** 22 bits, shared by pown
actor: owner
decided: 2026-10-03
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:982f823a669e1519cd3c6c4cacd192379d678b24a511b43139e1450b63adfdaa
---

| D28 | **decided 2026-10-03 by owner** (Q17, Q18, m14b-open's 4300 digits): pown of exact operands past one exact-result limit of about 2 ** 22 bits, shared by pown, `pow_` and exp2/exp10, is the tightest open float enclosure in the outward class and the value rounded to nearest in the nearest class, with a default-ignored warning; `elementary.EXACT_POWER_LIMIT` stays the float-corner threshold. pown to nearest is correctly rounded for every n (the exact power rounded once, `rounded_pow` past the threshold), not libm's `float ** int`. `repr` does not raise past python's 4300-digit limit (hex past it; `parse` reads it) | as stated | `references/owner-questions-2026-10-03/pown.md`; §2 "owner-answers" |
