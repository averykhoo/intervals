---
type: Decision
id: D-73
title: python 3.12 minimum (owner, D25)
actor: unrecorded
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:70b7771da2417ab61eee24c92bbff30000c9023e02cc047767a0a63cddee7111
---

### 2026-09-28 revision: python 3.12 minimum (owner, D25)

* CI's first run of M15 and M16 (36402681261) was red on python 3.11 alone: CPython 3.11's `Fraction.__pow__` answers a non-rational exponent with `float(a) ** b`, so `Fraction(1, 3) ** A` reached `A.__rpow__` as `0.3333333333333333 ** A`, unsound in `OutwardMultiInterval` (1/9 missed) and indistinguishable, inside the library, from a float base. the owner chose python >= 3.12 over documenting `Fraction ** interval` as unsupported on 3.11 (`pyproject.toml`, `ci.yml`); pinned by `tests/test_outward.py::test_a_fraction_base_stays_exact`
