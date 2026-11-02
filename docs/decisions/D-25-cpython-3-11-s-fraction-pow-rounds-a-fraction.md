---
type: Decision
id: D-25
title: CPython 3.11's `Fraction.__pow__` rounds a Fraction base to a float before...
actor: owner
decided: 2026-09-28
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:5982584ec0ff84531626f65495b77170461d04bdfc62efcece89276cae8ac2d9
---

| D25 | **decided 2026-09-28 by owner**: CPython 3.11's `Fraction.__pow__` rounds a Fraction base to a float before `MultiInterval.__rpow__` runs, so on 3.11 `Fraction(1, 3) ** OutwardMultiInterval(2)` misses 1/9, and nothing in the library can see it (the other Fraction operators defer correctly; 3.12 returns NotImplemented). drop 3.11, or keep it with `Fraction ** interval` documented as unsupported there? | **python >= 3.12**: `pyproject.toml` `requires-python`, CI's gate matrix 3.12-3.14; `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11 | — |
