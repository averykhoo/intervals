---
type: Decision
id: D-46
title: implementability review
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, date from the paragraph's text, 2026-09-23
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:e83dd28fafb1f5872a18811abb83bd64ffef0132e8d1b419dce66b6c1612a531
---

implementability review (2026-09-23, second pass), written into "current design" and the
milestones below: infinite result endpoints always go through attainment (the corner-flag rule is
wrong at ±inf); mul splits at zero; ±inf is exact; `-` is arithmetic, set difference is a method;
`==` does not coerce; relations return bool; import-time filters `append=True`; itf1788 hulls the
expected value too; `Cut` normalizes in a subclass because `typing.NamedTuple` refuses `__new__`.
