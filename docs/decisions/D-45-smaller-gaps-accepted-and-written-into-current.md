---
type: Decision
id: D-45
title: smaller gaps, accepted and written into "current design" 2026-09-23
actor: unrecorded
decided: 2026-09-23
session: 2026-11-02a
decision_status: BUILT
attribution: unrecorded, date from the paragraph's text, 2026-09-23
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:c895b83932bc3c2852ad49839e006eef7d4eb01aa9b55d5f74e542e1ad46787e
---

smaller gaps, accepted and written into "current design" 2026-09-23: `Size` has componentwise
`+`; `TruthSet.certainly`/`.possibly` on `{}` are True/False; `nan` in a constructor is a
`ValueError`; `x[a:b]` restricts to closed `[a, b]`; the import-time `'ignore'` filters are
process-global and overridden by `pytest -W` / `simplefilter`; itf1788 licence check before
vendoring.
