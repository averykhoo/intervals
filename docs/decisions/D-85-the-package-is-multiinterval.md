---
type: Decision
id: D-85
title: the package is `multiinterval`
actor: owner
decided: 2026-10-05
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:f8b107fec29a0b82ad3789ffda0fa7cf5d6634dc204c1dbbe92497b3458b8e85
---

### 2026-10-05 revision: the package is `multiinterval`

the owner, 2026-10-05: the library needs a name not taken on PyPI (`intervals` is another project's, so the two
could not be installed side by side). chosen: `multiinterval`, one word, singular, the same spelling for
`pip install` and `import` (the stdlib's and PEP 8's style for packages; it names the main class). the package
directory `intervals/` is now `multiinterval/`, every import, path and doc pointer follows (the dated snapshots
in `references/` keep the old name), and the backend's variable is `MULTIINTERVAL_BACKEND` (was
`INTERVALS_BACKEND`; never released), likewise `MULTIINTERVAL_COREMATH_CACHE`. unchanged: the class names, the
repo folder, the conda env `intervals`, and the run ledger's format id `intervals-gate-ledger/1`. `multiinterval`
was free on PyPI on 2026-10-05 (no project under the name; claimed only by the first upload)
