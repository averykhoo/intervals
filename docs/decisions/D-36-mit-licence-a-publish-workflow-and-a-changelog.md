---
type: Decision
id: D-36
title: MIT licence, a publish workflow and a changelog for 2.0.0; Q25 answered
actor: owner
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:68b8b088f40571ba46af123d8678fab2af3af2f0e69ebb5db43483c1ccc85541
---

### 2026-10-08: D36, MIT licence, a publish workflow and a changelog for 2.0.0; Q25 answered

the owner, 2026-10-08, answering Q25: asked what a patent grant is, then "what would you suggest, idm anything",
then, on the session's recommendation (MIT, plus a publish workflow and a changelog), "okay mit".

* **the licence is MIT**, for the library (`multiinterval/`, the wheel), copyright Avery Khoo. a `LICENSE` file at the
  root and `license = "MIT"` (with `license-files`) in `pyproject.toml`. why, over Apache 2.0: a sole author holding
  no patents gives Apache 2.0's patent grant nothing to cover; MIT is a few lines, asks only that the notice be kept,
  and is compatible with GPLv2 projects, which Apache 2.0 is not; the libraries beside it are BSD (mpmath, numpy)
  and the vendored CORE-MATH rows already MIT. the vendored test data keeps its own licences (D15): it is not in the
  wheel, so nothing changes there
* **the metadata `pyproject.toml` lacks**, filled with it: `readme`, `authors`, `urls`, classifiers
* **a publish workflow** (the session's recommendation, taken with "idm anything"): a tag push `v*` builds the sdist
  and wheel and uploads them to PyPI by trusted publishing (OIDC, no stored token). the owner registers the pending
  publisher on pypi.org before the first tag; the tag push is babysat through the upload, as every push is
* **a changelog** (likewise): `CHANGELOG.md`, starting at 2.0.0; its first entry, written from the v1 -> v2 surface
  map (`docs/archive/v2/v2-implementation-plan.md` §4), is the release notes row H1 asks for
* not built: row release-meta (`HANDOFF.md`) builds it, before H1
