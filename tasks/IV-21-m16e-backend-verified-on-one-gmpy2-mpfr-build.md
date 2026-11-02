---
type: Task
id: IV-21
title: 'M16e: backend verified on one gmpy2/MPFR build only; quiet-machine speed re-measure owed'
brief: gmpy2 2.3.1 / MPFR 4.2.2 on windows (python 3.13); free-threaded builds untested; gmpy2 3 needs a differential first
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 13 of 14. Source tag: M16e. The bullet below is verbatim.

* M16e: the whole suite under `MULTIINTERVAL_BACKEND=gmpy2` runs in CI's `gate-gmpy2` job since 2026-10-04
  (Q16(e)) and locally as `gate:gmpy2` when a backend file changed. the backend is verified only with gmpy2 2.3.1 / MPFR
  4.2.2 on windows (python 3.13); CI's linux jobs run `tests/test_backend.py` with the PyPI wheel, no
  other MPFR build has been run. free-threaded builds untested (the three contexts are shared module
  objects; `backend._use` is a global). the speed numbers were taken beside four other streams' runs:
  a quiet-machine re-measure (`tools/backend_speed.py`) is owed before any number goes into the
  README. gmpy2 3: `auto` and `[test]` stop below it; when it ships, run the differential against it
  before `backend.CEILING` and the `[test]` pin move together
  (`tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` keeps them equal)
