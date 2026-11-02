---
type: Decision
id: D-34
title: the fuzz and gate:gmpy2 phases run on pytest-xdist (fuzz-xdist; a build's defaults, awaiting the owner)
actor: unrecorded
decided: 2026-10-08
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:a0c4410f90578ca42855a4ea4100ef12ac033b782cbb480dc5f4f18a5575acd1
---

### 2026-10-08: D34, the fuzz and gate:gmpy2 phases run on pytest-xdist (fuzz-xdist; a build's defaults, awaiting the owner)

> **confirmed 2026-10-08 by the owner, as built** (Q27 (b), D35)

the build of HANDOFF's row fuzz-xdist (`docs/records.md`), on the 2026-10-06 study (`references/fuzz-speed-2026-10-06/`).
asked as Q27 (b).

* **which phases**: the fuzz phases and `gate:gmpy2`; the gate's own (`gate:itf`, `gate:rest`) and `docs` stay serial,
  as CLAUDE.md's gate is plain pytest. xdist splits tests between processes, never one test's examples, so a
  fuzz-x10 verdict means what it did serially, and the ledger keys nothing new
* **how many**: `GATE_WORKERS`, default 8 (the study's fastest, on this laptop's 2 P-cores + 8 E-cores); 0 or 1
  serial; the count is recorded in the row's facts (`workers=K`), never matched. without pytest-xdist a phase runs
  serially and says so, rather than failing on `-n`
* **where**: `tools/gate.py::pytest_command` adds `-n K --dist worksteal` after `phase_spec`'s arguments, which stay
  the selection (what ci.yml's gmpy2 job runs, pinned by `tests/test_gate_ledger.py::test_ci_runs_the_gmpy2_phase`).
  pytest-xdist joins `[test]`; CI installs it and does not use it (fuzz.yml and ci.yml unchanged)
* **the wall-clock tests** (four, one failed under load at `-n 4` in the study) measure CPU time
  (`time.process_time()`), the cost they guard (linear, not quadratic), which other processes do not inflate; their
  bounds unchanged
