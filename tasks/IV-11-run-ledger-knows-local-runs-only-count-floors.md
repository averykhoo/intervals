---
type: Task
id: IV-11
title: Run ledger knows local runs only; count floors are recorded, not enforced
brief: A push whose src is unchanged since origin/master trusts master's fuzz run was green on CI; it does not check
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "still owed", bullet 3 of 14. Source tag: none (an untagged bullet). The bullet below is verbatim.

* the run ledger (2026-10-01) knows local runs only: a push whose src is unchanged since `origin/master`
  trusts that master's fuzz run was green on CI (every push is watched to the end), it does not check.
  count floors (zanzibar's `MIN_TESTS_ALL`) are recorded in each row, not enforced: a gate that
  collected fewer tests would still read green
