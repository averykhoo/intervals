---
type: Task
id: IV-8
title: Release 2.0.0
brief: 'Waits on x2-cosh-zero, from-cuts-reason, release-meta; owner 2026-09-26: release when everything is fully done'
pri: NEXT
state: open
deps: [IV-1, IV-2, IV-3]
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
aliases: [H1]
---

Imported 2026-11-02 (CE-11 adoption of intervals) from `HANDOFF.md` at fa4cf12, section "open items (ranked)", row 8 of 8 (id `H1`). The cells below are verbatim.

What: release 2.0.0 (`pyproject.toml` is `2.0.0.dev0`, no tag). before it: rows x2-cosh-zero, from-cuts-reason and release-meta (the licence, metadata, publish workflow and changelog with the release notes, D36), the CORE-MATH full check again if the scalar evaluator changes (clean at `7dd2c0d`), claiming `multiinterval` on PyPI (the first upload, by the publish workflow)

Status / blocker: waiting on those rows (owner, 2026-09-26: release "when everything is fully done")

Spec: `docs/decisions.md` D5, D17, D36, H1 in "owner answers to the open questions" (2026-09-26)
