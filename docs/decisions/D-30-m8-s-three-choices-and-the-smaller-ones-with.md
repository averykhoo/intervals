---
type: Decision
id: D-30
title: M8's three choices and the smaller ones with them
actor: owner
decided: 2026-10-04
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:cacdcd88f865990335974b64ca62b08c5f8f8ccc22930aa0a5e685452e00e148
---

| D30 | **decided 2026-10-04 by owner: as recommended** (M8's three choices and the smaller ones with them; `references/m8-choices-2026-10-04/`). (a) a naive datetime is exact wall-clock seconds by subtraction from naive 1970-01-01, never `timestamp()`; an aware one its exact UTC seconds; mixing naive and aware raises `TypeError`; aware ends in different zones allowed, the left operand's zone kept for display only. (b) an infinite end reads out as one of two sentinels ordered below/above every datetime, date, timedelta and pandas type, taken back by the constructors; storage stays (a)'s ±inf, closed or open as written. (c) no end-of-day snap: a `date` is the half-open day `[d 00:00, d+1 00:00)`, a datetime an exact instant (v1's hour/minute/second snaps of datetime ends dropped too). with them: a non-microsecond end raises on read-out (a raw Fraction accessor beside it); comparisons return `TruthSet`; foreign `==` is `NotImplemented`, the wrappers hashable; `NaT`/nan raises | as decided | §2 M8 |
