---
type: Task
id: IV-23
title: Review the imported decisions' statuses, actors and synthesised titles
brief: Import set all 87 BUILT; status markers and owner near-misses listed
pri: LATER
state: open
deps: []
source: session 2026-11-02a
created: 2026-11-02
moved: 2026-11-02a
updated: 2026-11-02a
---

The decision log import of 2026-11-02 (CE-11 adoption; `docs/evidence/decision-import-2026-11-02.md`) set every
record BUILT and named an actor only where the text said so mechanically. Review, with the owner where a call is theirs:

- Status markers in the text (set each with `context-engine decision set <id> --set decision_status=<status>`, or
  link with `context-engine decision supersede`): blockquote markers on D-31 (superseded by D32), D-33 and D-34
  (confirmed), D-37, D-40, D-41, D-42 (amended same day), D-64; marker words only on D-4, D-19..D-22, D-35, D-43,
  D-61, D-62, D-78, D-84.
- Actor `unrecorded` though the text names the owner in another form ("the owner accepted/asked", "owner
  requests"): D-43, D-48, D-64, D-67, D-69, D-71, D-78, D-79, D-80, D-83, D-84. A record's actor is fixed at import;
  a confirmed one is an amendment (`context-engine record amend`), never an edit.
- Synthesised titles to check: D-11, D-15, D-25 (weakest), and the rest of D-1..D-30, D-45, D-46.
