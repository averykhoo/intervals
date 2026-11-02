---
type: Decision
id: D-76
title: fuzz on push, not on a schedule
actor: unrecorded
decided: 2026-09-29
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:83291b523f7b5ecfb8af35f914ac0678102abe81c582d161dbcca35fb8b77a7a
---

### 2026-09-29 revision: fuzz on push, not on a schedule

* the owner: the fuzz runs "fully autonomously or not at all"; nobody reads a scheduled run's
  failure email. so `fuzz.yml` runs on every push to `master` (and `workflow_dispatch`), not weekly;
  the same fuzz run happens locally before each push (`tools/prepush.sh`), and the session that
  pushed watches both workflows to the end (`tools/ci_watch.sh`, a babysitter agent), debugging a red
  run from its log and artifact. the push procedure is in `CLAUDE.md`
* the fuzz database was never kept on CI: hypothesis loads its `ci` profile at import under
  GitHub Actions, whose database is None, and the `fuzz` profile, registered without one, inherited
  it, so the first two GitHub runs saved no example and the carried `.hypothesis` held only
  hypothesis's constants cache (their artifacts have no `examples/`). found 2026-09-29 by replaying
  the second run's artifact locally, which replayed nothing. the profile now names its database;
  `tests/test_fuzz_profile.py` pins it (red without it, under a simulated CI)
