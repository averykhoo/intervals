---
type: Decision
id: D-49
title: M13a and M14's fuzz job and oracle built
actor: unrecorded
decided: 2026-09-25
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:65721f4e645ea626f1308dd7551a702360e53ecc78f5eadf781d76e61ff60823
---

### 2026-09-25 revision: M13a and M14's fuzz job and oracle built

built across 2026-09-25 and 26 (measured 2026-09-26), the first session M13 suggests; the details
and numbers are in v2-implementation-plan.md (M13a, M14). D14 and D15 moved into "current design"
(ieee 1788, testing):
* **vendored** all 19 files of oheim/ITF1788 at `b6ee1e2` and its three licence files, byte-exact,
  blob hashes checked; reading every header corrected D15: five files carry the all-permissive
  notice (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`)
* **the parser reads every statement** (9542): a strict tokenizer, brace-counted testcases,
  `NaN`, quoted text, `{...}` lists, `signal` clauses, two-value results. the old one ended a
  testcase at a list's `}` and silently dropped 11 statements of `libieeep1788_reduction.itl`
* **divergence keys drop decorations**, so the 18 old rows are 15 keys; the `[nai]` operands are
  rows generated in code under "decoration expectations" until M13g, each failing as stale once
  its vector matches. 4767 vectors, 49 keys, 0 unknown failures
* **fuzzing is a multiplier, not a profile's count**: a test's own `max_examples` overrides any
  profile, so under `HYPOTHESIS_PROFILE=fuzz` the conftest multiplies each test's own, once per
  function; the weekly `fuzz.yml` saves the example database even on failure. its first local run
  found `test_sound_float_identity_rounding[div]` red on mixed Fraction and float operands:
  python rounds `Fraction(1,3) / 2.75` twice, the library once, as "arithmetic" says it should. the
  fault was the test oracle's, which now rounds a mixed pair once (`tests/oracles.py::_once`)
* **the flint oracle requires correct rounding**, not a tolerance: no double may lie between an end
  and the value, decided by arb's proven comparisons, more bits when undecided; values arb cannot
  hold near the float range are compared through their logs. it found no bug
