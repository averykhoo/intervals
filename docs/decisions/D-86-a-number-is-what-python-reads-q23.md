---
type: Decision
id: D-86
title: a number is what python reads (Q23)
actor: owner
decided: 2026-10-06
session: 2026-11-02a
decision_status: BUILT
imported_from: docs/evidence/decisions-normalised-2026-11-02.md
body_sha: sha256:81cb3344c370e42f402bf7cf5203b705b525944b6aeae6f0df421926321ed6b9
---

### 2026-10-06 revision: a number is what python reads (Q23)

the owner, 2026-10-06, answering Q23: "each number should be something python can parse, split by a character
that's not a valid part of the number". the text syntax of `fmt.py` (`parse`, `parse_value`; not the 1788
literals, whose grammar is the standard's):
* two numbers with no separator are refused (`[0.1.2]`, `[-2-1]`, `{1-2}` were split into two);
* ASCII digits only (python's `\d`, `int` and `float` take other scripts': `[١٢]` was 12);
* a sign is attached to its number: `- 5` and `- inf` are refused (`float('- 5')` fails);
* white space around `/` stays (`Fraction(' 1 / 3 ')` reads it); `1/-3` stays refused;
* python's digit separators are read by python's rules: `1_000`, `1_0.5`, `1e1_0`; the malformed ones refused;
* `inf`, `infinity` and `∞` stay; hex stays (D28: `repr` writes an int past 4300 digits in hex); no binary or octal;
* the owner, the same day: "if spaces can be part of a fraction then should we require commas or semicolons as
  separators": yes, white space alone no longer separates two items (`[1 2]`, `{1 2}`, `1 2` were read); between
  a set's items `,` `;` `|` `∪` stay (the session's choice: explicit, documented, never white space).
the owner also said the string syntax is a side quest ("we could totally just not allow parsing strings and be
strict about creation"), kept because it is a natural way to write a set. reverses: yesterday's `1_000` refusal
(m14b-open), and the spaced sign accepted since M14-breadth (the v1 parity audit's `[- 5, 5]` row is history)
