# v3 interval-modulo: design notes (2026-08-15)

Session findings consolidating: bug audit of the current code, proven theory, a validated
replacement algorithm, and the remaining open items. Companion files:
`proof-two-edge-reduction.md` (Q1 constructive proof),
`proof-sign-symmetries-quadrants.md` (sign identities, quadrant inequivalence, all-quadrant
far-edge rule), `modulo_v3_prototype.py` (runnable validated prototype + oracle).

North star (owner's definition): `[a] mod [b]` = the closure over every possible value of
`a mod b` for `a ∈ A`, `b ∈ B`, under Python floor-mod semantics.

## 1. State of the current code (`multi_interval.py`, master @ f5a7910)

- `__mod__` (line 1439): **correct** for `non-negative finite MultiInterval % positive scalar`.
  Fuzzed 300 random cases × all 4 open/closed combos × 300 sample points: 0 soundness failures.
  Everything else (`interval % interval`, negatives, `__rmod__`, `__divmod__`) returns
  `NotImplemented`.
- `__modulo` (line 1165): the ~270-line geometric sweep engine is **dead code** (name-mangled,
  only self-referenced) and **substantially broken** — but see section 1b: the *geometry is
  correct*, the defects are mechanical.
  Symptoms: emits `[0, m]` where `[0, m)` is correct; collapses disjoint unions
  (`[12,18.7] mod [7.5]` → `[0,7.5]` instead of `{[0,3.7], [4.5,7.5)}`); unsound in at least
  one family (`[3,7.9] mod [7.9,12.6]` = `[3,7.9]`, but `7.9 % 7.9 = 0` is missing).
  Full diagnosis in section 1b.
- Unrelated: `__repr__` returns `NotImplemented` (a non-string), so `repr()` on any
  MultiInterval raises `TypeError`. `__str__` works, which is why it went unnoticed.
- `__floordiv__` (line 1142) discards openness: `[1,2) // 1` = `[1,2]`, should be `[1]`.
  (README claims floordiv is unimplemented; it is implemented, incorrectly.)

## 1b. `__modulo` diagnosis: the geometry is correct, the defects are mechanical

Differential test of `__modulo` against `modulo_v3_prototype.py`, 1499 random positive cases
with random open/closed flags:

    ORIGINAL disagrees: 278/1499 (18.5%)
    PATCHED  disagrees:  48/1499 ( 3.2%)     <- five localized fixes below
       wrong shape (interval locations differ):   0
       closure-only (same locations, wrong flags): 48

**What is already right.** `_second_end` (= y1) is the far B edge and `_first_end` (= x1) the
far A edge — exactly the proven two-edge reduction. `z = x1/(1 + x1//y1)` and the `2 +` variant
are the prototype's `z1`, `z2`. The code never computes `b = floor(x1/y0)`; instead
`z1 >= _second_start` and `z2 >= _second_start` are used, which are algebraically identical to
`b >= a+1` and `b >= a+2` (cheaper, and correct). Line 1211 is exactly the k=0 closure
exception of section 3b; lines 1216-1217 are exactly corner-touch 1 of the zero-classification.

**The five defects** (line numbers at master `f5a7910`):

1. **Axis conflation — 10 sites**: 1248, 1257, 1306, 1315, 1360, 1369, 1402, 1410, 1420, 1428.
   Each appends `_second_start` (the divisor coordinate y0) where the *output value*
   `_first_end % _second_start` (= x1 % y0) belongs. `[1,2] mod [3,4]` → `[1,3]` not `[1,2]`.
   **Line 1396 looks identical but is CORRECT** — there `y0 == z1` exactly, so the raw
   coordinate is the right answer. Do not blanket-replace.
2. **`break` should be `continue` — 15 sites** in 1204-1437. They exit the inner loop over the
   *divisor's* pieces; since every path breaks, the inner loop always runs exactly once and
   only `other`'s first sub-interval is processed. `[3,7] mod ({4} u {5})` → `[0,4]`, should
   be `[0,5)`.
3. **Hardcoded closed epsilon** at 1342 and 1394: `append((_first_end, 0))` ignores
   `_first_end_epsilon`.
4. **Missing corner-graze zero** in the `_first_end % _second_start == 0` branch (1377): a zero
   line grazes corner (x1,y0), so 0 belongs in the output iff x1 and y0 are both closed.
   Nothing is emitted — this is the `[3,7.9] mod [7.9,12.6]` unsoundness.
5. **Missing exit on that same branch** — the only branch without a `break`. It falls through
   into the "there are no zero-line intersections" block despite an intersection existing, and
   emits an inverted (start > end) pair. Defect 1 was masking this; fixing 1 alone surfaces it
   as an AssertionError. Note `merge_adjacent` sorts *pairs*, so it cannot repair an inverted
   pair, and its `len == 2` fast path (lines 845-847) asserts sortedness without sorting.

**The residual 48 are not typos.** Every one is closure-only, and they are the k=0 /
cross-sector cases of section 3b — e.g. `[1.07,4.09] mod (8.58,14.03)` must be `[1.07,4.09]`
fully closed (max(A) < min(B), so `x % y = x` is attained along whole vertical edges) but the
code gates that endpoint on `_second_start_epsilon == 0`. No local epsilon rule can fix this;
it is the structural limit that motivated shape-then-attainment. Fixing the five defects AND
replacing epsilon propagation with the attainment test would make the sweep fully correct —
if the geometry engine is kept at all.

## 2. Proven theory

- **Two-edge reduction** (proof-two-edge-reduction.md, Q1; proof-sign-symmetries-quadrants.md
  Thm C, all quadrants): for a rectangle in one open quadrant,
  `f(Q) = f(E_x) ∪ f(E_y)` where E_x, E_y are the edges **furthest from the origin**
  (larger |x| endpoint of A, larger |y| endpoint of B). Constructive witness in sector k:
  top-edge point `(v + k·y_far, y_far)` if it fits, else right-edge point
  `(x_far, (x_far − v)/k)` — the latter is the slides' `z = x1/(1 + floor(x1/y1))` family.
- **Near edges genuinely fail**: `[3,6]×[4,5]` — near pair misses all of `[4,5)`.
- **Sign identities** (Thm A): only the double flip is exact: `(−x) mod (−y) = −(x mod y)`.
  Single flips are the y-dependent complement `v ↦ y − v` (with `y | x ⇒ 0`), NOT a negation.
- **Quadrant inequivalence** (Thm B): equivalence classes are exactly {Q1, Q3} and {Q2, Q4}
  (antipodal map). Q1 ≄ Q2 provably: Q1/Q3 have vertical level lines (k=0 wedge), Q2/Q4 slopes
  all in [−1,0); topologically, for c>0 the level set {f=c} has a closed component in Q2/Q4,
  none in Q1/Q3. Consequence: the owner's intuition was right — the quadrants are NOT all
  equivalent — yet the far-edge rule holds in all four because the slope flip co-occurs with
  the far x-edge swapping sides. Q2 needs its own primitive derivation; Q3/Q4 come free via
  the antipodal identity applied to Q1/Q2 results.

## 3. Validated v3 algorithm (positive quadrant; prototype in modulo_v3_prototype.py)

Replace the 2-D sweep with two 1-D primitives plus a decoupled closure pass:

```
A mod B  =  (A mod y_far)  ∪  (x_far mod B)          # locations
```

- **P1: interval mod scalar** — already shipped as `__mod__`; case on
  `n = floor(x1/m) − floor(x0/m)`: n=0 → `[x0%m, x1%m]`; n=1 → `[0, x1%m] ∪ [x0%m, m)`;
  n≥2 → `[0, m)`.
- **P2: scalar mod interval** — `c mod [y0,y1]`, `a = floor(c/y1)`, `b = floor(c/y0)`:
  - `a == b`  → `[c%y1, c%y0]`
  - `b == a+1` → `[0, c%y0] ∪ [c%y1, z1)`
  - `b ≥ a+2` → `[0, z2) ∪ [c%y1, z1)`
  with `z1 = c/(a+1)`, `z2 = c/(a+2)` (slide 15's formulas). Special case `a == 0`
  (`y1 > c`): the right piece degenerates to the point `{c}`.
- **Closure is decided separately, not propagated.** Epsilon-propagation through the edge
  formulas is *provably insufficient*: with both operands open,
  `(11.75,15.05) mod (3.56,5.52)` attains `4.01` via interior points `14.01 % 5` even though
  both edge pieces are open there — a spurious hole. Instead:
  1. Compute locations with all endpoints treated as closed (correct shape incl. gaps).
  2. For each of the ≤4 resulting endpoints v, test attainment directly:
     `∃ k ≥ 0, y ∈ B ∩ (v, ∞): v + k·y ∈ A` — one interval-intersection per k.
  Validation: 3997 random cases × random open/closed flags on both operands, checked against
  an exact attainment oracle: **0 soundness failures, 0 endpoint-closure faults**
  (epsilon-propagation scored 604 closure faults on the same suite).
- Complexity: shape is O(1). The attainment loop is O(x1/y0) worst case; each endpoint comes
  from a known (edge, k) pair, so passing k through should make it O(1) — **not yet verified**.

## 3b. Corner and zero-touch closure semantics (verified 2026-08-15)

Owner's conjectured rules: (i) a corner value is included iff both neighboring edges are
included; (ii) if the only zero line touches the rectangle exactly at a corner, 0 is excluded
unless that corner is genuinely in A×B, even if the corner lies along an inclusive edge.
Both are correct in the generic case; the precise statements follow.

**Zero-touch classification (Q1, non-degenerate A, B).** Zero line `x = k·y` meets the closed
rectangle iff `x0/y1 ≤ k ≤ x1/y0`. The intersection is a *single point* iff `k = x0/y1`
exactly (touch at corner `(x0, y1)`) or `k = x1/y0` exactly (touch at corner `(x1, y0)`) —
these are the only tangential corners. A line through `(x0, y0)` or `(x1, y1)` always
continues into the open interior (single-point touch there is impossible when A and B are
non-degenerate). Hence:

    0 ∈ A mod B  ⟺  ∃ integer k strictly between x0/y1 and x1/y0     (unconditional)
                  ∨ (x0/y1 ∈ ℤ  ∧  x0 ∈ A  ∧  y1 ∈ B)                 (corner touch 1)
                  ∨ (x1/y0 ∈ ℤ  ∧  x1 ∈ A  ∧  y0 ∈ B)                 (corner touch 2)
                  ∨ (0 ∈ A)                                            (k = 0)

Refinements of rule (ii): both corner touches can coexist (`[3,4] mod [2,3]`: 0 iff either
corner included — an OR, not a single condition), and far-corner alignment (`x1/y1 ∈ ℤ`)
gives 0 *unconditionally* because that line always crosses the interior.

**Endpoint closure rules for the two-edge output (Q1):**
- `y1` (sup-B endpoints of `[…, y1)` pieces): **always open** — residues ≥ sup B are
  unattainable (`x mod y < y ≤ y1`).
- `z`-points `x1/(a+j)`: open from their own piece (own-sector supremum, jumps to 0 at the
  crossing); closed only via coincidental witnesses elsewhere.
- Corner values `x0%y1`, `x1%y0`, `x1%y1`: attained *at the corner* iff both operand
  endpoints closed — rule (i). Exceptions where the value has non-corner witnesses:
  - **k=0 sector** (level lines vertical): `[1,2] mod (3,4)` = `[1,2]` — corner values are
    attained along entire vertical edges, so only the A-side flag matters, B's flags are
    irrelevant. More generally the max needs only `x1 ∈ A` whenever B has any point > x1
    (e.g. `[1,3] mod [3,4]`: max 3 closed iff `x1 ∈ A`, despite `⌊x1/y0⌋ = 1`).
  - **Cross-sector coincidence**: the corner value may independently be attained in another
    wedge of the same rectangle; the attainment test enumerates all k and catches this.
- Degenerate output pieces (e.g. the `{0}` from a corner touch, or `{x1}` when `a = 0`) must
  be **dropped entirely** when their single value is unattainable — a phantom
  `(v, open, v, open)` is not an interval.

**Implementation caveat (multi-piece operands):** closure/attainment must be tested against
the FULL multi-intervals A and B, not per sub-rectangle — a value open in one A-piece ×
B-piece product can be closed via a witness in another.

Verification: 7 alignment geometries (lone touch at `(x0,y1)`; double touch; interior
crossing + far-corner alignment; pure k=0; `x1 = y0`; detached `{0}` point; far-corner +
generic corner + k=0 endpoint combined) × all 16 open/closed flag combinations = 112 cases:
prototype output matched hand-derived truth tables exactly, every endpoint closure matched an
independent exact-Fraction attainment oracle, and flag-respecting sampling found no missing
values. Suite persisted in `modulo_v3_prototype.py` (`__main__`); the 3997-case random fuzz
still passes (0 soundness / 0 closure faults) after adding the phantom-point filter.

## 3c. Implementation status (as of 2026-08-15)

**Branch `fix-interval-modulo` (uncommitted):** the five defects of section 1b are fixed,
epsilon propagation is replaced by the attainment test, and `__mod__` dispatches to
`__modulo` when `self.is_positive and other.is_positive` (strictly positive, zero excluded on
both sides — `is_positive` is `endpoints[0] > (0, 0)`). Verified: **0/2498 disagreements**
with the prototype through the `%` operator, **0/112** on the corner/zero-touch suite, scalar
path and other operators unchanged, multi-piece operands correct
(`[3,7] % ({4} u {5}) = [0,5)`).

How the closure pass is wired in `__modulo`: the operand epsilons are zeroed before the
geometry loop (so the case tree computes the closed hull), every emitted epsilon is forced
closed, `merge_adjacent()` runs, then each surviving endpoint is decided by
`_mod_attained(value, a_pieces, b_pieces)` against the FULL operands, with phantom degenerate
pieces dropped. `_mod_attained` and `_intervals_intersect` are module-level helpers above the
class.

**Known cost:** `_mod_attained` loops `k` up to `(x1 - value) / y0`, so it is O(x1/y0) rather
than O(1) in the worst case — a narrow interval far from the origin with a small divisor.
Measured: `[1e4,1e4+1] % [1,1.5]` 2 ms, `[1e5,...]` 15 ms, `[1e6,...]` 173 ms. Typical cases
are ~0.02 ms. The O(1) specialization (each endpoint is produced by a known `(edge, k)` pair,
so the generating `k` can be passed in instead of searched) is still unverified — see
section 4.

Still untouched on master: `__repr__` raises, `__floordiv__` drops openness.

`modulo_v3_prototype.py` is standalone and self-validating (`python modulo_v3_prototype.py`
prints `112 combos, 0 mismatches` and `3997 cases, 0 soundness failures, 0 closure faults`).

| capability | status |
|---|---|
| non-negative A, positive B, all open/closed combos | **works**, validated vs exact oracle |
| degenerate operands (`x0 == x1` and/or `y0 == y1`) | **works**, verified explicitly — needs no special-casing: degenerate B forces `a == b` in P2, degenerate A forces `n == 0` in P1, each collapsing to the other primitive. Spot-checked: `[5]%[2,3]=[0,1]∪[2,2.5)`, `[3,7]%[5]=[0,2]∪[3,5)`, `[7]%[3]=[1]`, `[6]%[3]=[0]`, `[2]%[5,9]=[2]`, `[4]%[4,7]=[0]∪[4]`. (The random fuzz essentially never generates these — keep testing them by hand.) |
| `x0 == 0` | works |
| negative operands | **not implemented** |
| operands crossing zero | **not implemented** |

Negatives/zero previously failed *silently*, which is why `mod()` now guards on
`A[0] < 0 or B[0] <= 0` and raises `NotImplementedError`. Observed pre-guard behaviour, kept
here as the regression target for whoever implements section 4:

    [-7,-3] mod [2,5]   -> ZeroDivisionError
    [3,7]   mod [-5,-2] -> {}          (silently empty; misses -1.80)
    [3,7]   mod [-2,5]  -> (0, 5)      (silently drops the negative half; misses -0.99)
    [-3,7]  mod [2,5]   -> [0, 5)      (correct only coincidentally — full range)

## 4. Remaining work

- Derive the Q2 primitive pair (negative dividend, positive divisor) in closed form; then
  Q3/Q4 = antipodal mirror. Zero-crossing operands: split A and B at 0 (≤4 sign-pure
  sub-rectangles, still O(1)), decide semantics for `x mod 0` (slide 12 suggests 0 rather
  than an error) and mod-by-interval-containing-zero.
- Port the prototype's `(lo, lo_closed, hi, hi_closed)` representation back to the
  endpoints/epsilon representation.
- Slide-12 conjecture `[x0,x1] mod [anything > x1] = [x0,x1] ∪ ([x0,x1] mod (B ∩ [0,x1]))`
  — untested.
- Fix independently: `__repr__` (one-liner), `__floordiv__` openness.
