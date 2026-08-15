# Proof: floor-mod sign identities, quadrant (in)equivalence, and the far-edge rule

> Provenance: generated 2026-08-15 by an independent Claude (Fable) subagent given only the
> floor-mod definition and the conjectures — no access to this repo's code, slides, or prior
> numerics. Key results: (A1) only the double flip is an exact negation; (Theorem B) the four
> quadrants form exactly two equivalence classes {Q1,Q3} and {Q2,Q4} — the quadrants are
> **provably not all equivalent** (topological invariant: closed level-set components exist in
> Q2/Q4 but not Q1/Q3); (Theorem C) the two-far-edges rule nevertheless holds in ALL four
> quadrants, with Q2 requiring its own direct proof because transfer from Q1 fails.
> See `proof-two-edge-reduction.md` and `v3-modulo-design-notes.md`.

---

# The floor‑mod `f(x, y) = x − y·⌊x/y⌋`: sign identities, level‑set geometry, and images of rectangles

Throughout, `⌊t⌋` is the floor, `⌈t⌉` the ceiling, and for real `x`, `y ≠ 0`

> **f(x, y) = x mod y = x − y·⌊x/y⌋.**

We write "`y | x`" for the statement `x/y ∈ ℤ` (exact real divisibility). Python's `%` on floats implements exactly this `f`; a spot check on 100,000 random pairs found `f(x,y)` and `x % y` identical.

## 0. Preliminaries

**Lemma 0.1 (range and sign).** If `y > 0` then `f(x,y) ∈ [0, y)`; if `y < 0` then `f(x,y) ∈ (y, 0]`. In particular `f` always has the sign of `y` or is `0`, and `f(x,y) = 0 ⟺ y | x`.

*Proof.* By definition of floor, `⌊x/y⌋ ≤ x/y < ⌊x/y⌋ + 1`. Subtract `⌊x/y⌋` and multiply by `y`: for `y > 0` this gives `0 ≤ x − y⌊x/y⌋ < y`; for `y < 0` the inequalities reverse, giving `y < x − y⌊x/y⌋ ≤ 0`. The zero case: `f(x,y) = 0 ⟺ x/y = ⌊x/y⌋ ∈ ℤ`. ∎

**Lemma 0.2 (periodicity).** `f(x + ky, y) = f(x, y)` for every `k ∈ ℤ`, since `⌊x/y + k⌋ = ⌊x/y⌋ + k`. Consequently `f(x, y) = f(f(x,y), y)` and, more generally, `x ≡ x′ (mod y) ⟹ f(x,y) = f(x′,y)`.

**Lemma 0.3 (homogeneity of degree 1).** For every real `λ ≠ 0`, `f(λx, λy) = λ·f(x, y)`.

*Proof.* `(λx)/(λy) = x/y`, so `f(λx, λy) = λx − λy⌊x/y⌋ = λ(x − y⌊x/y⌋)`. ∎

Note this holds for **negative** `λ` too, because the quotient `x/y` — the argument of the floor — is unchanged.

**Lemma 0.4 (floor under negation).** For `t ∈ ℝ`: `⌊−t⌋ = −⌈t⌉`; explicitly, `⌊−t⌋ = −⌊t⌋` if `t ∈ ℤ` and `⌊−t⌋ = −⌊t⌋ − 1` if `t ∉ ℤ`.

*Proof.* If `t ∈ ℤ` this is trivial. Otherwise write `t = ⌊t⌋ + θ` with `θ ∈ (0,1)`; then `−t = (−⌊t⌋ − 1) + (1 − θ)` with `1 − θ ∈ (0,1)`, so `⌊−t⌋ = −⌊t⌋ − 1 = −⌈t⌉`. ∎

---

## Part A — Exact sign identities

**Theorem A.** For all real `x` and `y ≠ 0`:

**(A1) Double flip is exact negation:**
```
f(−x, −y) = −f(x, y).
```

**(A2) Flipping x alone:**
```
f(−x, y) = y·⌈x/y⌉ − x  =  { 0            if y | x,
                            { y − f(x, y)  if y ∤ x.
```

**(A3) Flipping y alone:**
```
f(x, −y) = −f(−x, y)  =  { 0            if y | x,
                          { f(x, y) − y  if y ∤ x.
```

*Proof.*

(A1) is Lemma 0.3 with `λ = −1`: `f(−x,−y) = −x + y⌊x/y⌋ = −f(x,y)`.

(A2): `f(−x, y) = −x − y⌊−x/y⌋ = −x + y⌈x/y⌉` by Lemma 0.4. If `y | x`, `⌈x/y⌉ = x/y` and the value is `0`. If `y ∤ x`, `⌈x/y⌉ = ⌊x/y⌋ + 1`, so `f(−x,y) = −x + y⌊x/y⌋ + y = y − f(x,y)`.

(A3): applying (A1) to the pair `(−x, y)` gives `f(x, −y) = f(−(−x), −y) = −f(−x, y)`; now substitute (A2). ∎

**Which sign changes are a simple negation?** Only the **simultaneous** flip `(x, y) ↦ (−x, −y)`: identity (A1) holds with no exceptions. Flipping a **single** argument is *not* a negation: it is the "mod‑`y` complement" `v ↦ y − v` (equivalently `f(−x,y) = f(−f(x,y), y)` by Lemma 0.2), which coincides with `−f(x,y)` only on the zero set. Counterexample: `f(7,3) = 1`, but `f(−7,3) = 2 ≠ −1` and `f(7,−3) = −2 ≠ −1`, while `f(−7,−3) = −1` as (A1) predicts.

**Case x > 0, y > 0.** Here `f(x,y) ∈ [0, y)` and

```
f(−x, y) = y − f(x, y)   if y ∤ x   (equivalently, if f(x,y) ≠ 0),
f(−x, y) = 0             if y | x   (NOT  y − 0 = y).
```

The divisibility case must be split off: the complement map `v ↦ y − v` sends `[0, y)` to `(0, y]`, and the exceptional value `y` (which `f` can never take) is folded back to `0` exactly when `y | x`. The single closed form valid in both cases is `f(−x, y) = y⌈x/y⌉ − x`. (Numerical check: `f(6,3) = 0` and `f(−6,3) = 0`, not `3`; identities (A1)–(A3) held with error exactly `0` on 200,000 random pairs.)

---

## Part B — Level‑set geometry per quadrant

Write `Q₁ = {x>0, y>0}`, `Q₂ = {x<0, y>0}`, `Q₃ = {x<0, y<0}`, `Q₄ = {x>0, y<0}`.

### B.1 Wedge decomposition

For fixed integer `k`, the set `{⌊x/y⌋ = k}` is `{ky ≤ x < (k+1)y}` when `y > 0`, and `{(k+1)y < x ≤ ky}` when `y < 0` (multiplying `k ≤ x/y < k+1` by `y` reverses the inequalities). These are **wedges between consecutive rays through the origin**, each wedge containing exactly one of its two boundary rays — always the ray `x = ky`, on which `f = 0`. On each wedge, `f(x,y) = x − ky` is affine, so its level sets are parallel straight lines.

**Quadrant Q₁ (x>0, y>0).** Here `x/y > 0`, so `k = n ≥ 0`.
- Wedges: `Wₙ = {ny ≤ x < (n+1)y, y > 0}`, `n ≥ 1`, and `W₀ = {0 < x < y}`.
- On `Wₙ`: `f = x − ny`, values sweeping `[0, y)`.
- Level lines `x = ny + c`: **slope `dy/dx = 1/n`** for `n ≥ 1` (direction `(n,1)`), **vertical lines `x = c`** for `n = 0`. Each family is parallel to its wedge's own zero ray.
- Zero set: the rays `x = ny`, `y > 0`, `n = 1, 2, 3, …` (slopes `1, 1/2, 1/3, …` accumulating on the positive `x`‑axis).
- `f ≥ 0`, and `f(Q₁) = [0, ∞)` (for `c > 0` take `f(c, c+1) = c`).

**Quadrant Q₂ (x<0, y>0).** Here `x/y < 0`, so `k = −m`, `m ≥ 1`.
- Wedges: `W′ₘ = {−my ≤ x < (1−m)y, y > 0}` (e.g. `W′₁ = {−y ≤ x < 0}`).
- On `W′ₘ`: `f = x + my ∈ [0, y)`.
- Level lines `x = −my + c`: **slope `dy/dx = −1/m ∈ [−1, 0)`**. There is **no vertical family** and no slope outside `[−1,0)`.
- Zero set: rays `x = −my`, `y > 0`, `m ≥ 1`.
- `f ≥ 0`, `f(Q₂) = [0, ∞)`.

**Quadrant Q₃ (x<0, y<0).** `x/y > 0`, `k = n ≥ 0`.
- Wedges: `{(n+1)y < x ≤ ny, y < 0}`, `n ≥ 1`, and `{y < x < 0}` for `n = 0`.
- On wedge `n`: `f = x − ny ∈ (y, 0]`; level lines have **slope `1/n`**, vertical for `n = 0`.
- Zero set: rays `x = ny`, `y < 0`, `n ≥ 1`. `f ≤ 0`, `f(Q₃) = (−∞, 0]`.

**Quadrant Q₄ (x>0, y<0).** `x/y < 0`, `k = −m ≤ −1`.
- Wedges: `{(1−m)y < x ≤ −my, y < 0}` (e.g. `{0 < x ≤ −y}` for `m = 1`).
- On wedge `m`: `f = x + my ∈ (y, 0]`; level lines have **slope `−1/m`**, never vertical.
- Zero set: rays `x = −my`, `y < 0`, `m ≥ 1`. `f ≤ 0`, `f(Q₄) = (−∞, 0]`.

In every quadrant `f` is continuous exactly off the zero rays; crossing a zero ray, the lateral limit from the neighboring wedge is `y`, so `f` jumps by `|y|` (it is one‑sidedly continuous from the wedge that contains the ray).

### B.2 Mapping each quadrant onto Q₁ via Part A

- **Q₃ → Q₁ and Q₄ → Q₂, antipodal map `σ(x,y) = (−x,−y)`.** By (A1), `f∘σ = −f` **exactly**: `σ` maps wedge `n` to wedge `n`, zero rays to zero rays, and the level set `{f = c}` onto `{f = −c}`, point by point. The `y<0` pictures are the point reflections of the `y>0` pictures with all values negated.
- **Q₂ → Q₁, reflection `ρ(x,y) = (−x, y)`.** `ρ` maps the zero ray `x = −my` to the zero ray `x = my` and the open wedge `int W′ₘ` to `int W_{m−1}`, but by (A2) it transforms values by `v ↦ y − v` (off the zero rays). Concretely, the Q₂ level line `x = −my + c` maps to the line `x = my − c`, which is parallel to the **upper** boundary ray of `W_{m−1}` and along which `f = y − c` **varies with `y`** — not a level set. Each wedge carries two natural rulings (lines parallel to its lower ray = level lines of `f`; lines parallel to its upper ray = level lines of `y − f`), and `ρ` swaps them.
- **Q₄ → Q₁, reflection `(x, y) ↦ (x, −y)`.** By (A3) values transform by `v ↦ v − y`; same phenomenon.

### B.3 Which quadrants are genuinely equivalent

**Definition.** Quadrants `Q, Q′` are *equivalent* if there is a homeomorphism `φ: Q → Q′` with `f∘φ = f` (value‑preserving) or `f∘φ = −f` (value‑negating). Such a `φ` carries each level set `{f=c} ∩ Q` homeomorphically onto `{f=±c} ∩ Q′`. (Continuity is essential; see the Remark.)

**Theorem B.** The equivalence classes are exactly `{Q₁, Q₃}` and `{Q₂, Q₄}`. That is: `Q₁ ≃ Q₃` and `Q₂ ≃ Q₄` via the value‑negating (indeed linear) map `σ`; **no other pair is equivalent.**

*Proof.*

**(a) The two equivalences.** `σ(x,y) = (−x,−y)` is a linear homeomorphism swapping `Q₁ ↔ Q₃` and `Q₂ ↔ Q₄`, and `f∘σ = −f` by (A1).

**(b) Sign obstruction.** On `Q₁, Q₂` we have `f ≥ 0` with value `1` attained; on `Q₃, Q₄`, `f ≤ 0` with value `−1` attained. Hence a **value‑preserving** map cannot join a `y>0` quadrant to a `y<0` quadrant (the point with `f = 1` has no target), and a **value‑negating** map cannot join two quadrants with the same sign of `y`. So for `{Q₁,Q₂}` and `{Q₃,Q₄}` only value‑preserving maps are conceivable, and for `{Q₁,Q₄}` and `{Q₂,Q₃}` only value‑negating maps.

**(c) The closed‑component invariant: `Q₁ ≄ Q₂` (value‑preserving).** Fix any `c > 0` (`c` lies in both images). Solving within wedges:

- `{f = c} ∩ Q₁ = ⋃_{n≥0} ℓₙ`, where `ℓₙ = {(nt + c, t) : t > c}` (the constraint `ny ≤ ny+c < (n+1)y` is exactly `y > c`).
- `{f = c} ∩ Q₂ = ⋃_{m≥1} ℓ′ₘ`, where `ℓ′ₘ = {(c − mt, t) : t > c}`.

These open rays lie on pairwise non‑parallel lines meeting only at points with `t ≤ c`, so their closures are pairwise disjoint; hence the `ℓₙ` (resp. `ℓ′ₘ`) are precisely the connected components of the level set.

Now compare their closures **inside the open quadrant**. The ray `ℓₙ` has the limit point `((n+1)c, c)`, which lies in `Q₁` (on the zero ray `x = (n+1)y`) but not in `ℓₙ`: so **no** component of `{f=c} ∩ Q₁` is closed in `Q₁`. The ray `ℓ′ₘ` has the limit point `((1−m)c, c)`; for `m ≥ 2` this lies in `Q₂`, but for `m = 1` it is `(0, c) ∉ Q₂`: so `ℓ′₁` **is** closed in `Q₂`.

A value‑preserving homeomorphism `φ: Q₁ → Q₂` would map `{f=c} ∩ Q₁` onto `{f=c} ∩ Q₂`, carrying components onto components and preserving closedness relative to the ambient quadrant (a homeomorphism of the quadrants maps relatively closed sets to relatively closed sets). Then `φ⁻¹(ℓ′₁)` would be a closed component of `{f=c} ∩ Q₁` — contradiction. With (b), `Q₁ ≄ Q₂`.

**(d) The remaining pairs, by composing with `σ`.** If `φ: Q₃ → Q₄` were value‑preserving, then `σ∘φ∘σ: Q₁ → Q₂` would be value‑preserving (the two negations from (A1) cancel), contradicting (c). If `φ: Q₁ → Q₄` were value‑negating, `σ∘φ: Q₁ → Q₂` would be value‑preserving — contradiction. If `φ: Q₂ → Q₃` were value‑negating, `σ∘φ: Q₂ → Q₁` would be value‑preserving — contradiction. Together with (b), no pair other than `{Q₁,Q₃}` and `{Q₂,Q₄}` is equivalent. ∎

Two supplementary observations:

- **The Part‑A single‑flip maps fail already on rigid grounds:** `Q₁` and `Q₃` contain a family of *vertical* level lines (wedge `n = 0`), while every level line of `Q₂` and `Q₄` has slope in `[−1, 0)`. A reflection sends vertical lines to vertical lines, so `ρ(x,y) = (−x,y)` and `(x,y) ↦ (x,−y)` cannot match level sets to level sets — consistent with the value law `v ↦ y − v` of (A2)/(A3), which is genuinely `y`‑dependent, hence neither `+f` nor `−f`.
- **Remark (why continuity is required).** If arbitrary discontinuous bijections were allowed, the distinction would collapse: by periodicity (Lemma 0.2) the piecewise shear `φ(x,y) = (x + (2m−1)y, y)` on each `W′ₘ` is a value‑preserving *bijection* `Q₂ → Q₁` (it sends `W′ₘ` onto `W_{m−1}`). "Genuinely equivalent" therefore means equivalent through a homeomorphism — a class that contains the rigid maps of Part A, which already realize the two true equivalences.

---

## Part C — Image of `f` over a rectangle

Let `Q = [x₀, x₁] × [y₀, y₁]` lie in one open quadrant. Define the **far edges**:

- `E_x = {x_far} × [y₀, y₁]`, where `x_far ∈ {x₀, x₁}` has the larger `|x|`;
- `E_y = [x₀, x₁] × {y_far}`, where `y_far ∈ {y₀, y₁}` has the larger `|y|`.

Per quadrant, `(x_far, y_far)` is: `Q₁: (x₁, y₁)` (right, top); `Q₂: (x₀, y₁)` (left, top); `Q₃: (x₀, y₀)` (left, bottom); `Q₄: (x₁, y₀)` (right, bottom).

**Theorem C.** In **every** quadrant the conjecture is **true**:
```
f(Q) = f(E_x) ∪ f(E_y).
```
No per‑quadrant correction is needed: the correct pair is always the two edges furthest from the origin.

*Proof.* The inclusion `⊇` is trivial (`E_x ∪ E_y ⊆ Q`). For `⊆` we slide each point along its level line until it exits `Q`; the key is that in each quadrant the level lines can be traversed *toward the two far edges simultaneously*.

**Quadrant Q₁** (`0 < x₀ ≤ x₁`, `0 < y₀ ≤ y₁`). Let `(x, y) ∈ Q`, `n = ⌊x/y⌋ ≥ 0`, `c = f(x,y) = x − ny ∈ [0, y)`.

*Case `n = 0`:* then `c = x < y ≤ y₁`, so `⌊x/y₁⌋ = 0` and `f(x, y₁) = x = c`, attained at `(x, y₁) ∈ E_y`.

*Case `n ≥ 1`:* consider `γ(t) = (nt + c, t)`. Whenever `t > c` we have `(nt+c)/t = n + c/t ∈ [n, n+1)`, so `f(γ(t)) = c`. Set `t* = min(y₁, (x₁ − c)/n)`. Since `y = (x − c)/n ≤ (x₁ − c)/n` and `y ≤ y₁`, we get `t* ≥ y > c`. For `t ∈ [y, t*]` the point `γ(t)` stays in `Q`: its second coordinate lies in `[y, y₁]` and its first coordinate increases from `x ≥ x₀` to `nt* + c ≤ x₁`. At `t = t*`, either `t* = y₁` and `γ(t*) ∈ E_y` (top), or `nt* + c = x₁` and `γ(t*) ∈ E_x` (right). Either way `c ∈ f(E_x) ∪ f(E_y)`.

**Quadrant Q₂** (`x₀ ≤ x₁ < 0 < y₀ ≤ y₁`). Let `⌊x/y⌋ = −m`, `m ≥ 1`, `c = x + my ∈ [0, y)`. Consider `γ(t) = (c − mt, t)`; whenever `t > c`, `(c − mt)/t = c/t − m ∈ [−m, 1−m)`, so `f(γ(t)) = c`. Set `t* = min(y₁, (c − x₀)/m)`; since `x ≥ x₀` gives `y = (c − x)/m ≤ (c − x₀)/m`, again `t* ≥ y > c`. For `t ∈ [y, t*]`, the second coordinate stays in `[y, y₁]` and the first coordinate *decreases* from `x ≤ x₁` to `c − mt* ≥ x₀`, so `γ(t) ∈ Q`. At `t = t*` the point is on the top edge (`t* = y₁`) or the **left** edge `x = x₀` — exactly the far edges of `Q₂`. (No vertical‑line case arises: `m ≥ 1` always.)

**Quadrants Q₃ and Q₄, by transfer along the genuine equivalences.** If `Q ⊂ Q₃`, then `−Q = [−x₁, −x₀] × [−y₁, −y₀] ⊂ Q₁` is a rectangle whose far edges are exactly `−E_x` and `−E_y` (the right edge of `−Q` is `x = −x₀` and `|x₀| ≥ |x₁|`; likewise for `y`). Using (A1) in the set form `f(−S) = −f(S)` twice and the Q₁ case:
```
f(Q) = −f(−Q) = −( f(−E_x) ∪ f(−E_y) ) = f(E_x) ∪ f(E_y).
```
Identically, the Q₄ statement follows from the Q₂ statement via `σ`. ∎

**Three comments.**

1. **The near edges genuinely fail** (the conjecture is not vacuous). For `Q = [3,6] × [4,5] ⊂ Q₁`, Theorem C gives `f(Q) = f(E_x) ∪ f(E_y)` with `f(E_x) = {6 − y : y ∈ [4,5]} = [1,2]` and `f(E_y) = [0,1] ∪ [3,5)`, i.e. `f(Q) = [0,2] ∪ [3,5)`; the near pair yields only `{3} ∪ [0,2] ∪ [3,4) = [0,2] ∪ [3,4)`, missing all of `[4,5)` — e.g. `f(4.5, 5) = 4.5`. (Neither far edge suffices alone: `1.5` comes only from `E_x`, `4` only from `E_y`.)
2. **The proof does not transfer from Q₁ to Q₂ through Part A**, consistent with Theorem B: the reflection `(x,y) ↦ (−x,y)` maps rectangles to rectangles and far edges to far edges, but it transforms values by `v ↦ y − v` with `y` varying over the rectangle, so image sets do not transform. Q₂ needs its own (mirrored) argument — which succeeds because the sign flip of the level‑line slopes (`+1/n` to `−1/m`) occurs *together with* the far `x`‑edge moving from `x₁` to `x₀`. Transfer is legitimate exactly along the equivalences `{Q₁,Q₃}` and `{Q₂,Q₄}` of Part B.
3. Degenerate rectangles (`x₀ = x₁` or `y₀ = y₁`) are covered by the same proofs.

**Numerical spot checks** (Python, whose `%` is exactly this `f`). Across 400 random rectangles in all four quadrants (120,000 random points total), every landing point lay on a far edge inside `Q` with `|f(landing) − f(x,y)| ≤ 1.8·10⁻¹⁵`. In `Q₁`, for `Q = [3,6]×[4,5]`, 250,000 random samples of `f(Q)` all lay within `3.8·10⁻⁶` of the sampled far‑edge values (`x=6`, `y=5`), while the near‑edge pair missed values by up to `0.994`. In `Q₂`, for `[−6,−3]×[4,5]`, the far pair (`x=−6`, `y=5`) matched to within `4.4·10⁻⁶`, the near pair missed by up to `0.995`. In `Q₃`, for `[−6,−3]×[−5,−4]`, the far pair (`x=−6`, `y=−5`) matched to within `3.8·10⁻⁶`, the near pair missed by up to `0.999`. In `Q₄`, for `[3,6]×[−5,−4]`, the far pair (`x=6`, `y=−5`) matched to within `4.5·10⁻⁶`, the near pair missed by up to `0.997`.

---

## Summary

- **A.** `f(−x,−y) = −f(x,y)` always (degree‑1 homogeneity at `λ = −1`); single flips are *not* negations: `f(−x,y) = y⌈x/y⌉ − x`, i.e. `y − f(x,y)` off multiples and `0` on them, and `f(x,−y) = −f(−x,y)`.
- **B.** Each quadrant decomposes into wedges `{⌊x/y⌋ = k}` between rays through the origin; `f` is affine on each, with level lines parallel to the wedge's zero ray, of slope `1/n` (vertical for `n=0`) in `Q₁, Q₃` and `−1/m` in `Q₂, Q₄`. Exactly `Q₁ ≃ Q₃` and `Q₂ ≃ Q₄` (value‑negating antipodal map); all cross pairs are inequivalent, by a sign obstruction plus a topological invariant: for `c > 0`, `{f = c}` has a closed connected component in `Q₂` (and `Q₄`) but none in `Q₁` (or `Q₃`).
- **C.** The conjecture is **true in all four quadrants**: `f(Q)` equals the image of the two edges furthest from the origin — (right, top) in `Q₁`, (left, top) in `Q₂`, (left, bottom) in `Q₃`, (right, bottom) in `Q₄` — proved by sliding along level lines in `Q₁` and `Q₂`, and transported to `Q₃`, `Q₄` by the antipodal identity; the near‑edge pair fails in general.
