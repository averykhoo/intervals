# Interval floor-modulo in all four quadrants: audit of the Q1 work and full derivation

> Provenance: generated 2026-09-24 by a Claude (Fable) subagent. Inputs: the three
> companion files (`v3-modulo-design-notes.md`, `proof-two-edge-reduction.md`,
> `proof-sign-symmetries-quadrants.md`) and `modulo_v3_prototype.py`. Everything marked
> **Proof** below is argued; everything marked **Checked** was verified numerically by
> `modulo_allquadrants_prototype.py` (counts at the end of this file). The prototype is the
> executable form of every boxed statement here.

Semantics throughout: `f(x, y) = x mod y = x − y·⌊x/y⌋` (Python floor-mod), operands are
pieces `(lo, lo_closed, hi, hi_closed)`, `A mod B = { f(x,y) : x ∈ A, y ∈ B }` as a *set*,
each result endpoint closed iff attained. Notation: `A = [x0, x1]`, `B = [y0, y1]` denote the
closed hulls; flags are written separately (`x0 ∈ A` means the lower endpoint is closed).
"Near" and "far" edges are measured by distance from the origin (`|x|`, `|y|`).

---

## Part 1 — Audit of the existing Q1 work

### 1.1 What holds (re-derived, not just re-read)

| claim | source | verdict |
|---|---|---|
| Two-edge reduction `f(R) = f(top) ∪ f(right)` for `0 ≤ x0 ≤ x1`, `0 < y0 ≤ y1` | proof-two-edge-reduction.md | **holds**; the proof is complete, and correctly covers `x0 = 0` (Lemma 4 never uses `x0 > 0`) and degenerate `A` or `B`. |
| A1 `f(−x,−y) = −f(x,y)`; A2/A3 single flips are `v ↦ y − v` off the zero set | proof-sign-symmetries §A | **holds**; A1 is `f(λx,λy) = λf(x,y)` at `λ = −1`. |
| Thm B: exactly `{Q1,Q3}`, `{Q2,Q4}` are equivalent | §B | **holds**; not needed for the algorithm but it correctly predicts that Q2 needs its own primitives. |
| Thm C: far-edge rule in all quadrants; far pair per quadrant `Q1 (x1,y1)`, `Q2 (x0,y1)`, `Q3 (x0,y0)`, `Q4 (x1,y0)` | §C | **holds**. The Q2 proof is written for the open quadrant (`x1 < 0`); §2.1 below extends it to `x1 = 0`. |
| P1 case table on `n = ⌊x1/m⌋ − ⌊x0/m⌋` | design notes §3 | **holds** for *any* real `x0 ≤ x1` (see §2.1): the proof is pure periodicity and never uses `x ≥ 0`. |
| P2 `c mod [y0,y1]`, `a = ⌊c/y1⌋ ≤ b = ⌊c/y0⌋`: `a=b → [c%y1, c%y0]`; `b=a+1 → [0,c%y0] ∪ [c%y1, c/(a+1))`; `b≥a+2 → [0, c/(a+2)) ∪ [c%y1, c/(a+1))`; `a=0 ⇒` right piece is `{c}` | design notes §3 | **holds**. Sector `k ≥ 1` is `y ∈ (c/(k+1), c/k]` on which `f = c − ky` decreases from `c/(k+1)` (open, not attained) to `0`; sector `0` is `y > c` with `f ≡ c`. When `b ≥ a+2`, sector `a+1` lies wholly inside `B` (`y0 ≤ c/(a+2)` and `y1 > c/(a+1)`), giving `[0, c/(a+2))`, and every lower sector's image `[0, c/(k+1))` or `[0, c%y0]` is inside it (`c%y0 < y0 ≤ c/(a+2)`). |
| `c = 0` in P2 | — | **holds**: `a = b = 0`, result `{0}`. |
| Degenerate operands need no special case (`x0 = x1 ⇒ n = 0`; `y0 = y1 ⇒ a = b`) | design notes §3c | **holds** for the *shape*. But see 1.2(a): with a degenerate operand, hull pieces that merely touch must not be merged before the closure pass. |
| `y1` (sup B) is always an open endpoint | §3b | **holds** for finite `y1` in all quadrants (`|f| < |y| ≤ |y_far|`). **Exception at infinity**: if `B` is closed at `+∞` and `A` has a negative point, `f = +∞` is attained (`−3 % inf = inf`), so the result is closed at `+∞`. See §2.6. |
| z-points `c/(a+j)` open from their own piece | §3b | **holds**: `c/(a+j)` is the supremum of sector `a+j−1` on the right edge, where `f` jumps to `0`; it can only be closed by a witness in a different sector or on the other edge. The attainment test enumerates those. |
| Corner rule (i): corner value attained *at the corner* iff both flags closed | §3b | **holds**, with the listed exceptions (k=0 sector; cross-sector coincidence). Both exceptions are covered by a full attainment test. |
| Zero-touch classification | §3b | **holds** as a disjunction. One statement in the prose is imprecise: "`x = k·y` meets the closed rectangle iff `x0/y1 ≤ k ≤ x1/y0`" is for `k ≥ 1`; the `k = 0` line `x = 0` is the *whole left edge* when `x0 = 0`, never a corner touch. That case is correctly captured by the separate disjunct `0 ∈ A`, so the classification is right; only the "single point iff …" sentence needs "`k ≥ 1`". Also: "unconditional" presumes non-degenerate `A` and `B`; for a degenerate operand the zero line meets a segment or point, whose membership in `A×B` is automatic (a non-empty degenerate piece is closed), so the disjunction still holds verbatim. |
| Quadrant boundary `x = 0`: is it in Q1? | — | Yes for the algorithm: the two-edge proof and P1/P2 hold with `x0 = 0` (and, for Q2, `x1 = 0`, §2.1). The `y = 0` boundary is never in any box (`x mod 0` undefined; divisor pieces are cut at 0, §2.4). |
| Prototype `attained(v, A, B)` | modulo_v3_prototype.py | **correct as an oracle**: `k = 0` needs `v ∈ A` and `B ∩ (v,∞) ≠ ∅`; `k ≥ 1` needs `y ∈ B ∩ (v,∞)` with `v + ky ∈ A`; its loop bound `⌊(x1−v)/y0⌋ + 1` dominates the true bound `⌊(x1−v)/max(y0,v)⌋` (§2.5). |

### 1.2 What is wrong and the correction

**(a) Merging closed hulls before the attainment pass can hide a one-point hole.**
`modulo_v3_prototype.py::shape` normalises (merges) the hull pieces, then `mod` tests only the
endpoints of the *merged* pieces. When two hull pieces touch at a single value `v` that is
attained by neither, `v` disappears into the interior of the merged piece and is reported as
present. Counterexamples (all verified against the prototype's own `attained`, which returns
`False` for `v` in each case while `mod` returns a piece containing `v`):

| operands | prototype output | true result | why `v` is unattained |
|---|---|---|---|
| `(2, 2.5) mod (1, 1.5)` (both non-degenerate, both open) | `[0, 1.25)` | `[0, 0.5) ∪ (0.5, 1.25)` | hulls `[0, 0.5]` (from `x1 % y0 = 0.5`) and `[0.5, 1]` (from `x0 % y1 = 0.5`) touch at `0.5`; the only witnesses are the corners `(2.5, 1)` and `(2, 1.5)`, both excluded (`b = a' + 1`, so no interior `k` exists, see §2.5). |
| `(2.5, 3.5) mod {1}` (degenerate B) | `[0, 1)` | `[0, 0.5) ∪ (0.5, 1)` | `x mod 1 = 0.5` needs `x ∈ {2.5, 3.5}`, both excluded. |
| `{2.5} mod (1, 2)` (degenerate A) | `[0, 1.25)` | `[0, 0.5) ∪ (0.5, 1.25)` | `c%y0 = c%y1 = 0.5` because `(a+1)·y0 = a·y1`; sector 2 gives `[0, 0.5)`, sector 1 gives `(0.5, 1.25)`. |

The first row matters most: it needs neither a degenerate operand nor any closed flag, only
the coincidence `x1 % y0 = x0 % y1` with `⌊x1/y0⌋ = ⌊x0/y1⌋ + 1`. The random fuzz never
produced such an exact coincidence, and it only checked *closed* endpoints against the
oracle, so it could not see this.

**Correction (used by the new prototype).** Interior points of every *individual* hull piece
`(lo, hi)`, `lo < hi`, are always attained (proof: each piece is the image of a sector-portion
of an edge, which is an interval of positive length; its open interior lies in the operand
regardless of flags, and the affine map sends it onto `(lo, hi)`; when an operand is
degenerate the "edge" *is* the operand). Therefore

> **Assembly rule.** `A mod B = ⋃_i (lo_i, hi_i)  ∪  { v ∈ E : attained(v) }`, where the
> `(lo_i, hi_i)` are the open interiors of the *un-merged* hull pieces and `E` is the set of
> all their endpoints. Normalise afterwards. A degenerate hull piece `{v}` contributes only
> its point, and only if attained.

This subsumes the design notes' "drop phantom degenerate pieces" rule and repairs the hole.
It costs at most 5 attainment tests in Q1/Q2 per box (`0`, the two corner values, the
z-point, the P1 corner) — the same as before.

**(b) Everything else checked held.** In particular the attainment oracle is exact, the P2
`a = 0` degenerate-point rule is right, and the design notes' "test against the FULL
operands" caveat is *not required* once each box is computed exactly (§2.4).

---

## Part 2 — All sign combinations

### 2.1 Q2 primitives (dividend ≤ 0, divisor > 0)

Sectors in Q2: for `x < 0 < y`, `⌊x/y⌋ = −m` with `m ≥ 1`, and `f = x + my ∈ [0, y)`.
Sector `m` (as a condition on `y` for fixed `c = x < 0`) is

    −m ≤ c/y < 1−m   ⟺   |c|/m ≤ y  and  (m−1)·y < |c|
                     ⟺   y ∈ [ |c|/m , |c|/(m−1) )   for m ≥ 2,     y ∈ [ |c| , ∞ )  for m = 1.

On sector `m`, `f = c + my` is **increasing** in `y`: it starts at `0` (attained, at
`y = |c|/m`) and tends to `c + m·|c|/(m−1) = |c|/(m−1)` (not attained) for `m ≥ 2`, and to
`+∞` for `m = 1`. For `c = 0`, `⌊0/y⌋ = 0` and `f ≡ 0`; call this "sector 0". The
convenient sector index is `m(y) = ⌈|c|/y⌉ = −⌊c/y⌋`.

**P1 (interval mod scalar), any sign of x. Proof.** `f(·, m)` is `m`-periodic and on each
period `[km, (k+1)m)` equals `x − km`, increasing from `0` (closed) to `m` (open). The Q1
proof of the case table only uses this, so it is valid for every real interval, in
particular for `x ≤ 0`:

> **Box P1.** For `m > 0` and real `x0 ≤ x1`, with `n = ⌊x1/m⌋ − ⌊x0/m⌋`:
> `n = 0 → [x0%m, x1%m]`;  `n = 1 → [0, x1%m] ∪ [x0%m, m)`;  `n ≥ 2 → [0, m)`.
> Closure: `m` is never attained; `x0%m` / `x1%m` are attained at `x0` / `x1` iff that
> endpoint is closed (or by another witness `x0%m + km ∈ A`); `0` (when `n ≥ 1`) is attained
> iff some multiple of `m` lies in `A` — automatic if `n ≥ 2` or if the multiple is interior.

**P2 (scalar mod interval), c ≤ 0.** Let `0 < y0 ≤ y1` and `m_lo = ⌈|c|/y1⌉`,
`m_hi = ⌈|c|/y0⌉` (so `m_lo ≤ m_hi`; both `0` iff `c = 0`).

> **Box P2⁻ (Q2 scalar mod interval).** For `c < 0`:
> - `m_hi = m_lo`: `[c%y0, c%y1]` (one sector, `f` increasing in `y`).
> - `m_hi = m_lo + 1`: `[0, c%y1] ∪ [c%y0, |c|/m_lo)`.
> - `m_hi ≥ m_lo + 2`: `[0, c%y1] ∪ [0, |c|/m_lo)`, i.e. `[0, max(c%y1, |c|/m_lo))` with the
>   max attained iff it is `c%y1` (and `(c, y1) ∈ A×B` or another witness).
> For `c = 0`: `{0}`.
> Endpoint closure (own-piece witnesses): `c%y1` at `(c, y1)`; `c%y0` at `(c, y0)`;
> `0` at `(c, |c|/m_lo)`, which is an interior point of `B` unless `|c|/m_lo = y1`
> (then `c%y1 = 0` and the witness is the corner `(c, y1)`); `|c|/m_lo` is the open supremum
> of sector `m_lo + 1` and is never attained from its own piece.

**Proof.** `B` meets sectors `m_lo, …, m_hi` and no others, because `m(y)` is
non-increasing in `y`. Sector `m_lo`'s part of `B` is `[max(y0, |c|/m_lo), y1]` with image
`[c + m_lo·max(y0,|c|/m_lo), c%y1]`; if `m_hi > m_lo` then `y0 < |c|/m_lo`, so this is
`[0, c%y1]`, and `0` is attained at `y = |c|/m_lo ≤ y1` (`m_lo = ⌈|c|/y1⌉ ≥ |c|/y1`).
Sector `m_hi`'s part is `[y0, |c|/(m_hi−1))` with image `[c%y0, |c|/(m_hi−1))`. If
`m_hi = m_lo + 1` these two are all, giving the second line. If `m_hi ≥ m_lo + 2`, sector
`m_lo + 1` lies wholly in `B` (`y0 < |c|/(m_lo+1)` because `m_hi ≥ m_lo + 2` means
`|c|/y0 > m_lo + 1`, and `|c|/m_lo ≤ y1`), contributing `[0, |c|/m_lo)`; every sector
`m ≥ m_lo + 2` contributes a subset of `[0, |c|/(m−1)) ⊆ [0, |c|/(m_lo+1))` which is
inside it. Monotonicity of `f` on each sector gives the closure claims. ∎

**Divisor pieces open at 0.** After the zero split (§2.4) a divisor box may be `(0, y1]`,
i.e. `y0 = 0` with an open flag. Then `B` contains arbitrarily small positive divisors, so
every sector `m ≥ m_lo` is present: read `m_hi = ⌈|c|/y0⌉ = +∞` (and in Q1 `b = ⌊c/y0⌋ = +∞`),
which selects the `m_hi ≥ m_lo + 2` line; `c % y0` is then never needed. (For `c = 0` the
result is `{0}` before any division.)

Note the structural difference from Q1: Q1's sector `0` has `f ≡ c` (a *point*), Q2's
sector `1` has `f = c + y` unbounded. This is why `[c%y0, c%y1]` grows with `y1` in Q2
while Q1's `[c%y1, c%y0]` shrinks — the hint in the task statement is confirmed exactly.

**Two-edge union for a Q2 box** (Thm C, far edges `(x_far, y_far) = (x0, y1)`):

> **Box Q2.** For `x0 ≤ x1 ≤ 0 < y0 ≤ y1` (finite): `hull(A mod B) = P1([x0,x1], y1) ∪ P2⁻(x0, [y0,y1])`, then the assembly rule of §1.2(a).

The Thm C proof for Q2 assumes `x1 < 0`. Extension to `x1 = 0`: a point `(0, y)` has
`k = 0`, `f = 0`, and `f(0, y1) = 0` is on the top edge, so it is covered by P1
(`0 % y1 = 0`). Points with `x < 0` follow the original argument (their level lines move
left, staying in `x < 0`). ∎

### 2.2 Closure and attainment for Q2

**Zero-touch classification (Q2, non-degenerate A, B).** The zero lines are `x = −m·y`,
`m ≥ 1`, plus `x = 0`. Line `m ≥ 1` meets the closed box iff `|x1|/y1 ≤ m ≤ |x0|/y0`.
Its direction is `(−m, 1)`: from the corner `(x1, y1)` (near-x, far-y) both directions
leave the box, likewise from `(x0, y0)` (far-x, near-y); from `(x0, y1)` and `(x1, y0)`
the line enters the interior. Hence, exactly mirroring Q1 in near/far language:

    0 ∈ A mod B  ⟺  ∃ integer m ≥ 1 strictly between |x1|/y1 and |x0|/y0     (unconditional)
                  ∨ (|x1|/y1 ∈ ℤ≥1  ∧  x1 ∈ A  ∧  y1 ∈ B)                 (touch at (near-x, far-y))
                  ∨ (|x0|/y0 ∈ ℤ≥1  ∧  x0 ∈ A  ∧  y0 ∈ B)                 (touch at (far-x, near-y))
                  ∨ (0 ∈ A)                                                (line x = 0)

**Which corner values need which flags (Q2).** The hull endpoints of a Q2 box are
`x0%y1`, `x1%y1` (P1 corners: far-far and near-far), `x0%y0` (P2, far-near), `0`, `y1`, and the
z-point `|x0|/m_lo`. The near-near corner value `x1%y0` is never an endpoint (it lies inside the
far-edge image). Rule (i) holds: each corner value is attained *at its corner* iff both
operand flags are closed; other witnesses can exist (cross-sector coincidence) and the
attainment test finds them. There is **no k = 0-sector exception** in Q2 (no vertical level
lines), so B's flags always matter for corner values — unlike Q1's `[1,2] mod (3,4) = [1,2]`.

**Degenerate hull pieces that must be dropped when unattained:** `{0}` (corner touch),
`{x0%y1} = {x1%y1}` when `A` is degenerate, `{c%y0} = {c%y1}` when `B` is degenerate, and
the piece `{+∞}` of §2.6.

### 2.3 Q3 and Q4 by the antipodal identity

`f(−x, −y) = −f(x, y)` pointwise (A1), so for sets `f(−A, −B) = −f(A, B)`. With
`−(lo, lc, hi, hc) = (−hi, hc, −lo, lc)` on pieces (flags travel with their endpoints):

> **Box Q3/Q4.** `A mod B = −( (−A) mod (−B) )`, where `(−A, −B)` is a Q1 box when `A ≤ 0, B < 0` and a Q2 box when `A ≥ 0, B < 0`. Closed endpoints map to closed endpoints, because a witness `(x, y)` for `v` is exactly a witness `(−x, −y)` for `−v`.

Attainment transfers the same way: `attained(v, A, B) = attained(−v, −A, −B)`.

### 2.4 Zero-crossing operands

Split `A` into `A⁻ = A ∩ [−∞, 0]` and `A⁺ = A ∩ [0, +∞]` (0 closed in both iff `0 ∈ A`);
split `B` into `B⁻ = B ∩ [−∞, 0)` and `B⁺ = B ∩ (0, +∞]` (0 always removed, open on both
sides; a piece `{0}` becomes empty). Then `A × B = ⋃ A^s × B^t` over the ≤ 4 sign-pure boxes,
and since the image of a union is the union of the images,

> **Box split.** `A mod B = ⋃_{s,t} (A^s mod B^t)` **exactly**, as sets, where each box result
> is the exact set (interior + attained endpoints) of §2.1–2.3. Per-box attainment suffices.

Proof of "per-box suffices": `attained(v)` for the full operands is `∃ (x,y) ∈ A×B` with
`f(x,y) = v`, i.e. `∃ box with a witness in it`, i.e. `v` is in some box's exact result.
The union of exact sets is exact; the normalisation step only has to implement set union
correctly — a value that is an open endpoint of one box's piece and closed (or interior) in
another's is closed in the union. The design notes' "test against the FULL operands"
warning (§3b) is about a different pipeline: if one first merges the *hulls* of all boxes
and then tests only the surviving endpoints, the per-box witnesses are lost. With the
assembly rule of §1.2(a) applied per box, the two pipelines give the same set; the per-box
one is simpler and is what the prototype does. ∎

Multi-piece operands are the same statement with more boxes.

### 2.5 An O(1) attainment test, all quadrants

**Q1** (`0 ≤ x0 ≤ x1`, `0 < y0 ≤ y1`, `v ≥ 0`). Let `B_v = B ∩ (v, ∞) = (y_l, y1]` with
`y_l = max(y0, v)`, lower flag `= (y0 > v ∧ y0 ∈ B)`. A witness is `(v + k·y, y)`, `k ≥ 0`,
`y ∈ B_v`, `v + k·y ∈ A`.

- `k = 0`: attained iff `v ∈ A` and `B_v ≠ ∅`.
- `k ≥ 1`: need `y ∈ I_k := [(x0−v)/k, (x1−v)/k] ∩ B_v`. The closed hulls meet iff
  `(x0−v)/k ≤ y1` and `(x1−v)/k ≥ y_l`, i.e. `k ∈ [K_lo, K_hi]` with
  `K_lo = max(1, ⌈(x0−v)/y1⌉)`, `K_hi = ⌊(x1−v)/y_l⌋`.

> **Box O(1) test (Q1).** `attained(v) ⟺ (v ∈ A ∧ B_v ≠ ∅) ∨ exact(K_lo) ∨ exact(K_hi) ∨ (K_hi ≥ K_lo + 2)`, where `exact(k)` is the flag-aware test `I_k ≠ ∅` and `K_hi ≥ K_lo + 2` is only consulted when `B_v ≠ ∅`.

**Proof.** Any `k` outside `[K_lo, K_hi]` has empty hull intersection, so it is not a
witness. For `K_lo < k < K_hi`: `k > (x0−v)/y1` gives `(x0−v)/k < y1` strictly and
`k < (x1−v)/y_l` gives `(x1−v)/k > y_l` strictly; also `(x0−v)/k ≤ (x1−v)/k` and
`y_l ≤ y1`. If `x0 < x1` and `y_l < y1`, the two open intervals `((x0−v)/k, (x1−v)/k)` and
`(y_l, y1)` therefore overlap (each lower end is below each upper end), and any `y` in the
overlap is interior to `B` with `v + ky` interior to `A` — a witness independent of flags.
If `A = {c}` is degenerate, `y = (c−v)/k` lies strictly inside `(y_l, y1)` and `c ∈ A`
(a non-empty degenerate piece is closed); if `B_v` is degenerate then `B` is (`B_v = {y1}`
forces `y0 = y1`), and `v + k·y1` lies strictly inside `(x0, x1)`; if both are degenerate no
`k` is strictly inside the range. So every strictly interior `k` is a witness, and only the
two extreme integers need the flag-aware check. If no `k ≥ 1` satisfies the hull condition
(`K_hi < K_lo`) only `k = 0` remains. ∎

*Boundary `y_l = 0`.* `y_l = max(y0, v)` is `0` only when `v = 0` and `B` is open at `0`
(a split divisor `(0, y1]`). Then `K_hi = +∞` if `x1 > v` (arbitrarily small `y` gives
arbitrarily large `k`; any `x > 0` in `A` is `k·(x/k)` for large `k`), and no `k ≥ 1` works
if `x1 = v`. *Infinite bounds*: `x1 = +∞` gives `K_hi = +∞`; `y1 = +∞` gives `K_lo = 1`
(`(x0−v)/∞ = 0`). `K_hi = +∞` with `K_lo` finite falls under `K_hi ≥ K_lo + 2`. The `k = 0`
witness with `y = +∞ ∈ B` is `x % ∞ = x`, i.e. again `v ∈ A`; for `k ≥ 1` the divisor must be
finite, so `B_v` is intersected with `y < ∞` first.

**Q2** (`x0 ≤ x1 ≤ 0`, `0 < y0 ≤ y1`, `v ≥ 0`). Witnesses are `(v − m·y, y)`, `m ≥ 0`,
`y ∈ B_v`; `m = 0` forces `v = 0 ∈ A`. For `m ≥ 1`: `y ∈ [(v−x1)/m, (v−x0)/m] ∩ B_v`, hull
condition `m ∈ [M_lo, M_hi]`, `M_lo = max(1, ⌈(v−x1)/y1⌉)`, `M_hi = ⌊(v−x0)/y_l⌋`.

> **Box O(1) test (Q2).** `attained(v) ⟺ (v = 0 ∧ 0 ∈ A ∧ B_v ≠ ∅) ∨ exact(M_lo) ∨ exact(M_hi) ∨ (M_hi ≥ M_lo + 2)`.

The proof is the Q1 proof with `(x0, x1) ↦ (−x1, −x0)`: the substitution `x = v − my`
turns `x ∈ [x0, x1]` into `my ∈ [v − x1, v − x0]`, the same shape of constraint. ∎
Boundary cases as in Q1 (`y_l = 0 ⇒ M_hi = +∞` iff `x0 < v`; `x0 = −∞ ⇒ M_hi = +∞`;
`y1 = +∞ ⇒ M_lo = 1`), plus the infinite value: `attained(+∞) ⟺ +∞ ∈ B ∧ A ∩ (−∞, 0) ≠ ∅`.

**Q3/Q4**: `attained(v, A, B) = attained(−v, −A, −B)` (§2.3), which lands in Q1/Q2.

Cost: three interval intersections and two integer divisions per value, for any operand
magnitude. The old loop's `⌊(x1 − v)/y0⌋` iterations are replaced by a range whose interior is
known to be feasible without inspection.

### 2.6 Infinite operands

Policy D8 (adopted): a dividend endpoint at `±∞` is not a value (`inf % y` is `nan`) and is
dropped, i.e. the flag is forced open; a piece `{±∞}` becomes empty. A divisor endpoint at
`±∞` that is **closed** is a real point with Python's scalar rule, which is also the limit:

    x % +∞ = x  (x ≥ 0),  +∞ (x < 0);        x % −∞ = x  (x ≤ 0),  −∞ (x > 0);   0 % ±∞ = 0.

(The antipodal identity holds for these too: `f(3, −∞) = −f(−3, +∞) = −∞`.)

**Q1 with unbounded pieces** (`A ⊆ [0,∞)`, `B ⊆ (0, ∞]`):

> **Box Q1-∞.**
> - `x1 = +∞`, `y1` finite: `A mod B = [0, y1)` (`0` attained at `x = k·y`, `k` large).
> - `x1 = +∞`, `y1 = +∞` (either flag): `[0, ∞)`; `+∞` is never a value in Q1.
> - `x1` finite, `y1 = +∞` (either flag): `hull = A ∪ P2(x1, B)` with the conventions `⌊x1/∞⌋ = 0`, `x1 % ∞ = x1`; this is the ordinary formula (`a = 0`, right piece `{x1}`, plus `[0, x1%y0]` or `[0, x1/2)` by `b`), together with P1's `n = 0` piece `[x0, x1]`. Closure of `x0`, `x1` follows A's flags (witness `(x, y)` for any `y > x`, always available). A closed `+∞` in `B` adds `{x : x ∈ A}`, already present.

Proof: level rays go north-east; with `x1 = ∞` every ray exits through the top edge
(`v + k·y1 ≤ ∞`), so `f(A×B) = f(A×{y1}) = P1(A, y1)` and `n = ∞ ≥ 2`. With `y1 = ∞` and `x1`
finite, rays with `k ≥ 1` exit through the right edge and the `k = 0` ray `x = v` is
vertical with `f ≡ v = x`, attained already at its own foot; hence `f = P2(x1, B) ∪ A`. With
both infinite, for any `v ≥ 0` take `y = max(y0, v+1)`, `k ≥ x0/y`. ∎

**Q2 with unbounded pieces** (`A ⊆ (−∞, 0]`, `B ⊆ (0, ∞]`):

> **Box Q2-∞.**
> - `x0 = −∞`, `y1` finite: `[0, y1)`.
> - `x0 = −∞`, `y1 = +∞`: `[0, ∞)`, plus the point `+∞` (closed) iff `+∞ ∈ B` and `A` has a negative point.
> - `x0` finite, `y1 = +∞`: `hull = P2⁻(x0, B)` with `m_lo = 1`, `x0 % ∞ := +∞` (an open hull endpoint), i.e. `m_hi = 1 → [x0 + y0, ∞)`; `m_hi = 2 → [0, ∞)` (since `[0,∞) ∪ [c%y0, |c|)`); `m_hi ≥ 3 → [0, ∞)`; together with `{0}` iff `0 ∈ A`. The endpoint `+∞` is closed iff `+∞ ∈ B` and `A` has a negative point.
> - `x0 = 0` (`A = {0}`): `{0}`.

Proof: level lines have direction `(−m, 1)`, exiting through the top (`y = y1`) or the far
x-edge `x = x0`. With `x0 = −∞` only the top exists and `P1(A, y1)` has `n = ∞`. With
`y1 = ∞` every `m ≥ 1` line exits through `x = x0`, giving `P2⁻(x0, B)`; sector 1 there is
`y ≥ |x0|` with `f = x0 + y → ∞`. The value `+∞` itself needs `y = +∞ ∈ B` and some
`x < 0` in `A` (`x % ∞ = ∞`); `0 % ∞ = 0` contributes nothing new. ∎

**Degenerate divisor `{+∞}`** (`y0 = y1 = +∞`, closed): there is no finite divisor, so the
P1/P2 machinery does not apply; only the point rule does. Q1: `A mod {∞} = A` (endpoint
flags of `A`). Q2: `{+∞}` if `A` has a negative point, together with `{0}` if `0 ∈ A`.
An open `{+∞}` is empty.

**Q3/Q4-∞**: negate (`+∞ ↔ −∞`). E.g. `[1, ∞) mod [−3, −2] = −((−∞, −1] mod [2, 3]) = −[0, 3) = (−3, 0]`.

Worked examples (all **checked** by the prototype):

| expression | result |
|---|---|
| `[0, ∞) mod [2, 3]` | `[0, 3)` |
| `[2, 3] mod [1, ∞)` | `[0, 3/2) ∪ [2, 3]` |
| `[2, 3] mod [1, ∞]` | `[0, 3/2) ∪ [2, 3]` (closed `∞` adds nothing in Q1) |
| `(−∞, −1] mod [2, ∞]` | `[0, ∞]` (closed at `∞`) |
| `(−∞, −1] mod [2, ∞)` | `[0, ∞)` |
| `[−3, −2] mod [1, ∞]` | `[0, ∞]`; `mod [1, ∞)` → `[0, ∞)` |
| `[−3, −2] mod {∞}` | `{∞}`;  `[2, 3] mod {∞}` → `[2, 3]` |
| `[1, ∞] mod [2, 3]` | `[0, 3)` (the `∞` dividend point is dropped) |
| `{∞} mod [2, 3]` | `∅` |

---

## What is proved, what is only checked, what is not proved

**Proved (argued above):** the P1 case table for any real dividend interval (§2.1); the Q2
scalar-mod-interval closed form P2⁻ with its own-piece closure rules (§2.1); the Q2 zero-touch
classification and the absence of a k=0 exception (§2.2); Q3/Q4 by negation with flag
transfer (§2.3); exactness of the sign-pure split and sufficiency of per-box attainment
(§2.4); the O(1) attainment test in Q1 and Q2, including the `y_l = 0`, infinite-bound and
degenerate cases (§2.5); the assembly rule "open interiors + attained endpoints" (§1.2(a));
the unbounded-piece formulas and the closed-`∞` rule (§2.6). The Thm C far-edge rule and the
Q1 two-edge reduction are taken from the companion proofs after re-reading them (Part 1).

**Checked numerically only:** that the prototype implements the boxed statements without
transcription slips — this is what the harness below establishes. The brute-force oracle's
`k` range is an elementary bound (`|k| ≤ (|x| + |v|)/|y|`) when every divisor part is
bounded away from 0 and `A` is finite; when `B` reaches 0 or `A` is unbounded it uses the
far divisor end plus a margin, which is ample for the grid and fuzz ranges but is not a
proof of exhaustiveness in general.

**Not proved / open:** nothing in the task list is left unproved. Two things were *not
attempted*: an independent sector-by-sector 2-D image computation as a second oracle
(the brute oracle is per-value, not per-set), and a bound on the number of result pieces
for multi-piece operands (each box contributes at most 3 hull pieces, so it is at most
`3 · #boxes`, but no tighter statement was sought).

## Harness (`modulo_allquadrants_prototype.py`, `__main__`)

Checks per operand pair `(A, B)`, all with exact `Fraction` arithmetic:

- *lattice soundness*: every `(x, y)` on a lattice inside `A × B` (step 1/2 on the grid,
  1/4 in the fuzz; infinite ends replaced by finite proxies; closed `±∞` divisor points
  included) has `x mod y` in the result;
- *sampled soundness*: random rational `(x, y)` (endpoints included when closed) likewise;
- *closure*: each result endpoint is closed iff the brute-force oracle says attained;
- *sharpness*: sampled interior result points are attained (brute-force oracle);
- *gap*: a point in every gap between result pieces, and one unit outside each end, is not
  attained;
- *o1_vs_brute*: the O(1) test and the brute-force oracle agree on every value probed.

Results (run 2026-09-24, `C:/Users/user/anaconda3/envs/intervals/python.exe modulo_allquadrants_prototype.py`, ~3 min; verbatim):

    (a) old Q1 corner/zero-touch suite: 112 combos, 0 mismatches
        hole regression: (2, 5/2) mod (1, 3/2) = [0, 1/2) U (1/2, 5/4)
        hole regression: (5/2, 7/2) mod [1, 1] = [0, 1/2) U (1/2, 1)
        hole regression: [5/2, 5/2] mod (1, 2) = [0, 1/2) U (1/2, 5/4)
    (b) grid: 276 dividend pieces x 325 divisor pieces = 89700 pairs; failures: lattice-soundness=0, sampled-soundness=0, closure=0, sharpness=0, gap=0, o1_vs_brute=0
    (c) fuzz: 6000 cases; failures: lattice-soundness=0, sampled-soundness=0, closure=0, sharpness=0, gap=0, o1_vs_brute=0

Grid (b) is every pair of pieces `lo ≤ hi` over `{−∞, −3, −2, −3/2, −1, −1/2, 0, 1/2, 1,
3/2, 2, 3, +∞}` with all four flag combinations (dividend pieces deduplicated after the
`±∞`-drop; open-degenerate pieces are empty and skipped). Fuzz (c) uses random
`Fraction(n, d)`, `|n| ≤ 24`, `d ≤ 4`, 8 % chance of a `±∞` endpoint, random flags, so it
covers all four quadrants, zero-crossing operands and unbounded pieces.

The design notes' four pre-guard regression targets (§3c) now give
`[−7,−3] mod [2,5] = [0,5)`, `[3,7] mod [−5,−2] = (−5,0]`, `[3,7] mod [−2,5] = (−2,5)`,
`[−3,7] mod [2,5] = [0,5)`.
