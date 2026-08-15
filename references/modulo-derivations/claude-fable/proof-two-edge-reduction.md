# Proof: two-edge reduction for interval modulo (positive quadrant)

> Provenance: generated 2026-08-15 by an independent Claude (Fable) subagent given only the
> bare claim — no access to this repo's code, slides, or prior numerics. Hand-checked
> (inequality chains in both witness cases) and consistent with dense numerical validation
> run separately in-session. See also `proof-sign-symmetries-quadrants.md` for the other
> quadrants and `v3-modulo-design-notes.md` for how this feeds the implementation.

---

# Every value of x mod y on a rectangle is attained on its top or right edge

## Claim

Let $0 \le x_0 \le x_1$ and $0 < y_0 \le y_1$, let $f(x,y) = x \bmod y = x - y\lfloor x/y\rfloor$, and let $R = [x_0,x_1]\times[y_0,y_1]$. Then

$$f(R) \;=\; f\big([x_0,x_1]\times\{y_1\}\big) \;\cup\; f\big(\{x_1\}\times[y_0,y_1]\big).$$

The claim is **true**. The inclusion $\supseteq$ is trivial, since both edges are subsets of $R$. The substance is $\subseteq$, proved constructively below.

## Structural lemmas

Work in the quadrant $Q = \{(x,y) : x \ge 0,\ y > 0\}$, which contains $R$.

**Lemma 1 (sector decomposition and zero lines).**
$Q$ is the disjoint union over integers $k \ge 0$ of the sectors
$$S_k = \{(x,y) \in Q : ky \le x < (k+1)y\},$$
and on $S_k$ we have $\lfloor x/y\rfloor = k$ and $f(x,y) = x - ky$, an affine function. The *zero lines* are the rays $Z_k = \{x = ky,\ y>0\}$; they are exactly the zero set of $f$, and $Z_k$ is the lower boundary of $S_k$ (contained in $S_k$). $f$ is continuous on each sector but jumps across zero lines (approaching $Z_{k+1}$ from within $S_k$, $f \to y$; on $Z_{k+1}$ itself, $f = 0$).

*Proof.* $\lfloor x/y\rfloor = k \iff ky \le x < (k+1)y$ by definition of floor; these conditions partition $Q$. The rest is immediate. $\square$

**Lemma 2 (level sets are north-east rays).**
Fix an integer $k \ge 0$ and a real $v \ge 0$. Then
$$\{(x,y) \in S_k : f(x,y) = v\} \;=\; L_{k,v} := \{(v + kt,\; t) : t > v\},$$
a ray with direction vector $(k,1)$, whose coordinates are both nondecreasing in $t$ (strictly increasing when $k \ge 1$; vertical when $k=0$). In particular, $f \equiv v$ exactly on all of $L_{k,v}$.

*Proof.* For $(x,y) \in S_k$, $f(x,y) = v \iff x = v + ky$, and the sector condition $ky \le v+ky < (k+1)y$ is equivalent to $0 \le v < y$, i.e. $y > v$ (using $v \ge 0$). Conversely every point $(v+kt, t)$ with $t > v$ satisfies $kt \le v+kt < (k+1)t$, hence lies in $S_k$ with $\lfloor(v+kt)/t\rfloor = k$ and $f = v$ **exactly** — not by a continuity argument but by direct evaluation of the floor. $\square$

**Lemma 3 (level rays and zero lines).**
For $v > 0$, $L_{k,v}$ is a translate of the zero line $Z_k$ lying strictly between $Z_k$ and $Z_{k+1}$; a path along $L_{k,v}$ never meets a zero line, so it never crosses a discontinuity of $f$ and the value $v$ is conserved exactly. For $v = 0$, $L_{k,0} = Z_k$: the path lies *on* a zero line for its entire length, where $f \equiv 0$ identically. Either way, following a level ray never changes the value. (This is why the construction below is safe even when $R$ contains part of a zero line.)

**Lemma 4 (north-east exit lemma).**
Let $p = (x,y) \in R$ and $d = (k,1)$ with integer $k \ge 0$. Set
$$s^\* = \min\Big(y_1 - y,\ \tfrac{x_1 - x}{k}\Big) \quad (\text{the second term} = +\infty \text{ if } k = 0).$$
Then $s^\* \ge 0$, the segment $\{p + s\,d : 0 \le s \le s^\*\}$ lies in $R$, and $q = p + s^\* d$ lies on the top edge $[x_0,x_1]\times\{y_1\}$ (if $s^\* = y_1 - y$) or the right edge $\{x_1\}\times[y_0,y_1]$ (if $s^\* = (x_1-x)/k$).

*Proof.* $s^\* \ge 0$ since $x \le x_1$, $y \le y_1$. Along $p + sd$ both coordinates are nondecreasing, so the constraints $x \ge x_0$, $y \ge y_0$ can never become violated; the only constraints that can bind are $x \le x_1$ and $y \le y_1$, and $s^\*$ is by definition the first parameter at which one of them binds, at which point $y = y_1$ (top edge) or $x = x_1$ (right edge) respectively, with the other coordinate still within its interval. $\square$

## Constructive proof of the claim

Let $(x,y) \in R$, $k = \lfloor x/y\rfloor \ge 0$ (an integer since $x \ge 0$, $y > 0$), and $v = f(x,y) = x - ky \in [0, y)$. Define the **witness point**:

$$\boxed{\;q = \begin{cases} \big(v + k y_1,\; y_1\big) & \text{if } v + k y_1 \le x_1 \quad(\text{top edge; always the case when } k = 0),\\[4pt] \Big(x_1,\; \dfrac{x_1 - v}{k}\Big) & \text{otherwise (right edge; here necessarily } k \ge 1).\end{cases}\;}$$

This is exactly the endpoint $p + s^\* d$ of Lemma 4 applied to the level ray $L_{k,v}$ through $(x,y)$ (which sits on it at parameter $t = y > v$, by Lemma 2). We verify each case directly.

**Case 1: $v + ky_1 \le x_1$ (top edge).**
- *In $R$:* the $x$-coordinate satisfies $v + ky_1 \ge v + ky = x \ge x_0$ (using $y_1 \ge y$) and $v + ky_1 \le x_1$ by assumption; the $y$-coordinate is $y_1$. So $q \in [x_0,x_1]\times\{y_1\}$.
- *Same value:* $0 \le v < y \le y_1$, so by Lemma 2 (with $t = y_1 > v$), $f(v + ky_1,\, y_1) = v$.

Note that if $k = 0$ then $v = x \le x_1$, so Case 1 always applies; hence Case 2 has $k \ge 1$ and no division by zero.

**Case 2: $v + ky_1 > x_1$ (right edge), $k \ge 1$.** Let $t^\* = (x_1 - v)/k$.
- *In $R$:* $t^\* \ge (x - v)/k = y \ge y_0$ (using $x \le x_1$), and $t^\* < y_1$ by the case assumption rearranged; the $x$-coordinate is $x_1$. So $q \in \{x_1\}\times[y_0,y_1]$.
- *Same value:* $t^\* \ge y > v$, and $x_1 = v + k t^\*$, so by Lemma 2 (with $t = t^\* > v$), $f(x_1, t^\*) = x_1 - k t^\* = v$.

In both cases $q$ lies on one of the two named edges and $f(q) = v$ exactly; moreover the whole connecting segment $\{(v+kt,\,t) : y \le t \le t_q\}$ stays inside $R$ (Lemma 4) and inside the single sector $S_k$ (Lemma 2), so the value is constant along the entire path, never merely "approximately preserved," and the path never crosses a zero line (Lemma 3). This proves $f(R) \subseteq f(\text{top}) \cup f(\text{right})$, and with the trivial reverse inclusion, the claim. $\blacksquare$

**Geometric summary.** Within each sector $S_k$ the level sets of $f$ are rays parallel to the zero line $x = ky$, all pointing "north-east" (direction $(k,1)$, both components $\ge 0$). A north-east ray launched from any point of a rectangle exits through the top or the right edge — never the bottom or left. Sliding each point of $R$ along its level ray until it exits therefore compresses all of $f(R)$ onto those two edges. (This also explains why the claim would be *false* with the bottom/left edges in place of top/right: the level rays point away from them.)

## Edge cases

- **Degenerate $x_0 = x_1$:** $R$ *is* the right edge; the claim reads $f(R) = f(\text{top}) \cup f(R)$, true since $\text{top} \subseteq R$. The construction still works verbatim (it never uses $x_0 < x_1$).
- **Degenerate $y_0 = y_1$:** $R$ is the top edge; symmetric to the above. Both degeneracies together give a single point; all three sets coincide.
- **$x_1$ an exact multiple of some $y \in [y_0,y_1]$:** then $f(x_1, y) = 0$, so $0$ is attained on the right edge directly. In the construction this appears as a level path with $v = 0$ landing at $(x_1,\, x_1/k)$.
- **$R$ contains part of a zero line $x = ky$:** $f$ is discontinuous across it (values jump from near $y$ down to $0$), but the construction is unaffected: each level path stays in one sector, and the $v = 0$ paths travel *along* the zero line (Lemma 3), where $f \equiv 0$ exactly, ending at $(ky_1, y_1)$ or $(x_1, x_1/k)$.
- **$x_0 = 0$:** the point $(0, y)$ has $k = 0$, $v = 0$; Case 1 gives witness $(0, y_1)$ with $f = 0$. No special handling needed. (More generally, any point with $x < y$ has $k = 0$ and $v = x < y \le y_1$, so its witness $(x, y_1)$ is just its vertical projection to the top edge.)
- **Exactness of the floor:** the proof never takes limits; the identity $\lfloor (v+kt)/t\rfloor = k$ holds exactly whenever $0 \le v < t$, which is maintained throughout since $t \ge y > v$.

## Numerical spot-check

The explicit construction was verified at 135,010 random points across 150 random rectangles plus targeted edge cases (degenerate rectangles, $x_0 = 0$, $x_1$ an exact multiple of several $y$, rectangles crossed by many zero lines): in every single case the witness point lay on the top or right edge, inside $R$, with $f$-value matching to within $10^{-7}$ (0 failures, 0 floating-point artifacts). An independent set-level check — 1,000,000 dense interior samples of $f$ over four rectangles compared against densely sampled edge values — found every interior value matched on an edge within $10^{-3}$ (0 misses; an earlier apparent mismatch was traced to under-sampling the edges in the test itself, not to the claim).
