This is a comprehensive collation of the design, mathematics, and implementation strategy for your **Multi-Interval Arithmetic Library**.

It organizes the evolution of the concept from basic set theory to the advanced topological "Stratified Fiber Bundle" required to handle Signed Zero correctly.

---

# 1. The Core Primitives

### The Data Structure: `MultiInterval`
Instead of a single interval `[a, b]`, the atomic unit is a **Finite Set of Disjoint Intervals**.
*   **Mathematical Definition:** $S = \bigcup_{i} [a_i, b_i]$.
*   **Why:** To handle operations that fracture connectivity (e.g., Division by zero, Modulo).
*   **Closure:** This system is closed under $+,-,\times,\div$ (provided infinity is handled correctly).

### The Coordinate System: Hyperreal Tuples
To handle Open vs. Closed bounds and Signed Zero without complex logic trees, we map the Real line to a **Lexicographically Ordered** tuple space.

*   **Format:** `(value, epsilon)`
    *   `value` (float): The location on the real line.
    *   `epsilon` (int): The topological offset.
*   **Sorting:** Python standard sort order `(a, b) < (c, d)` handles the math automatically.
*   **Theoretical Basis:** **Truncated Hyperreals**. The tuple represents the number $x = \text{value} + (\text{epsilon} \cdot \delta)$, where $\delta$ is an infinitesimal.

---

# 2. The Architecture of Epsilon (The Topology)

This was the central evolution of the discussion. How do we map topological states (Limits, Points, Signed Zeros) to integers?

### The Final Verdict: "The Bifurcated Fiber"
**Enum:** `{-2, -1, 0, 1, 2}`
**Merge Rule:** `diff <= 2`

| Enum Value | Name | Usage | Meaning |
| :--- | :--- | :--- | :--- |
| **-2** | `OPEN_END` | All bounds | Limit from below ($x \to a^-$) |
| **-1** | `NEG_POINT` | **Zero Only** | The Point **-0.0** |
| **0** | `STD_POINT` | **Non-Zero** | The Standard Point $a$ ($a \neq 0$) |
| **1** | `POS_POINT` | **Zero Only** | The Point **+0.0** |
| **2** | `OPEN_START` | All bounds | Limit from above ($x \to a^+$) |

### Why this is the Superior Choice
1.  **Stratification:** It acknowledges that the number line has different structures at different points.
    *   **At $x=5$:** The fiber is `{-2, 0, 2}`. (Symmetric).
    *   **At $x=0$:** The fiber is `{-2, -1, 1, 2}`. (Bifurcated).
2.  **Symmetry:** Standard numbers are symmetric around 0. Zero is symmetric around the singularity.
3.  **Strictness:** The distance between `NEG_POINT` (-1) and `OPEN_START` (2) is **3**.
    *   Since the merge rule is **2**, `[-0]` does **not** merge with `(0...`.
    *   This preserves the crucial topological distinction between "Stopping at Negative Zero" and "Starting after Positive Zero."

### Alternative Options (Discarded)
*   **`{-1, 0, 1}`:** Too simple. Cannot distinguish Signed Zero from Open/Closed without messy `if val==0` checks.
*   **`{-2, -1, 0, 2}` (The Shifted Model):** Uses merge rule `diff <= 3`. Merges `[-0]` and `(0...`. Good for "Loose" logic, but loses directional information.
*   **`{-2, -1, 1, 2}` (The Dualist Model):** Forces standard numbers to be signed (5 is `+1`). Creates asymmetry where closing a gap from the left is topologically "harder" than from the right.

---

# 3. The Topology of Zero

This is the most complex constraint. IEEE 754 defines $-0.0 == +0.0$, but Interval Arithmetic requires distinguishing them for division bounds.

### The "Line with Two Origins"
You have implemented a **Hausdorff** internal representation (sorted, distinct) with a **Non-Hausdorff** external interface (merging).

*   **The Gap:** Between $-0$ and $+0$ lies the "Singularity."
*   **Strict Logic (Recommended):**
    *   `[..., -0)` merges with `[+0]`. (Limit touches Point).
    *   `[-0]` does **NOT** merge with `(0...`. (Negative Point does not touch Positive Limit).
*   **Why Strict?** It preserves branch cuts for functions like $1/x$, $\sqrt{x}$, and $\log(x)$.

### The "Pragmatic" Argument
If you want to support "Loose" logic (where $-0$ and $+0$ are effectively the same), you simply change the merge rule to `diff <= 3`. The Bifurcated Enum allows you to toggle this behavior globally by changing one integer constant.

---

# 4. Algorithms & Operations

### A. Division (`Reciprocal`)
Standard floating point division handles the sign, provided you pass the correct bounds.
*   **Input:** `[-5, 5]`.
    *   Split into `[-5, -0]` and `[+0, 5]` using the Enum logic.
*   **Calc:**
    *   $1 / -0.0 \to -\infty$.
    *   $1 / +0.0 \to +\infty$.
*   **Result:** `[-inf, -0.2] U [0.2, inf]`.
*   **Note:** Infinity should be **Closed** (`[a, inf]`) to strictly contain the result of division by zero.

### B. Modulo (`%`)
Do not use geometric projection. Use **Quotient Analysis**.
For $A \pmod B$:
1.  Calculate integer quotients of endpoints: $q_{low} = \lfloor a_{low}/b \rfloor$, $q_{high} = \lfloor a_{high}/b \rfloor$.
2.  **Case 1 ($q_{low} == q_{high}$):** No wrap. Result $[a\%b, b\%b]$.
3.  **Case 2 ($q_{high} == q_{low} + 1$):** Single wrap. Result $[a\%b, b) \cup [0, b\%b]$.
4.  **Case 3 ($q_{high} > q_{low} + 1$):** Multi wrap. Result $[0, b)$.

### C. Multiplication by Zero
*   **Problem:** $[0, 0] \times [\infty, \infty]$.
*   **Pure Math:** Undefined.
*   **Standard Float:** NaN.
*   **Interval Safe Mode:** `[-inf, inf]`. (Because $0 \times \infty$ is an indeterminate form that could be any value depending on underflow rate).
*   **Signed Logic:** If you know it is `+0` and `+inf`, you can claim `[0, inf]`. But `[-inf, inf]` is the safest default.

---

# 5. Theoretical Underpinnings (Glossary)

*   **Lexicographic Product:** The mechanism of your sort. $\mathbb{R} \times \mathbb{Z}$. It turns a 2D state space into a 1D totally ordered chain.
*   **Stratified Space:** The fact that the fiber at Zero (`{-2,-1,1,2}`) is different from the fiber at Five (`{-2,0,2}`).
*   **Khalimsky Topology:** The concept of the number line as an alternating sequence of Open Intervals and Closed Points.
*   **Monad:** The "Cloud" of hyperreal numbers surrounding a standard real number. Your Enum is a low-resolution map of this cloud.
*   **Sheaf of Piecewise Functions:** What your library effectively manipulates—functions that are continuous almost everywhere but handle jump discontinuities explicitly.

---

# 6. Python Implementation Best Practices

### Formatting
*   **`__str__`:** Output the mathematical notation `"[1, 2)"`.
*   **`__repr__`:** Output the constructor string `"MultiInterval('[1, 2)')"`.
*   **Avoid:** Outputting lists like `[[1], (1, 2)]`. It confuses users and implies false discontinuity.

### Input Parsing
*   **`(1, 1)`:** This is **Empty**, not Error.
*   **`[1, 1)`:** This is **Empty**, not Error.
*   **`[2, 1]`:** This is **Error**.

### Negative Zero Detection
Python treats `0.0 == -0.0` as True. You must use `math.copysign(1, x)` to detect the sign bit during object construction.

```python
if value == 0.0 and math.copysign(1, value) < 0:
    self.eps = Eps.NEG_POINT
```

### Final Summary
You have built a **Lexicographically Ordered, Stratified Fiber Bundle over the Affine Extended Reals**. It uses a **Bifurcated Enum** to handle the topological singularity at zero, and a **Tolerance Relation** (`diff <= 2`) to define connectivity. It is mathematically rigorous, computationally efficient ($O(1)$ comparisons), and handles edge cases that break standard libraries.