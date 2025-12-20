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


---


This is the **Grand Unified Collation** of the MultiInterval library design.

It documents the final architecture you chose (`{-2, -1, 0, 2}`), the mathematical theory behind it, the algorithmic implementations, and the rejected alternatives.

---

# 1. The Core Architecture (Selected)

### The Coordinate System
You represent real numbers as **Lexicographically Ordered Tuples**: `(value, epsilon)`.
*   **Primary Sort:** `value` (Standard IEEE 754 float).
*   **Secondary Sort:** `epsilon` (Topological offset/state).

### The Epsilon Enum: `{-2, -1, 0, 2}`
You selected the **Symmetric/Shifted** Enum. This provides symmetry for standard numbers while maintaining a distinction for signed zero.

| Enum Value | Name | Mapped To | Topological Meaning |
| :--- | :--- | :--- | :--- |
| **-2** | `OPEN_END` | `(..., x)` | Limit from below ($x \to a^-$) |
| **-1** | `NEG_POINT` | `[-0]` | The Point $-0.0$ (Zero Only) |
| **0** | `STD_POINT` | `[x]` / `[+0]` | The Point $x$ (if $x \ne 0$) or $+0.0$ |
| **2** | `OPEN_START` | `(x, ...)` | Limit from above ($x \to a^+$) |

### The Merge Rule: `diff <= 3`
This rule defines the **Connectivity** (Digital Topology) of the number line.
*   **Standard Numbers ($x=5$):**
    *   Left Limit (`-2`) to Point (`0`): Diff **2**. (Merge).
    *   Point (`0`) to Right Limit (`2`): Diff **2**. (Merge).
*   **The Singularity ($x=0$):**
    *   Left Limit (`-2`) to NegPoint (`-1`): Diff **1**. (Merge).
    *   NegPoint (`-1`) to PosPoint (`0`): Diff **1**. (Merge).
    *   PosPoint (`0`) to Right Limit (`2`): Diff **2**. (Merge).
    *   **The Bridge:** NegPoint (`-1`) to Right Limit (`2`): Diff **3**. (**Merge**).

**Why this works:** It allows `[-0]` and `(0, ...)` to merge (filling the "Zero Gap"), creating a "least astonishing" experience for general users, while internally preserving the Signed Zero distinction for operations that need it (like Division).

---

# 2. Mathematical Foundations

### Theoretical Constructs
1.  **Stratified Fiber Bundle:** The number line is the base space $B$. Attached to every point is a fiber $F$.
    *   For $x \ne 0$, the fiber is `{-2, 0, 2}`.
    *   For $x = 0$, the fiber is `{-2, -1, 0, 2}`.
    *   Because the fiber changes structure at the origin, the space is **Stratified**.
2.  **Lexicographic Product:** The sort order of `(float, int)` creates a **Totally Ordered Chain**, turning the 2D state space into a 1D lattice.
3.  **Truncated Hyperreals:** The tuple maps to $x = \text{val} + \text{eps} \cdot \delta$.
    *   You are using **First-Order Infinitesimals**.
    *   Higher orders (acceleration/curvature $\epsilon^2$) are discarded as irrelevant for Interval Sets.
4.  **Khalimsky Topology:** The integers define an alternating sequence of Open and Closed sets.
    *   Your Enum maps the continuous neighborhood of a real number to this discrete topological grid.

### The "Double Origin"
You have implemented the **Line with Two Origins** (non-Hausdorff) but fixed the sorting problem to make it Hausdorff internally.
*   **Internal State:** $-0 \ne +0$. (Separated/Hausdorff).
*   **External API:** $-0$ merges with $+0$. (Glued/Non-Hausdorff).

---

# 3. Arithmetic Operations

### A. Division (The Splitter)
This is the primary reason for the complex architecture.
*   **Operation:** $1 / [-5, 5]$.
*   **Internal Logic:**
    1.  Split interval at Zero using Enum: `[-5, -0]` and `[+0, 5]`.
    2.  Apply $1/x$:
        *   $1 / -0.0 \to -\infty$.
        *   $1 / +0.0 \to +\infty$.
*   **Result:** `[-inf, -0.2] U [0.2, inf]`.
*   **Constraint:** Infinity must be **Closed** (`[a, inf]`) to strictly contain the expansion of the zero point.

### B. Modulo (The Sawtooth)
Do not use geometric projections. Use **Quotient Analysis**.
For $A \pmod B$ (where $B$ is scalar):
1.  Calculate integer quotients: $q_{low} = \lfloor a_{low}/b \rfloor$, $q_{high} = \lfloor a_{high}/b \rfloor$.
2.  **Case 1 ($q_{low} == q_{high}$):** No wrap. `[a%b, b%b]`.
3.  **Case 2 ($q_{high} == q_{low} + 1$):** Single wrap (Split). `[a%b, b) U [0, b%b]`.
4.  **Case 3 ($q_{high} > q_{low} + 1$):** Full coverage. `[0, b)`.
*   *Note:* This handles negative inputs automatically via `math.floor`.

### C. Multiplication involving Infinity
*   **Problem:** $0 \times \infty$.
*   **Pure Math:** Undefined.
*   **Pragmatic Interval:** `[-inf, inf]`. (Indeterminate form).
*   **Standard Numbers:** `0 * 5 = 0`.
*   **Logic:** If inputs are `[0, 0]` and `[inf, inf]`, return Full Line. If inputs are `[0, 5]` (approaching 0) and `[inf]`, you *can* technically claim `[0, inf]`, but `[-inf, inf]` is safer if the origin of the zero is unknown.

---

# 4. Implementation Details

### Parsing & Empty Sets
*   **Input `(1, 1)`:** This represents $\{ x \in \mathbb{R} \mid 1 < x < 1 \}$.
    *   **Action:** Normalize to **Empty Set**. Do not raise Error.
*   **Input `[2, 1]`:** This is a contradiction.
    *   **Action:** Raise **ValueError**.

### Signed Zero Detection
Python treats `0.0 == -0.0` as True.
*   **Detection:** Use `math.copysign(1, x)`.
*   **Normalization:** Do **not** normalize `-0.0` to `0.0` in your constructor. Store the value as `0.0` (for simple math) but set the `epsilon` Enum to `NEG_POINT` (`-1`).

### Output Representations
*   **`__str__`:** Use math notation: `"[1, 2)"`.
*   **`__repr__`:** Use constructor string: `"MultiInterval('[1, 2)')"`.
*   **Avoid:** Decomposed lists `[[1], (1, 2)]`. They imply false discontinuity.

---

# 5. Rejected Alternatives (The Graveyard)

### Option: `{-1, 0, 1}` (Simple)
*   *Structure:* Limit, Point, Limit.
*   *Why Rejected:* Cannot distinguish `-0` from `+0` topologically without checking `if val == 0`. Leaky abstraction.

### Option: `{-2, -1, 1, 2}` (The Dualist/Strict)
*   *Structure:* No center. Negative Limit/Point, Positive Point/Limit.
*   *Why Rejected:*
    *   **Asymmetry:** For standard numbers ($x=5$), "Left Limit" is distance 3 away, "Right Limit" is distance 1 away.
    *   **Strictness:** It forces `[-0]` and `(0...` to remain disjoint (Distance 3, Merge Rule 1 or 2). While useful for solvers, it is "Astonishing" for general users doing standard math.

### Option: Complex Numbers (2D Tuples)
*   *Structure:* `(Real_Dir, Imag_Phase)`.
*   *Why Rejected:* Python cannot sort Complex numbers. Requires custom comparators, killing performance. The Enum is a flattened "Projection" of this 2D structure anyway.

### Option: Higher Order Hyperreals ($\epsilon^2$)
*   *Structure:* `(val, slope, curvature)`.
*   *Why Rejected:* Overkill. Interval sets only care about boundaries (Order 1), not grazing/curvature (Order 2).

---

# 6. Final Summary of Your System

You have constructed a library based on the **Affine Extended Reals** with a **Stratified Topology** at the origin.

1.  **User View:** A standard set of intervals. `[-0]` and `[0]` behave "normally" and merge with neighbors.
2.  **Internal View:** A high-precision topological ribbon. `-0` is strictly less than `+0`.
3.  **Bridge:** The Enum `{-2, -1, 0, 2}` and Merge Rule `diff <= 3` act as the "Gluing Map" that collapses the internal high-resolution singularity into a user-friendly continuous line.

It is theoretically sound, performant ($O(1)$ ops), and handles the edge cases of limits and division robustly.


---


To create a robust, "Pythonic" warning system for your library, you should define a custom Warning class that inherits from `RuntimeWarning`.

This allows your users to distinguish between "Standard Python Warnings" and "Your Library's Mathematical Edge Cases."

Here is the implementation strategy.

### 1. Define the Custom Warning
Define this at the top of your module or in a dedicated `exceptions.py` file. Inheriting from `RuntimeWarning` signals that this is a mathematical issue that occurred during execution, not a syntax or deprecation issue.

```python
import warnings

class IntervalIndeterminateWarning(RuntimeWarning):
    """
    Warns when an operation results in a total loss of precision 
    (e.g., 0/0 -> [-inf, inf] or 0 * inf -> [-inf, inf]).
    """
    pass
```

### 2. Emit the Warning (With the correct Stack Level)
When you detect the $0/0$ or $0 \times \infty$ case, emit the warning.

**Crucial Detail:** Use `stacklevel=2`.
*   **Default (`stacklevel=1`):** The warning points to **your library code** (the line where you typed `warnings.warn`). This is useless to the user.
*   **Correct (`stacklevel=2`):** The warning points to **the user's code** that called your division method.

```python
# Inside MultiInterval class
def __truediv__(self, other):
    # ... logic to determine if it is 0/0 ...
    is_indeterminate = (self.is_zero and other.is_zero) 

    if is_indeterminate:
        warnings.warn(
            "Indeterminate form 0/0 encountered. Returning [-inf, inf].",
            category=IntervalIndeterminateWarning,
            stacklevel=2
        )
        return MultiInterval.full_line() # Your [-inf, inf] constructor
    
    # ... rest of division logic ...
```

### 3. How the User Controls Behavior

Because you used a custom class, the user now has full granular control using Python's standard `warnings` filter.

#### A. Default Behavior (The "Helpful" Log)
The code runs, calculates `[-inf, inf]`, but prints a message to stderr so the user knows *why* their bounds exploded.
```python
x = MultiInterval(0)
y = x / x
# Output to console: 
# UserWarning: Indeterminate form 0/0 encountered...
# Result: [-inf, inf]
```

#### B. The "Strict" Mode (Crash on 0/0)
If a user is writing a critical solver and wants to forbid indeterminate forms, they can turn your specific warning into an Exception.

```python
import warnings
from your_library import IntervalIndeterminateWarning

# Turn ONLY your warning into an error
warnings.simplefilter("error", IntervalIndeterminateWarning)

try:
    y = x / x
except IntervalIndeterminateWarning:
    print("Caught the 0/0 calculation!")
    # Handle the crash
```

#### C. The "Production" Mode (Silence)
If the user knows this will happen and doesn't care (e.g., they are intersecting the result later anyway), they can silence it.

```python
# Ignore specific warning
warnings.simplefilter("ignore", IntervalIndeterminateWarning)

y = x / x # No print, no crash. Just returns [-inf, inf]
```

### 4. Advanced: Context Manager for Temporary Strictness
This is the cleanest way to handle the "Solver vs. General Math" dichotomy we discussed. You can be strict only within a specific block of code.

```python
import warnings

# Temporarily crash on 0/0
with warnings.catch_warnings():
    warnings.simplefilter("error", IntervalIndeterminateWarning)
    
    # Mathematical critical section
    perform_sensitive_calculation()

# Back to normal (non-crashing) behavior here
perform_loose_calculation()
```

### Summary
1.  **Class:** `class IntervalIndeterminateWarning(RuntimeWarning): pass`
2.  **Usage:** `warnings.warn("msg", IntervalIndeterminateWarning, stacklevel=2)`
3.  **Benefit:** This satisfies the **Containment Principle** (returning full set) while satisfying the **Debugging Requirement** (alerting the user), without forcing you to add messy `strict=True` flags to every method signature.


---

* (wrapped) gmpy for math, which supports round up or down, and will be used to keep interval math accurate - alternatively just plus or minus an ulp an a dummy class, or use decimal which also supports rounding but is slower (and still doesn't support trig)
* turns out all the other interval libraries don't support multi interval math and just stretch a single interval over the whole thing
* and they also don't really support mod
* consider using the itl test vectors from IntervalArithmetic.jl
* 

