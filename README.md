# (Time) Intervals for Python

*A GLORIOUS EXERCISE IN YAK-SHAVING*

## Numeric Intervals

* `multi_interval.MultiInterval`: a non-contiguous interval
  * `__mod__` is only partly implemented, `__rmod__` is not yet implemented
    * it's not impossible, but since the intuition is hard to grasp, it's probably not useful
    * diagrams in the [ppt](./interval-modulo.pptx)
  * the `__pow__` and `__rpow__` operations are not closed
    * negative number raised to fractional powers result in complex numbers
    * if modulo is specified, then the same problems appear as for mod
    * but the special cases are enumerable, so it's possible to implement
  * neither are `__divmod__`, `__floordiv__`, or `__rfloordiv__`
    * those can result in infinite integer sequences, e.g. [0, inf) // 1 == range(infinity)
    * operations on them can result in weird sequences to infinity which are not straightforward to represent
  * does not and will never support complex numbers, because that requires building a graphics processing library
    * minkowski addition/subtraction
    * convex hulls
    * boolean operations on shapes with curves
    * erosion
    * translation, rotation, scaling, skewing
    * spirals (eg. raising complex number to a power), circles, ellipses, maybe parabolas

## Time Intervals

* `time_interval.DateTimeInterval` and `time_interval.TimeDeltaInterval`: non-contiguous time intervals
  * behaves somewhat like `datetime.datetime` and `datetime.timedelta` merged with `multi_interval.MultiInterval`
  * also accepts `pandas.Timestamp` and `pandas.Timedelta`

## alternative implementation of numeric intervals, for debugging

* `interval.Interval`: the usual contiguous intervals, where endpoints can be open or closed
  * support for usable set functions
  * support for most numeric operations
* `interval.MultipleInterval`: operations on groups of `Interval` objects

## notes:

* for mod, can use a perspective transform (projective but not affine)
  to convert the rectangle between the 2 intervals in the into a trapezoid in a cartesian-like space then use the
  coverage to determine the output
  * but in practice only need 2 points, check for overlap, then determine if closed
  * can probably overlap with the mod of real numbers, or real number mod
  * see powerpoint and screenshots for an idea of how it works

# TODO

* MAYBE FIX: 1 / [-2, 2) = (-inf, -0.5], (0.5, inf) <- make sure to drop the gap (-0.5, 0.5], but maybe this should only be valid when inf can be closed, i.e. 1/0=[inf], and -1/0=[-inf] as denegerete intervals 
* create `adjoining()`
  * adjacent -> next to but not touching
  * adjoining -> touching but not intersecting
  * intersecting -> not fully overlapping
  * overlapping -> complete intersection (ie. proper subset or superset)
* finish `__mod__()`
* allow interval modulo for `__pow__()`
* redo illustrations with negative and positive bits
  * use excel chart rather than conditional formatting
  * variable min max for x and y axis
    * with re-sampling so that zero is always a line, and then vary the min/max a bit to compensate
  * probably 500x500 should be good enough, although maybe aim for 800x800?
    * or use a different aspect ratio? 600x800?
  * use better colors
  * zoom into x axis a bit to show there are infinite lines near there



Geminis feedback:

This is a **fascinating and highly sophisticated implementation**.

You have solved the hardest problem in interval arithmetic—boolean comparison of open/closed endpoints—using the `(-1, 0, 1)` epsilon strategy. That specific design choice drastically reduces the cyclomatic complexity of your merge and set operations.

However, scrutinizing this file reveals that **you are significantly under-utilizing the power of your own class in the `reciprocal` (division) logic**, and the `__modulo` implementation is a potential maintenance nightmare.

Here is the deep dive.

### 1. The "Epsilon Tri-State" (Genius Move)
Using `epsilon in {-1, 0, 1}` is brilliant because it aligns strictly with python's tuple comparison logic:
*   Start Points: `(x, 0)` (Closed) comes before `(x, 1)` (Open). **Correct** ($[x$ includes $x$, $(x$ does not).
*   End Points: `(x, -1)` (Open) comes before `(x, 0)` (Closed). **Correct** ($x)$ excludes $x$, $x]$ includes it).

This allows your `merge` and `difference_update` functions to use simple linear scans without complex `if` trees checking `is_open`.

### 2. The Critical Flaw: `reciprocal` (Division)
You constructed a `MultiInterval` class specifically to handle disjoint sets, but your `reciprocal` method reverts to standard interval behavior (total information loss) when zero is involved.

**Current Code:**
```python
if 0 in self:
    return MultiInterval(start=-math.inf, end=math.inf, ...)
```

**The Problem:**
If `self` is `[-2, 2]`:
*   $1 / [-2, 2]$ theoretically maps to $(-\infty, -0.5] \cup [0.5, \infty)$.
*   Your code returns $(-\infty, \infty)$.
*   **You lost the gap $(-0.5, 0.5)$.**

**The Fix:**
You must split the interval at zero *before* inverting.
```python
def reciprocal(self) -> 'MultiInterval':
    if self.is_empty:
        return MultiInterval()

    # Create a clean split at zero, preserving openness/closedness
    # This relies on your difference_update logic being correct
    zero_interval = MultiInterval(0) # Degenerate [0, 0]
    
    # Check if we actually contain 0 (not just boundaries)
    has_zero = 0 in self
    
    out = MultiInterval()
    
    # If we have positive parts
    pos_part = self.positive # You already have this property
    if not pos_part.is_empty:
        # Invert [0, 5] -> [0.2, inf], (0, 5] -> [0.2, inf)
        # Your current loop logic actually handles the inversion math correctly
        # if you feed it strictly positive/negative intervals
        out.update(self._invert_component(pos_part))

    # If we have negative parts
    neg_part = self.negative
    if not neg_part.is_empty:
        out.update(self._invert_component(neg_part))
        
    # If the input strictly contained 0 (e.g. [-2, 2]), the result spans to +/- inf
    # If the input was [0, 2], the result is [0.5, inf] (closed at inf per your design)
    return out
```

### 3. The `__modulo` Geometry Engine
Your `__modulo` function is implementing a geometric sweep-line algorithm to calculate $y = x \pmod m$.
*   **Critique:** It is excessively complex (cycling through cases of 0, 1, or 2 wraps).
*   **Risk:** This is the most likely place for bugs, especially with floating point precision errors near the "zero lines" ($k \cdot m$).
*   **Alternative:** Since you already have `MultiInterval` capabilities, you can define Modulo as a series of Intersection and Subtraction operations, which you have already perfected.

**Simpler Algorithm (Conceptual):**
For $A \pmod B$:
1.  If $B$ is a scalar `m`:
    *   Slice $A$ into chunks of size $m$: $A_0 = A \cap [0, m)$, $A_1 = A \cap [m, 2m)$, etc.
    *   Shift them all back to origin: $A'_1 = A_1 - m$, $A'_2 = A_2 - 2m$.
    *   Union them: $\bigcup A'_i$.
2.  This utilizes your robust `intersection` and `add/sub` logic instead of custom handling edge cases.

### 4. Minor Implementation Notes

#### A. The Python `or` Trick
In `apply_monotonic_binary_function`:
```python
# min(..., x_open or y_open)
# max(..., -x_open or -y_open)
```
You are relying on Python's `or` returning the integer value.
*   `0 (Closed) or 1 (Open)` $\to$ `1` (Open). **Correct.**
*   `0 (Closed) or 0 (Closed)` $\to$ `0` (Closed). **Correct.**
*   `-1 (Open) or 0 (Closed)` $\to$ `-1` (Open). **Correct.**
*   `-1 (Open) or -1 (Open)` $\to$ `-1` (Open). **Correct.**
*   **Note:** This is clever, but dangerous if you ever port this code to a language with boolean coercion.

#### B. Regex in `merge`
```python
re_num = re.compile(r'(?:-\s*)?(?:inf|\d+(?:\.\d+)?(?:e-?\d+)?)\s*', flags=re.U)
```
Inside `merge` (which is used by `intersection`, `union`, etc.), you define and compile regexes.
*   **Performance Hit:** Even though Python caches regex compilation, putting this logic inside the hottest path of your library (comparisons/merges) is expensive.
*   **Fix:** Move the regex compilation to the module level (global constant).

#### C. `__pow__` and Complex Numbers
```python
elif self.is_negative:
    return NotImplemented 
```
Good choice. If you ever implement this, remember that `MultiInterval` implies Real numbers. Roots of negative numbers produce a circle in the complex plane. You cannot represent that with `endpoints` on a 1D line. You would need a `ComplexInterval` class (Rectangle or Disc).

### Summary for "Production Readiness"

1.  **Fix `reciprocal`**: Do not return `(-inf, inf)` for inputs containing zero. Return the union of the inverted negative part and inverted positive part.
2.  **Move Regex**: Pull `re.compile` out of the `merge` method.
3.  **Trust Your Set Logic**: Use your own `difference` and `intersection` methods to implement advanced math (like modulo) rather than writing custom geometric overlap logic.

The core architecture (`endpoints` list + epsilon logic) is solid. It's strictly better than the standard "List of Interval Objects" approach.