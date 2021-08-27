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

# TODO

* create `adjoining()`
  * adjacent -> next to but not touching
  * adjoining -> touching but not intersecting
  * intersecting -> not fully overlapping
  * overlapping -> complete intersection (ie. proper subset or superset)
