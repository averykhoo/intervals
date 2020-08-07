#   (Time) Intervals for Python
*A GLORIOUS EXERCISE IN YAK-SHAVING*

##  Numeric Intervals
*   `multi_interval.MultiInterval`: a non-contiguous interval
    *   as above, but more support for more things (someday)
    *   the `__pow__` operation is not closed (neither is `__rpow__`)
    *   neither are `__divmod__`, `__floordiv__`, or `__rfloordiv__`
        *   those can result in infinite integer sequences, e.g. [0, inf) // 1 == range(infinity)
        *   operations on them can result in weird sequences to infinity which are not straightforward to represent
    *   does not and will never support complex numbers, because that requires building a graphics processing library
        *   minkowski addition/subtraction
        *   convex hulls
        *   boolean operations on shapes with curves
        *   erosion
        *   translation, rotation, scaling, skewing
        *   spirals
        
##  Time Intervals
*   `time_interval.DateTimeInterval` and `time_interval.TimeDeltaInterval`: non-contiguous time intervals
    *   behaves somewhat like `datetime.datetime` and `datetime.timedelta` merged with `multi_interval.MultiInterval`
    *   also accepts `pandas.Timestamp` and `pandas.Timedelta`
    *   uses `dateutil` to parse date strings

##  alternative implementation of numeric intervals, for debugging
*   `interval.Interval`: the usual contiguous intervals, where endpoints can be open or closed
    *   support for usable set functions
    *   support for most numeric operations
*   `interval.MultipleInterval`: operations on groups of `Interval` objects
