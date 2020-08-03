#   ~~Time~~ Intervals for Python
*A GLORIOUS EXERCISE IN YAK-SHAVING*

*   `interval.Interval`: the usual contiguous intervals, where endpoints can be open or closed
    *   support for usable set functions
    *   support for most numeric operations
    *   BUG: expects the unbounded interval to contain infinity, which is not okay since it doesn't allow 
*   `multi_interval.MultiInterval`: a non-contiguous interval
    *   as above, but more support for more things (someday)
    *   the `__pow__` operation is not closed (neither is `__rpow__`)
    *   neither are `__divmod__`, `__floordiv__`, or `__rfloordiv__`,
        since that can result in infinite integer sequences, e.g. [0, inf) // 1 == range(infinity)   
    *   does not and will never support complex numbers, because that requires building a graphics processing library
        *   minkowski addition/subtraction
        *   convex hulls
        *   boolean operations on shapes with curves
        *   erosion
        *   translation, rotation, scaling, skewing
        *   spirals
*   `time_interval.DateTimeInterval` and `time_interval.TimeDeltaInterval`: non-contiguous time intervals
    *   behaves somewhat like `datetime.datetime` and `datetime.timedelta` merged with `multi_interval.MultiInterval`
*   `Segment`, `MultiSegment`
    *   an attempt to figure out the logic of intervals and multi-intervals
        
#   todo?
*   iterate over int within multi-interval with at most one half-ray
*   special classes for datetime-multi-intervals and timedelta-multi-intervals
    *   note to self:
        need to differentiate ***position/coordinate vectors*** from ***distance/displacement vectors***,
        i.e. ***timestamps*** and ***timedeltas*** 
