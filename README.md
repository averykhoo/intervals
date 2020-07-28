#   (Time) Intervals for Python
*A GLORIOUS EXERCISE IN YAK-SHAVING*

*   `interval.Interval`: the usual contiguous intervals, where endpoints can be open or closed
    *   support for usable set functions
    *   support for most numeric operations
    *   BUG: expects the unbounded interval to contain infinity, which is not okay since it doesn't allow 
*   `multi_interval.MultiInterval`: a non-contiguous interval
    *   as above, but more support for more things (someday)
*   `TimeInterval`, `TimeIntervalUnion`
    *   hacky initial implementation for time spans and ranges (groups of time spans)
    *   note to self:
        need to differentiate ***position/coordinate vectors*** from ***distance/displacement vectors***,
        i.e. ***timestamps*** and ***timedeltas*** 
*   `Segment`, `MultiSegment`
    *   an attempt to figure out the logic of intervals and multi-intervals

#   notes
*   `TimeInterval` == time span
    *   has a start time and a non-negative duration
    *   stored as a tuple of start_time, end_time
    *   always inclusive of start and end times
*   `TimeIntervalUnion` == time range
    *   zero or more non-overlapping time spans
    *   if an overlapping time span is added, it is merged into the existing span
    *   same idea for removing a time span
    *   boolean logic works on time ranges
        *   somewhat non-intuitive since boundaries aren't respected
        *   e.g. (1pm to 4pm) minus (3pm to 5pm) equals (1pm to 3pm)
        *   even though 3pm was removed from the range, it's still included in the range
        
        
#   todo?
*   iterate over int within multi-interval with at most one half-ray
