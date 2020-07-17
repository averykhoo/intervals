#   time-range
time spans and ranges (groups of time spans)

#   what is this
*   time span
    *   has a start time and a non-negative duration
    *   stored as a tuple of start_time, end_time
    *   always inclusive of start and end times
*   time range
    *   zero or more non-overlapping time spans
    *   if an overlapping time span is added, it is merged into the existing span
    *   same idea for removing a time span
    *   boolean logic works on time ranges
        *   somewhat non-intuitive since boundaries aren't respected
        *   e.g. (1pm to 4pm) minus (3pm to 5pm) equals (1pm to 3pm)
        *   even though 3pm was removed from the range, it's still included in the range
*   segment, multi-segment
    *   an attempt to figure out the logic, using integers instead of datetime.datetime
    *   datetime.datetime is quantized anyway, since the minimum datetime.timedelta is one microsecond
    *   also datetime.datetime can be converted to an integer by multiplying the timestamp by a million
*   interval, interval-union
    *   hacky initial implementation