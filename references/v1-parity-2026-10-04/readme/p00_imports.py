import sys; sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import interval as v1i
import time_interval as v1t
import compare as v1c
import intervals as v2
print('ok', v1.MultiInterval, v1i.Interval, v1t.DateTimeInterval, v2.MultiInterval, v1.INFINITY_IS_NOT_FINITE)
print([n for n in dir(v2) if not n.startswith('_')])
