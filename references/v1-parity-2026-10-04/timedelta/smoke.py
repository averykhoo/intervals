import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import pandas as pd
import time_interval as v1t
import multi_interval as v1
import intervals as v2
from intervals.time_interval import TimeDeltaInterval as T2, NEG_INF, POS_INF
T1 = v1t.TimeDeltaInterval
print(T1(dt.timedelta(1), dt.timedelta(2)))
print(T2(dt.timedelta(1), dt.timedelta(2)))
print(v2.TimeDeltaInterval is T2, pd.__version__)
