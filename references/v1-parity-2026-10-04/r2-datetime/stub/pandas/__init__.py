# minimal stand-in for pandas, so archive/v1/time_interval.py imports under WSL's python (no pandas there).
# v1 uses pandas only for isinstance checks against Timestamp/Timedelta and pd.isna on its bounds;
# these probes never pass pandas objects, so the stand-in classes are never instantiated.
import math
class Timestamp:  # never instantiated
    pass
class Timedelta:  # never instantiated
    pass
def isna(x):
    return x is None or (isinstance(x, float) and math.isnan(x))
NaT = object()  # identity sentinel, never equal to a probe value
