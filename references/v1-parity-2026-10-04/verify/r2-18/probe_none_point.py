import sys, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
from intervals import MultiInterval as MI
for sf, ef in [(None, False), (False, None), (None, None), (True, 1), (0, False)]:
    try: r1 = v1.MultiInterval(5, start_closed=sf, end_closed=ef).endpoints
    except Exception as e: r1 = type(e).__name__
    try: r2 = str(MI(5, start_closed=sf, end_closed=ef))
    except Exception as e: r2 = f'{type(e).__name__}: {e}'
    print(repr(sf), repr(ef), 'v1', r1, 'v2', r2)
