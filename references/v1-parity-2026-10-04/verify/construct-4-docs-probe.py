import sys; sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
M = v2.MultiInterval
cases = [((1, 1), dict(start_closed=False)), ((1, 1), dict(end_closed=False)), ((1, 1), dict(start_closed=False, end_closed=False))]
for args, kw in cases:
    try: r1 = repr(v1.MultiInterval(*args, **kw))
    except Exception as e: r1 = f'{type(e).__name__}: {e}'
    try: r2 = M(*args, **kw); r2s = f'{r2!r} is_empty={r2.is_empty}'
    except Exception as e: r2s = f'{type(e).__name__}: {e}'
    print(args, kw, '| v1:', r1, '| v2:', r2s)
for s in ['(1, 1)', '[1, 1)', '(1, 1]', '[2, 1]']:
    try: r1 = repr(v1.MultiInterval.from_str(s)) if hasattr(v1.MultiInterval, 'from_str') else 'n/a'
    except Exception as e: r1 = f'{type(e).__name__}: {e}'
    try: r2 = repr(M.from_str(s)) if hasattr(M, 'from_str') else repr(M(s))
    except Exception as e: r2 = f'{type(e).__name__}: {e}'
    print(repr(s), '| v1:', r1, '| v2:', r2)
try: M(2, 1); print('v2 [2,1] did not raise')
except ValueError as e: print('v2 [2,1] ValueError', e)
