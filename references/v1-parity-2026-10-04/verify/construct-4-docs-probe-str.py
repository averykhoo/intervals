import sys; sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
M = v2.MultiInterval
for s in ['(1, 1)', '[1, 1)', '(1, 1]', '[1, 1]', '[2, 1]']:
    try: r1 = repr(v1.MultiInterval(s))
    except Exception as e: r1 = f'{type(e).__name__}: {e}'
    try: r2 = repr(M.parse(s))
    except Exception as e: r2 = f'{type(e).__name__}: {e}'
    print(repr(s), '| v1:', r1, '| v2:', r2)
# sanity: the probe can fail
assert M.parse('[1, 1]').is_empty is False
assert M.parse('[1, 1)').is_empty is True
