import sys; sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import interval as v1i
import intervals as v2
def run(f):
    try: return ('ok', str(f()))
    except Exception as e: return (type(e).__name__, str(e)[:60])
for flag in ['no', 'x', None, 0, 1, [], 'False']:
    print(repr(flag),
          'v1.MultiInterval:', run(lambda: v1.MultiInterval(0, 1, start_closed=flag)),
          'v1i.Interval:', run(lambda: v1i.Interval(0, flag, 1, True)),
          'v2:', run(lambda: v2.MultiInterval(0, 1, start_closed=flag)))
# sabotage: a wrong expectation must be caught
r = run(lambda: v1.MultiInterval(0, 1, start_closed='no'))
print('sabotage caught' if r[0] != 'TypeError' else 'sabotage MISSED')
