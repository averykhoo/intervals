import sys; sys.path[:0] = ['.', 'tests']
import intervals.time_interval as ti
import test_time_interval as t
def run(label):
    try:
        t.test_tz_dates_property(); print(label, 'PASS')
    except BaseException as e:
        print(label, 'FAIL', type(e).__name__, str(e).splitlines()[0][:200] if str(e) else '')
run('unpatched')
ti._TimeInterval.__bool__ = lambda self: True   # v1's object-default truthiness
run('bool always True (v1)')
