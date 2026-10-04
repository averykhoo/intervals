"""non-bool open/closed flags: v1 refuses (TypeError), v2 reads truthiness"""
from common import *  # noqa
import numpy as np


def run(f):
    try:
        return 'ok', f()
    except Exception as ex:  # noqa
        return type(ex).__name__, str(ex)


for label, f1, f2 in [
    ("flag 'no' (meant open?)", lambda: V1(0, 'no', 1, True), lambda: MI(0, 1, start_closed='no')),
    ('flag None', lambda: V1(0, None, 1, True), lambda: MI(0, 1, start_closed=None)),
    ('flag 0/1 ints', lambda: V1(0, 1, 1, 0), lambda: MI(0, 1, start_closed=0, end_closed=1)),
    ('flag np.bool_', lambda: V1(0, np.True_, 1, np.False_), lambda: MI(0, 1, start_closed=np.False_, end_closed=np.False_)),
    ('flag [] (falsy list)', lambda: V1(0, [], 1, True), lambda: MI(0, 1, start_closed=[])),
    ("point, flags 'a'/'b'", lambda: V1(1, 'a', 1, 'b'), lambda: MI(1, start_closed='a', end_closed='b')),
    ("from_pieces flag 'x'", lambda: None, lambda: MI.from_pieces([(0, 1, 'x', 'x')])),
]:
    print(f'{label:26} v1 {run(f1)}  v2 {run(f2)}')
check('SABOTAGE', run(lambda: MI(0, 1, start_closed='no'))[0] == 'TypeError')
print('sabotage caught', len(MISMATCHES), 'of 1')
