"""random v1/v2 pairs built from the same pieces (ends off the whole second, so v1's snap never fires)"""
from common import *


def rand_piece(r, whole=False, days=6):
    kind = r.random()
    a = rand_dt(r, span_days=days, whole=whole)
    if kind < 0.2:
        return (a,), {}
    if kind < 0.3:
        return (a.date(),), {}
    b = a + dt.timedelta(seconds=r.randrange(1, 86400 * 2), microseconds=r.randrange(1, 10 ** 6) if not whole else 0)
    if not whole and b.microsecond == 0:
        b += US
    if kind < 0.4:
        return (a.date(), b.date()), {}
    return (a, b), {'start_closed': r.random() < .5, 'end_closed': r.random() < .5}


def rand_pair(r, max_pieces=4, whole=False, days=6):
    n = r.randrange(0, max_pieces + 1)
    ps = [rand_piece(r, whole, days) for _ in range(n)]
    a1 = V1D()
    a2 = V2D()
    for args, kw in ps:
        a1 = a1.union(V1D(*args, **kw))
        a2 = a2.union(V2D(*args, **kw))
    return a1, a2, ps
