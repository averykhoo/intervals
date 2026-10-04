from common import *
import numpy as np

def outcome(f):
    try:
        r = f()
        return 'ok', r
    except Exception as e:
        return type(e).__name__, str(e)[:70]

cases = [
    ('plain', (0, False, 1, True), (0, 1, True, True)),
    ('open-open', (0, True, 1, False), (0, 1, False, False)),
    ('point', (2, False, 2, True), (2, 2, True, True)),
    ('half-open point [1,1)', (1, False, 1, False), (1, 1, True, False)),
    ('half-open point (1,1]', (1, True, 1, True), (1, 1, False, True)),
    ('open point (1,1)', (1, True, 1, False), (1, 1, False, False)),
    ('backwards', (2, False, 1, True), (2, 1, True, True)),
    ('nan start', (math.nan, False, 1, True), (math.nan, 1, True, True)),
    ('start +inf', (INF, True, INF, False), (INF, INF, False, False)),
    ('end -inf', (-INF, True, -INF, False), (-INF, -INF, False, False)),
    ('closed -inf', (-INF, False, 1, True), (-INF, 1, True, True)),
    ('closed +inf', (0, False, INF, True), (0, INF, True, True)),
    ('open rays', (-INF, True, INF, False), (-INF, INF, False, False)),
    ('point at inf', (INF, False, INF, True), (INF, INF, True, True)),
    ('string start', ('0', False, 1, True), ('0', 1, True, True)),
    ('flag int 0', (0, 0, 1, True), (0, 1, 0, True)),
    ('bool endpoint', (False, False, True, True), (False, True, True, True)),
    ('Fraction', (F(1, 3), False, F(2, 3), True), (F(1, 3), F(2, 3), True, True)),
    ('numpy float64', (np.float64(0.5), False, np.float64(1.5), True), (np.float64(0.5), np.float64(1.5), True, True)),
    ('numpy int64', (np.int64(1), False, np.int64(3), True), (np.int64(1), np.int64(3), True, True)),
    ('-0.0', (-0.0, False, 1, True), (-0.0, 1, True, True)),
    ('None start', (None, False, 1, True), (None, 1, True, True)),
]
for name, a, b in cases:
    o1 = outcome(lambda: I(*a))
    o2 = outcome(lambda: M(b[0], b[1], start_closed=b[2], end_closed=b[3]))
    print(f'{name:24s} v1: {o1[0]:10s} {str(o1[1])[:45]:45s} | v2: {o2[0]:10s} {str(o2[1])[:60]}')
    if o1[0] == 'ok' and o2[0] == 'ok':
        same_reals('construct ' + name, o1[1], o2[1])

# properties
for name, a in [('plain', (0, False, 1, True)), ('open', (0, True, 1, False)), ('pt', (2, False, 2, True)), ('ray', (-INF, True, 3, True))]:
    i = I(*a); m = to_v2(i)
    print(name, 'start_closed', i.start_closed, m.inf_closed, '| end_open', i.end_open, not m.sup_closed,
          '| is_degenerate', i.is_degenerate, m.is_degenerate, '| length', i.length, m.size.length, m.wid(),
          '| start/end', i.start, i.end, m.inf, m.sup)
    check('start_closed ' + name, i.start_closed == m.inf_closed)
    check('end_open ' + name, i.end_open == (not m.sup_closed))
    check('is_degenerate ' + name, i.is_degenerate == m.is_degenerate)
    check('start ' + name, i.start == m.inf)
    check('end ' + name, i.end == m.sup)

# start_tuple / end_tuple as an order: compare to v2 cuts
rng = random.Random(1)
ivs = [rand_interval(rng) for _ in range(400)]
for _ in range(2000):
    x, y = rng.choice(ivs), rng.choice(ivs)
    cx, cy = to_v2(x).cuts, to_v2(y).cuts
    check('start_tuple order', (x.start_tuple < y.start_tuple) == (cx[0] < cy[0]), (x, y))
    check('end_tuple order', (x.end_tuple < y.end_tuple) == (cx[-1] < cy[-1]), (x, y))
    check('start vs end tuple', (x.start_tuple <= y.end_tuple) == (cx[0] < cy[-1]), (x, y))
    # dataclass order=True vs sort_key
    check('dataclass order', (x < y) == (to_v2(x).sort_key < to_v2(y).sort_key), (x, y))
    check('eq/hash', (x == y) == (to_v2(x) == to_v2(y)) and ((x == y) <= (hash(to_v2(x)) == hash(to_v2(y)))), (x, y))
print('sorted lists equal:', [to_v2(i) for i in sorted(ivs)] == sorted([to_v2(i) for i in ivs], key=lambda m: m.sort_key))
check('sorted', [to_v2(i) for i in sorted(ivs)] == sorted([to_v2(i) for i in ivs], key=lambda m: m.sort_key))
print('v1 frozen:', outcome(lambda: setattr(I(0, False, 1, True), 'start', 5)))
print('v2 frozen:', outcome(lambda: setattr(M(0, 1), '_cuts', 5)))
print('v1 length of [0,1]:', I(0, False, 1, True).length, ' (start - end: negative)')
# SELFTEST: a wrong expectation must be caught
same_reals('SELFTEST expected mismatch', I(0, False, 1, True), M(0, 1, end_closed=False))
report_end(__file__)
