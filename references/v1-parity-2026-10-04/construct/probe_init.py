"""MultiInterval.__init__: every argument form, v1 vs v2"""
from common import *
import random
from decimal import Decimal
import numpy as np

inf, nan = math.inf, math.nan
def show(label, args, kw={}):
    o1 = outcome(lambda: M1(*args, **kw))
    o2 = outcome(lambda: M2(*args, **kw))
    s1 = str(o1[1]) if o1[0] == 'ok' else 'RAISE ' + o1[1]
    s2 = str(o2[1]) if o2[0] == 'ok' else 'RAISE ' + o2[1]
    same = (o1[0] == o2[0] == 'ok' and v1_to_v2(o1[1]) == o2[1]) or (o1[0] == o2[0] == 'raise')
    print(f'{"SAME" if same else "DIFF"} {label:42s} v1={s1:40s} v2={s2}')
    return same

print('--- edge table')
show('empty ()', ())
show('empty, flags differ', (), dict(start_closed=True, end_closed=False))
show('end without start', (None, 1))
show('point int', (3,))
show('point float', (2.5,))
show('point Fraction', (F(1, 3),))
show('point -0.0', (-0.0,))
show('point inf', (inf,))
show('point -inf', (-inf,))
show('open degenerate (x)', (3,), dict(start_closed=False, end_closed=False))
show('half-open degenerate [x)', (3,), dict(end_closed=False))
show('start_closed=None (open?)', (1, 2), dict(start_closed=None))
show('[1,2]', (1, 2))
show('[1,2)', (1, 2), dict(end_closed=False))
show('(1,2]', (1, 2), dict(start_closed=False))
show('(1,2)', (1, 2), dict(start_closed=False, end_closed=False))
show('[2,1] reversed', (2, 1))
show('[1,1] two-arg point', (1, 1))
show('[1,1) empty-by-flags', (1, 1), dict(end_closed=False))
show('(1,1) empty-by-flags', (1, 1), dict(start_closed=False, end_closed=False))
show('[1, inf] closed at inf', (1, inf))
show('[1, inf)', (1, inf), dict(end_closed=False))
show('[-inf, 1]', (-inf, 1))
show('(-inf, 1]', (-inf, 1), dict(start_closed=False))
show('(-inf, inf)', (-inf, inf), dict(start_closed=False, end_closed=False))
show('(inf, inf] start after inf', (inf, inf), dict(start_closed=False))
show('(inf, inf)', (inf, inf), dict(start_closed=False, end_closed=False))
show('[-inf, -inf)', (-inf, -inf), dict(end_closed=False))
show('(-inf,-inf)', (-inf, -inf), dict(start_closed=False, end_closed=False))
show('[-0.0, -0.0]', (-0.0, -0.0))
show('[-0.0, 1]', (-0.0, 1))
show('[-1, -0.0]', (-1, -0.0))
show('nan alone', (nan,))
show('nan, 2 (v1 swaps)', (nan, 2))
show('1, nan', (1, nan))
show('nan, nan', (nan, nan))
show('bool True', (True,))
show('bool False, True', (False, True))
show('str "1"', ('1',))
show('str "1","2"', ('1', '2'))
show('Decimal 1.5', (Decimal('1.5'),))
show('np.float64 1.5', (np.float64(1.5),))
show('np.int64 3, 5', (np.int64(3), np.int64(5)))
show('int, Fraction mix', (1, F(5, 2)))
show('float, Fraction mix', (0.5, F(5, 2)))
show('int 1, float 1.0', (1, 1.0))
show('huge int 10**400', (10 ** 400,))
show('complex 1j', (1j,))

print('--- what v2 str shows for [1, 1.0] (mixed types), v1 str:', M1(1, 1.0), '| v2:', M2(1, 1.0))

print('--- random sweep: (a, b, flags) from int/Fraction/float, incl. a > b and a == b')
random.seed(1234)
t = Tally('init sweep')
pool = [-3, -1, 0, 1, 2, 5, F(1, 3), F(-7, 2), 0.5, -2.25, 1.0, 2.0, inf, -inf]
for _ in range(600):
    a, b = random.choice(pool), random.choice(pool)
    sc, ec = random.random() < .5, random.random() < .5
    single = random.random() < .15
    args = (a,) if single else (a, b)
    kw = dict(start_closed=sc, end_closed=ec)
    o1 = outcome(lambda: M1(*args, **kw)); o2 = outcome(lambda: M2(*args, **kw))
    if o1[0] == 'ok' and o2[0] == 'ok':
        pts = probe_points([a] if single else [a, b])
        bad = same_set_by_membership(lambda p: v1_contains(o1[1], p), lambda p: p in o2[1], pts)
        t.check(not bad, ('membership', args, kw, bad[:3]))
    elif o1[0] != o2[0]:
        # who refuses what: classify
        t.check(False, ('outcome', args, kw, o1[0], o1[1] if o1[0] == 'raise' else str(o1[1]),
                        o2[0], o2[1] if o2[0] == 'raise' else str(o2[1])))
    else:
        t.check(True, None)
t.report(show=0)
from collections import Counter
kinds = Counter()
for b in t.bad:
    if b[0] == 'membership':
        kinds['membership'] += 1
    else:
        _, args, kw, k1, s1, k2, s2 = b
        hasinf = any(isinstance(x, float) and math.isinf(x) for x in args)
        key = ('v1 raises' if k1 == 'raise' else 'v2 raises') + (' inf-arg' if hasinf else ' finite') + \
              (' a==b' if len(args) == 2 and args[0] == args[1] else '') + (' single' if len(args) == 1 else '')
        kinds[key] += 1
        if kinds[key] <= 2:
            print('   e.g.', key, args, kw, '| v1:', s1, '| v2:', s2)
print('mismatch kinds:', dict(kinds))

print('--- sabotage: a deliberately wrong expectation must be caught')
s = Tally('sabotage'); x = M2(1, 2)
s.check(not same_set_by_membership(lambda p: v1_contains(M1(1, 2), p), lambda p: p in M2(1, 2, end_closed=False), probe_points([1, 2])), 'wrong flag')
assert s.report() == 1, 'sabotage NOT caught'
print('sabotage caught')
