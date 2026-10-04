"""string parsing (merge(str)), __str__ text format, __repr__, and: can v2 read v1's str output back?"""
from common import *
import random
inf = math.inf

def v1_parse(s):
    return M1.merge(s)

print('=== 1. v1 grammar corpus: v1 merge(str) vs v2 MultiInterval.parse(str)')
corpus = ['[1, 2]', '[1,2]', '[0]', '{0}', '{}', '[]', '()', '(123)', '{ [1, 2) | [3, 4) }', '{[1,2),[3,4)}',
          '[1,2)[3,4)', '(1, 2)', '(1, 2]', '[1; 2]', '{1, 2, 3}', '{1; 2}', '{ 1 , 2 }', '[-1, 5)', '(-inf, 5]',
          '(-inf, inf)', '[2.5, 3.75)', '[1e-5, 1]', '[1e5, 1e6]', '[- 3, 4]', '[-3, - 4.5]', '[1)', '[1, 2, 3]',
          '{ [1, 2) , [3] }', '[1, inf)', '[1, inf]', '[-inf, 0)', 'foo [1, 2] bar', '[1, 2] garbage', '5',
          '-inf', '[1/2, 1]', '[1e+20]', '[1E5]', '[.5, 1]', '[1., 2]', '{ [1, 2] , [2, 3] }', '[2, 1]', '(1, 1)',
          '[1, 1)', '{ (0, 1) ∪ [2] }', '[0x10, 20]', '[+1, 2]', '[1,2] [3,4]', '{[1,2]}{[5,6]}', '  [ 1 , 2 ]  ',
          '{ [1, 2) , (2, 3] }', '[-0, 1]', '[-0.0, 1]', '[∞]', '(-∞, 0]', '[-1,1)(1,2]']
from collections import Counter
kinds = Counter()
for s in corpus:
    o1 = outcome(lambda: v1_parse(s))
    o2 = outcome(lambda: M2.parse(s))
    if o1[0] == 'ok':
        c = outcome(lambda: v1_to_v2(o1[1]))
        t1 = ('ok', c[1]) if c[0] == 'ok' else ('raise', 'converted: ' + c[1])
        s1 = outcome(lambda: str(o1[1]))
        d1 = s1[1] if s1[0] == 'ok' else f'INVALID OBJ {o1[1].endpoints}'
    else:
        t1, d1 = o1, 'RAISE ' + o1[1]
    d2 = str(o2[1]) if o2[0] == 'ok' else 'RAISE ' + o2[1]
    if t1[0] == o2[0] == 'ok':
        tag = 'SAME' if t1[1] == o2[1] else 'DIFF'
    elif t1[0] == o2[0]:
        tag = 'SAME'
    else:
        tag = 'DIFF'
    kinds[tag] += 1
    print(f'{tag} {s!r:28s} v1={d1:34s} v2={d2}')
print(dict(kinds))

print('=== 2. __str__ text: v1 vs v2 for the same set (random v1 intervals incl. Fractions and odd floats)')
random.seed(99)
vals = [-100, -7, -1, 0, 1, 3, 42, F(1, 3), F(-22, 7), 0.1, 0.1 + 0.2, 1e-05, 1e20, 1e-300, 2.5, -1.75, 123456789.125, 1e16, 5e-324]
t_text, t_round_v2, t_round_v1, t_round_v1_bad = Tally('str text v1 == v2'), Tally('v2.parse(str(v1)) == v1 set'), Tally('v1.merge(str(v1)) == v1 set'), []
for _ in range(600):
    m1 = M1()
    for _ in range(random.randint(0, 4)):
        a, b = sorted(random.sample(vals, 2))
        if random.random() < .25:
            b = a
        lc, hc = (True, True) if a == b else (random.random() < .5, random.random() < .5)
        if random.random() < .1 and a != b:
            a, lc = -inf, False
        if random.random() < .1 and a != b:
            b, hc = inf, False
        m1 = m1.union(M1(a, b, start_closed=lc, end_closed=hc))
    if random.random() < .3:  # v1's own generator too
        m1 = v1.random_multi_interval(-50, 50, random.randint(0, 4), random.choice([0, 1, 2]))
    s = str(m1)
    ref = v1_to_v2(m1)
    t_text.check(s == str(ref), (s, str(ref)))
    p2 = outcome(lambda: M2.parse(s))
    t_round_v2.check(p2[0] == 'ok' and p2[1] == ref, (s, p2))
    p1 = outcome(lambda: v1_to_v2(M1.merge(s)))
    ok1 = p1[0] == 'ok' and p1[1] == ref
    t_round_v1.check(ok1, (s, str(p1[1]) if p1[0] == 'ok' else p1))
t_text.report(6); t_round_v2.report(6); t_round_v1.report(8)

print('=== 3. __repr__')
print('v1 repr(M1(1,2)):', outcome(lambda: repr(M1(1, 2))))
print('v1 repr([M1(1,2)]):', outcome(lambda: repr([M1(1, 2)])))
t_repr = Tally('v2 eval(repr(x)) == x')
random.seed(5)
from intervals import MultiInterval
for _ in range(300):
    m2 = M2.from_pieces([tuple(sorted(random.sample(vals, 2))) + (random.random() < .5, random.random() < .5) for _ in range(random.randint(0, 3))])
    r = repr(m2)
    t_repr.check(eval(r, {'MultiInterval': MultiInterval}) == m2, r)
t_repr.report()
print('example v2 repr:', repr(M2.parse('{ [1, 2) , (3, inf] }')), '| str:', str(M2.parse('{ [1, 2) , (3, inf] }')))

print('=== sabotage: a wrong expectation must be caught')
s = Tally('sabotage'); s.check(str(M1(1, 2, end_closed=False)) == str(M2(1, 2)), 'text differs')
assert s.report() == 1
print('sabotage caught')
