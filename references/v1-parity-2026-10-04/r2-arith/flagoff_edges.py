"""hand-picked flag-off cases: closed +-inf ends and [inf] points through v1 + - * / reciprocal ** and v2; each checked
against the exact oracle at the test points (incl. +-inf)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oracle import *
v1.INFINITY_IS_NOT_FINITE = False

def show(label, f1, f2, oracle_fn=None, extra=()):
    try: r1 = f1(); s1 = str(r1)
    except Exception as e: r1 = None; s1 = f'RAISES {type(e).__name__}: {str(e)[:50]}'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: r2 = f2(); s2 = str(r2)
        except Exception as e: r2 = None; s2 = f'RAISES {type(e).__name__}: {str(e)[:50]}'
    ws = sorted({x.category.__name__ for x in w})
    verdict = ''
    if oracle_fn and r2 is not None:
        pts = test_points(ends2(r2), extra, ends1(r1) if r1 is not None else [])
        bad2 = [str(z) for z in pts if (z in r2) != oracle_fn(z)]
        bad1 = [str(z) for z in pts if r1 is not None and v1_mem(r1, z) != oracle_fn(z)]
        verdict = f'  v2=oracle:{not bad2}{bad2[:2] if bad2 else ""}  v1=oracle:{(not bad1) if r1 is not None else "-"}{bad1[:3] if bad1 else ""}'
    print(f'{label:34s} v1 {s1:28s} v2 {s2:28s} {ws}{verdict}')
    return r1, r2

P = lambda *a: (a[0], a[1] if len(a) > 1 else a[0], a[2] if len(a) > 2 else True, a[3] if len(a) > 3 else True)
cases = [
    ('[1, inf]', [P(1, INF)]), ('[-inf, -1]', [P(-INF, -1)]), ('[inf]', [P(INF)]), ('[-inf]', [P(-INF)]),
    ('[-inf, inf]', [P(-INF, INF)]), ('[0, inf]', [P(0, INF)]), ('[-inf, 0]', [P(-INF, 0)]), ('(0, inf]', [P(0, INF, False, True)]),
    ('[0]', [P(0)]), ('[-1, 1]', [P(-1, 1)]), ('[2, 3]', [P(2, 3)]), ('{[-inf],[inf]}', [P(-INF), P(INF)]),
]
print('=== reciprocal')
for lab, ps in cases:
    A = pieces2(mk2(ps))
    show(f'1/{lab}', lambda: mk1(ps).reciprocal(), lambda: mk2(ps).reciprocal(), lambda z: recip_member(A, z), ends2(mk2(ps)))
print('=== A op B, closed-inf operands')
import itertools
pairs = [('[1, inf]', '[2, 3]'), ('[1, inf]', '[-inf, -1]'), ('[inf]', '[inf]'), ('[inf]', '[-inf]'), ('[inf]', '[0]'),
         ('[0, inf]', '[-1, 1]'), ('[2, 3]', '[0, inf]'), ('[2, 3]', '(0, inf]'), ('[2, 3]', '[inf]'), ('[-inf, inf]', '[2, 3]'),
         ('[-1, 1]', '[-inf, inf]'), ('[2, 3]', '[0]'), ('{[-inf],[inf]}', '{[-inf],[inf]}'), ('[1, inf]', '[1, inf]')]
d = dict(cases)
for a, b in pairs:
    A, B = pieces2(mk2(d[a])), pieces2(mk2(d[b]))
    for op in '+-*/':
        f = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}[op]
        show(f'{a} {op} {b}', lambda: f(mk1(d[a]), mk1(d[b])), lambda: f(mk2(d[a]), mk2(d[b])),
             lambda z: member(op, A, B, z), ends2(mk2(d[a])) + ends2(mk2(d[b])))
print('=== scalar operands (number on either side), incl. +-inf and 0')
for lab in ['[1, inf]', '[-1, 1]', '[inf]', '[0, inf]']:
    for x in [INF, -INF, 0, F(2), -3]:
        A = pieces2(mk2(d[lab])); X = [(fx(x), fx(x), True, True)]
        for op in '+-*/':
            f = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}[op]
            show(f'{lab} {op} {x}', lambda: f(mk1(d[lab]), x), lambda: f(mk2(d[lab]), x), lambda z: member(op, A, X, z), ends2(mk2(d[lab])) + [x])
            show(f'{x} {op} {lab}', lambda: f(x, mk1(d[lab])), lambda: f(x, mk2(d[lab])), lambda z: member(op, X, A, z), ends2(mk2(d[lab])) + [x])
print('=== ** negative exponents on [0] and on bases holding 0, flag off')
for base in ['[0]', '[0, inf]', '[-1, 1]', '(0, inf]']:
    for e in [-1, -2, F(-1, 2), -0.5]:
        show(f'{base} ** {e}', lambda: mk1(d.get(base) or [P(0, INF, False, True)]) ** e, lambda: mk2(d.get(base) or [P(0, INF, False, True)]) ** e)
for e in [[P(-1, 1)], [P(-2, -1)], [P(-INF, 0)], [P(-INF, -1)]]:
    show(f'[0] ** {fmt(e)}', lambda: mk1([P(0)]) ** mk1(e), lambda: mk2([P(0)]) ** mk2(e))
print('=== sabotage: a wrong expectation is caught')
A = pieces2(mk2([P(1, INF)])); B = pieces2(mk2([P(2, 3)]))
r2 = mk2([P(1, INF)]) + mk2([P(2, 3)])
assert all((z in r2) == member('+', A, B, z) for z in test_points(ends2(r2)))
assert not all((z in (r2 | V2(0))) == member('+', A, B, z) for z in test_points(ends2(r2), [0])), 'sabotage missed'
assert not all((z in V2(3, INF, end_closed=False)) == member('+', A, B, z) for z in test_points([3])), 'inf sabotage missed'
print('sabotage caught (an extra point, a missing closed inf)')
