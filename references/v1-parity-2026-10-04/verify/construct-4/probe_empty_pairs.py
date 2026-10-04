import sys, random, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import intervals as v2
M1, M2 = v1.MultiInterval, v2.MultiInterval

def run(f):
    try:
        r = f()
        return ('ok', r)
    except Exception as e:
        return ('err', f'{type(e).__name__}: {e}')

def v1_members(m, pts):
    return tuple(p in m for p in pts)
def v2_members(m, pts):
    return tuple(p in m for p in pts)

print('== hand cases ==')
cases = [
    ('2-arg [1,1)', lambda M: M(1, 1, end_closed=False)),
    ('2-arg (1,1]', lambda M: M(1, 1, start_closed=False)),
    ('2-arg (1,1)', lambda M: M(1, 1, start_closed=False, end_closed=False)),
    ('2-arg [1,1]', lambda M: M(1, 1)),
    ('2-arg [2,1]', lambda M: M(2, 1)),
    ('2-arg (2,1)', lambda M: M(2, 1, start_closed=False, end_closed=False)),
    ('1-arg (1)', lambda M: M(1, start_closed=False, end_closed=False)),
    ('1-arg [1)', lambda M: M(1, end_closed=False)),
    ('0-arg flags (,]', lambda M: M(start_closed=False)),
    ('0-arg flags [,)', lambda M: M(end_closed=False)),
    ('0-arg flags (,)', lambda M: M(start_closed=False, end_closed=False)),
    ('2-arg (inf,inf]', lambda M: M(float('inf'), float('inf'), start_closed=False)),
    ('2-arg [-inf,-inf)', lambda M: M(-float('inf'), -float('inf'), end_closed=False)),
    ('2-arg [F(1,3),F(1,3))', lambda M: M(F(1,3), F(1,3), end_closed=False)),
    ('2-arg [1, 1.0)', lambda M: M(1, 1.0, end_closed=False)),
]
for name, f in cases:
    a, b = run(lambda: f(M1)), run(lambda: f(M2))
    sa = a[1] if a[0] == 'err' else f'empty={a[1].is_empty}'
    sb = b[1] if b[0] == 'err' else f'empty={b[1].is_empty} {b[1]}'
    print(f'{name:24s} v1 {sa:60s} | v2 {sb}')

print('== strings: v1 merge(str) vs v2 parse ==')
for s in ['(1, 1)', '[1, 1)', '(1, 1]', '[1, 1]', '[2, 1]', '(1)', '[1)', '()', '[]', '{}', '[1, 1) | [2, 3]']:
    a = run(lambda: M1.merge(s)); b = run(lambda: M2.parse(s))
    sa = a[1] if a[0] == 'err' else f'empty={a[1].is_empty}'
    sb = b[1] if b[0] == 'err' else f'empty={b[1].is_empty} {b[1]}'
    print(f'{s!r:20s} v1 {sa:60s} | v2 {sb}')

print('== v1 merge tuple (1,1) / list [1,1] ==')
print('v1 merge((1,1))', run(lambda: M1.merge((1, 1))))
print('v1 merge((2,1))', run(lambda: M1.merge((2, 1))))
print('v2 from_pieces([(1,1,False,False)])', run(lambda: M2.from_pieces([(1, 1, False, False)])))

print('== seeded sweep: v1 raise <=> v2 empty-or-raise; set agreement otherwise ==')
rng = random.Random(4)
vals = [-2, -1, 0, 1, 2, F(1, 2), 0.5, 1.0, -0.0, float('inf'), -float('inf')]
stats = {}
mism = []
for i in range(600):
    a, b = rng.choice(vals), rng.choice(vals)
    sc, ec = rng.random() < .5, rng.random() < .5
    r1 = run(lambda: M1(a, b, start_closed=sc, end_closed=ec))
    r2 = run(lambda: M2(a, b, start_closed=sc, end_closed=ec))
    pts = sorted({a, b, -3, 3, F(1, 4), 0, 1}, key=float) if all(abs(x) != float('inf') for x in (a, b)) else [-3, 0, 1, 3]
    pts = [p for p in pts if abs(p) != float('inf')]
    if r1[0] == 'ok' and r2[0] == 'ok':
        k = 'both ok'
        if v1_members(r1[1], pts) != v2_members(r2[1], pts):
            mism.append((a, b, sc, ec, 'membership'))
    elif r1[0] == 'err' and r2[0] == 'err':
        k = 'both raise'
    elif r1[0] == 'err':
        exact_empty = (F(a) == F(b)) if all(abs(x) != float('inf') for x in (a, b)) else (a == b)
        k = f'v1 raise, v2 ok ({"a==b" if a == b else "a!=b"}, v2 empty={r2[1].is_empty})'
        # what does v1 raise for?
        k += ' / v1: ' + r1[1].split(':')[0]
        if a == b and 'after end' not in r1[1]:
            k += ' [' + r1[1][:50] + ']'
    else:
        k = f'v1 ok, v2 raise: {r2[1][:60]}'
        mism.append((a, b, sc, ec, r2[1]))
    stats[k] = stats.get(k, 0) + 1
for k, v in sorted(stats.items()):
    print(f'{v:4d}  {k}')
print('mismatches', len(mism), mism[:10])

print('== v2 composition reproducing v1 error on finite a==b open cases ==')
def strict(a, b, sc=True, ec=True):
    m = M2(a, b, start_closed=sc, end_closed=ec)
    if not m:
        raise ValueError(f'empty piece {a!r},{b!r}')
    return m
agree = total = 0
for a in [-1, 0, F(1, 3), 2.5]:
    for sc in (True, False):
        for ec in (True, False):
            r1 = run(lambda: M1(a, a, start_closed=sc, end_closed=ec))
            r2 = run(lambda: strict(a, a, sc, ec))
            total += 1; agree += (r1[0] == r2[0])
print(f'strict() agrees with v1 on raise/ok: {agree}/{total}')

# fail-ability check: a deliberately wrong expectation must be caught
wrong = M2(1, 1, end_closed=False)
assert not wrong, 'v2 [1,1) should be empty'
try:
    assert wrong, 'DELIBERATE WRONG EXPECTATION CAUGHT: v2 [1,1) is empty'
except AssertionError as e:
    print(e)
