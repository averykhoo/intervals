# v1 README TODO items: affine extended reals / degenerate [inf]; 1/[-2,2) keeping the gap; 1/0 = [inf]?;
# negative zero; the cardinality "hyperreal"; adjoining()/adjacent/intersecting/overlapping
from common import *
def run(f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = f()
        r = 'NotImplemented' if r is NotImplemented else r
        return (str(r) + (f'  [warn: {",".join(sorted({type(x.message).__name__ for x in w}))}]' if w else ''))
    except Exception as e: return f'{type(e).__name__}: {str(e)[:60]}'

print('--- affine extended reals, [inf] (TODO: "yes we allow degen intervals at inf")')
for name, f1, f2 in [
    ('[inf]', lambda: V1(math.inf), lambda: MI(math.inf)),
    ('[1, inf]', lambda: V1(1, math.inf), lambda: MI(1, math.inf)),
    ('[1, inf)', lambda: V1(1, math.inf, end_closed=False), lambda: MI(1, math.inf, end_closed=False)),
    ('inf in [1, inf)', lambda: math.inf in V1(1, math.inf, end_closed=False), lambda: math.inf in MI(1, math.inf, end_closed=False)),
    ('[1,inf) == [1,inf]', lambda: 'n/a (v1 cannot build [1,inf])', lambda: MI(1, math.inf, end_closed=False) == MI(1, math.inf)),
]:
    print(f'{name:22s} v1: {run(f1):55s} v2: {run(f2)}')
# v1 with its module flag flipped: can v1 do it at all?
v1.INFINITY_IS_NOT_FINITE = False
print('v1 with INFINITY_IS_NOT_FINITE=False: [inf] ->', run(lambda: V1(math.inf)), '| [1,inf] ->', run(lambda: V1(1, math.inf)),
      '| 1/[1,inf] ->', run(lambda: V1(1, math.inf).reciprocal()), '(v2:', run(lambda: MI(1, math.inf).reciprocal()), ')')
v1.INFINITY_IS_NOT_FINITE = True

print('--- reciprocal (TODO "MAYBE FIX: 1 / [-2, 2) = (-inf, -0.5], (0.5, inf) <- drop the gap")')
for name, f1, f2 in [
    ('1/[-2,2)', lambda: 1 / V1(-2, 2, end_closed=False), lambda: 1 / MI(-2, 2, end_closed=False)),
    ('[-2,2).reciprocal()', lambda: V1(-2, 2, end_closed=False).reciprocal(), lambda: MI(-2, 2, end_closed=False).reciprocal()),
    ('1/[-2,2]', lambda: 1 / V1(-2, 2), lambda: 1 / MI(-2, 2)),
    ('1/[0]', lambda: 1 / V1(0), lambda: 1 / MI(0)),
    ('-1/[0]', lambda: -1 / V1(0), lambda: -1 / MI(0)),
    ('1/[0,2]', lambda: 1 / V1(0, 2), lambda: 1 / MI(0, 2)),
    ('1/(0,2]', lambda: 1 / V1(0, 2, start_closed=False), lambda: 1 / MI(0, 2, start_closed=False)),
    ('[1,2] / 0', lambda: V1(1, 2) / 0, lambda: MI(1, 2) / 0),
]:
    print(f'{name:22s} v1: {run(f1):55s} v2: {run(f2)}')
# seeded sweep: reciprocal at finite nonzero y: y in 1/A  iff  1/y in A (both semantics agree there)
rng = random.Random(17); n = v1w = v2w = 0; ex = []
for _ in range(400):
    ps = []
    for _ in range(rng.randint(1, 3)):
        a, b = sorted(Fraction(rng.randint(-10, 10), 2) for _ in range(2))
        ps.append((a, b, True, True) if a == b else (a, b, rng.random() < .5, rng.random() < .5))
    A2 = MI.from_pieces(ps); A1 = tov1(A2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r1 = conv(A1.reciprocal()); r2 = A2.reciprocal()
    pts = {1 / p for p in probes_of(A2) if p != 0 and math.isfinite(p)} | {Fraction(p) for p in probes_of(r2) if p != 0 and math.isfinite(p)}
    n += 1; b1 = b2 = False
    for y in pts:
        t = (1 / Fraction(y)) in A2
        b1 |= (y in r1) != t; b2 |= (y in r2) != t
    v1w += b1; v2w += b2
    if b1 and len(ex) < 3: ex.append((str(A2), str(r1), str(r2)))
print(f'reciprocal sweep: cases {n} v1_wrong {v1w} v2_wrong {v2w}', ex)
assert (Fraction(1, 4) in MI(-2, 2).reciprocal()) is False  # sabotage: the gap is checked

print('--- negative zero (TODO: "to support negative zero ...")')
for name, f1, f2 in [
    ('[-0.0]', lambda: V1(-0.0).endpoints, lambda: MI(-0.0).cuts),
    ('copysign of inf of [-0.0]', lambda: math.copysign(1, V1(-0.0).infimum), lambda: math.copysign(1, MI(-0.0).inf)),
    ('[-1, -0.0]', lambda: V1(-1, -0.0), lambda: MI(-1, -0.0)),
    ('1/[-1, -0.0]', lambda: 1 / V1(-1, -0.0), lambda: 1 / MI(-1, -0.0)),
    ('1/[-0.0]', lambda: 1 / V1(-0.0), lambda: 1 / MI(-0.0)),
]:
    print(f'{name:22s} v1: {run(f1):55s} v2: {run(f2)}')

print('--- cardinality (v1) vs size (v2)  (TODO: "cardinality measure is basically a hyperreal a*w + b + c*eps")')
for txt in ['(1, inf)', '[1, 2)', '[1, 2]', '(1, 2)', '[1]', '{ [1, 2) , [2] , (2, 3) }', '[1, 3)', '(-2, -1]', '(-inf, 5]', '(-inf, inf)', '(-inf, -1)', '{}']:
    B = MI.parse(txt.replace('inf]', 'inf)').replace('[-inf', '(-inf'))
    try: c1 = tov1(B).cardinality
    except Exception as e: c1 = f'{type(e).__name__}'
    s = B.size
    print(f'{txt:28s} v1 {str(c1):26s} v2 {s}   v1==(rays, length, 2*points)? {c1 == (s.rays, s.length, 2 * s.points) if isinstance(c1, tuple) else "-"}')
# the v1 docstring's invariants, on v2's size
inv = [('[1,2)', '[2,3)'), ('[1,2)', '{ [1] , (1, 2) }'), ('[1,3)', '{ [1, 2) , [2, 3) }'), ('{ [1, 2) , [2] , (2, 3) }', '[1, 3)'), ('[1,2)', '(-2, -1]')]
print('v1 docstring invariants hold in v2:', [MI.parse(a).size == MI.parse(b).size for a, b in inv],
      ' and in v1:', [tov1(MI.parse(a)).cardinality == tov1(MI.parse(b)).cardinality for a, b in inv])
assert MI.parse('[1,2]').size != MI.parse('[1,2)').size  # sabotage

print('--- adjoining()/adjacent/intersecting/overlapping (TODO)')
for a, b in [('[0, 1)', '[1, 2]'), ('[0, 1]', '[1, 2]'), ('[0, 1)', '(1, 2]'), ('[0, 2]', '[1, 3]'), ('[0, 3]', '[1, 2]'), ('[0, 1]', '[2, 3]')]:
    A, B = MI.parse(a), MI.parse(b)
    o1 = tov1(A).overlaps(tov1(B)); o1a = tov1(A).overlaps(tov1(B), or_adjacent=True)
    print(f'{a} vs {b}: v1 overlaps={o1} or_adjacent={o1a} | v2 adjoins={A.adjoins(B)} overlaps={A.overlaps(B)} allen={A.allen(B).name}')
