"""copy, __sizeof__, _consistency_check, CONSISTENCY_CHECK, __bool__/__float__/__int__/__complex__"""
from common import *
import copy, pickle, sys, random
from intervals import kernel
from intervals.cuts import Cut, Side
inf = math.inf

print('=== copy')
A1 = M1.merge('{ [1, 2) , (3, 4] }'); B1 = A1.copy(); A1.update(M1(10))
print('v1: copy independent of later in-place update:', str(B1), '| original now', str(A1))
A2 = M2.parse('{ [1, 2) , (3, 4] }')
print('v2: has .copy?', hasattr(A2, 'copy'), '| copy.copy == A:', copy.copy(A2) == A2, '| deepcopy == A:', copy.deepcopy(A2) == A2,
      '| pickle round trip:', pickle.loads(pickle.dumps(A2)) == A2)
print('v2: in-place mutation refused:', outcome(lambda: setattr(A2, '_cuts', ())), '| A2 |= 10 rebinds, old object kept:', end=' ')
C2 = A2; C2 |= 10; print(str(A2), '/', str(C2))
print('v2: has update?', hasattr(A2, 'update'))

print('=== __sizeof__')
for n in (0, 1, 5, 50):
    a1 = M1.merge(*[[2 * i, 2 * i + 1] for i in range(n)]); a2 = v1_to_v2(a1)
    print(f'  pieces={n:3d}  v1 a.__sizeof__()={a1.__sizeof__():5d} (endpoints list {a1.endpoints.__sizeof__()})'
          f'  v2 sys.getsizeof(a)={sys.getsizeof(a2):4d}  a.__sizeof__()={a2.__sizeof__():3d}  a.cuts.__sizeof__()={a2.cuts.__sizeof__():5d}')

print('=== _consistency_check on invalid structures (v1 endpoints vs v2 from_cuts)')
B, A_ = Side.BELOW, Side.ABOVE
cases = [
    ('odd length', [(1, 0)], (Cut(1, B),)),
    ('unsorted', [(3, 0), (4, 0), (1, 0), (2, 0)], (Cut(3, B), Cut(4, A_), Cut(1, B), Cut(2, A_))),
    ('start after end', [(2, 0), (1, 0)], (Cut(2, B), Cut(1, A_))),
    ('empty piece (start == end cut)', [(1, 1), (1, -1)], (Cut(1, A_), Cut(1, A_))),
    ('touching pieces not merged', [(1, 0), (2, -1), (2, 0), (3, 0)], (Cut(1, B), Cut(2, B), Cut(2, B), Cut(3, A_))),
    ('closed at inf (v1 flag on)', [(1, 0), (inf, 0)], (Cut(1, B), Cut(inf, A_))),
    ('bad epsilon / side', [(1, 2), (2, 0)], None),
    ('non-real value', [('a', 0), ('b', 0)], None),
    ('nan value', [(math.nan, 0), (math.nan, 0)], None),
    ('-0.0 value', [(-0.0, 0), (1, 0)], None),
]
for label, ep, cuts in cases:
    m = M1(); m.endpoints = list(ep)
    o1 = outcome(m._consistency_check)
    if cuts is not None:
        o2 = outcome(lambda: M2.from_cuts(cuts)); v = kernel.is_valid(cuts)
    else:
        o2, v = ('n/a', '-'), '-'
    print(f'  {label:32s} v1 check -> {o1[0] if o1[0]=="ok" else "RAISE " + o1[1][:40]:40s} v2 from_cuts -> {str(o2[1])[:55]:55s} is_valid={v}')
print('  v2 Cut(nan, BELOW):', outcome(lambda: Cut(math.nan, B)), '| Cut("a", BELOW):', outcome(lambda: Cut('a', B)),
      '| Cut(-0.0, BELOW).value:', Cut(-0.0, B).value, '| Side(2):', outcome(lambda: Cut(1, 2)))

print('=== CONSISTENCY_CHECK=False in v1: the check is skipped')
v1.CONSISTENCY_CHECK = False
m = M1(); m.endpoints = [(3, 0), (1, 0)]
print('  v1 with flag off, invalid endpoints: check ->', outcome(m._consistency_check), '| str ->', outcome(lambda: str(m)))
v1.CONSISTENCY_CHECK = True

print('=== __bool__, __float__, __int__, __complex__')
samples = [('empty', M1(), M2()), ('point int 3', M1(3), M2(3)), ('point Fraction 7/2', M1(F(7, 2)), M2(F(7, 2))),
           ('point float -2.5', M1(-2.5), M2(-2.5)), ('point -0.0', M1(-0.0), M2(-0.0)),
           ('point 10**20+1', M1(10 ** 20 + 1), M2(10 ** 20 + 1)), ('point -7/2', M1(F(-7, 2)), M2(F(-7, 2))),
           ('interval [1,2]', M1(1, 2), M2(1, 2)), ('two points {1,2}', M1.merge({1, 2}), M2.from_pieces([(1, 1), (2, 2)])),
           ('point 1e308*10? no: 1.5e308', M1(1.5e308), M2(1.5e308)), ('point 2**1100 (int)', None, M2(2 ** 1100)),
           ('point inf (v2 only)', None, M2(inf))]
t = Tally('conversions')
for label, a1, a2 in samples:
    row = []
    for fn in (bool, float, int, complex):
        o1 = outcome(lambda: fn(a1)) if a1 is not None else ('n/a', '')
        o2 = outcome(lambda: fn(a2))
        same = (o1[0] == o2[0] == 'ok' and o1[1] == o2[1] and type(o1[1]) is type(o2[1])) or (o1[0] == o2[0] == 'raise')
        if a1 is not None:
            t.check(same, (label, fn.__name__, o1, o2))
        row.append(f'{fn.__name__}: v1={o1[1] if o1[0] != "raise" else "RAISE"} v2={o2[1] if o2[0] == "ok" else "RAISE " + o2[1][:30]}')
    print(f'  {label:28s}', ' | '.join(row))
t.report(10)
print('  exact int(v1(10**20+1)) =', outcome(lambda: int(M1(10 ** 20 + 1))), ' exact value 10**20+1 =', 10 ** 20 + 1)

print('=== sabotage')
s = Tally('sabotage'); s.check(int(M2(F(7, 2))) == 4, 'int 7/2 is not 4'); assert s.report() == 1; print('sabotage caught')
