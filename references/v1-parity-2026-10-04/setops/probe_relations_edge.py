"""hand-picked edges for issubset/issuperset/isdisjoint/__contains__/overlapping"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
from decimal import Decimal

def run(label, f1, f2):
    try: r1 = f1()
    except Exception as e: r1 = f'raise {type(e).__name__}: {e}'
    try: r2 = f2()
    except Exception as e: r2 = f'raise {type(e).__name__}: {e}'
    flag = '' if str(r1) == str(r2) else '   <-- DIFF'
    print(f'{label:55s} v1={r1!s:40.40s} v2={r2!s:40.40s}{flag}')
    return r1, r2

third = Fraction(1, 3)
big = 2 ** 60 + 1
run('issubset(Fraction 1/3) of [1/3]', lambda: M1(third).issubset(third), lambda: M2(third).issubset(third))
run('issubset(2**60+1) of [2**60+1]', lambda: M1(big).issubset(big), lambda: M2(big).issubset(big))
run('issubset(2**60) of [2**60+1] (false)', lambda: M1(big).issubset(2**60), lambda: M2(big).issubset(2**60))
run('issubset(1) of {[1],[1]} single', lambda: M1(1).issubset(1), lambda: M2(1).issubset(1))
run('issubset(1.0) of [1]', lambda: M1(1).issubset(1.0), lambda: M2(1).issubset(1.0))
run('issubset(inf) of empty', lambda: M1().issubset(inf), lambda: M2().issubset(inf))
run('empty issubset(empty)', lambda: M1().issubset(M1()), lambda: M2().issubset(M2()))
run('empty issuperset(empty)', lambda: M1().issuperset(M1()), lambda: M2().issuperset(M2()))
run('empty in empty', lambda: M1() in M1(), lambda: M2() in M2())
run('empty in [0,1]', lambda: M1() in M1(0, 1), lambda: M2() in M2(0, 1))
run('[0,1] issuperset(1)', lambda: M1(0, 1).issuperset(1), lambda: M2(0, 1).issuperset(1))
run('[0,1) issuperset(1)', lambda: M1(0, 1, end_closed=False).issuperset(1), lambda: M2(0, 1, end_closed=False).issuperset(1))
run('isdisjoint(empty, empty)', lambda: M1().isdisjoint(M1()), lambda: M2().isdisjoint(M2()))
run('[0,1) isdisjoint [1,2]', lambda: M1(0, 1, end_closed=False).isdisjoint(M1(1, 2)), lambda: M2(0, 1, end_closed=False).isdisjoint(M2(1, 2)))
# contains: left-hand types
X1 = M1(0, 1) .union(M1(2, 3, start_closed=False)); X2 = M2(0, 1) | M2(2, 3, start_closed=False)
for v in [0, 1, 2, 3, 2.5, Fraction(5, 2), -0.0, inf, -inf, float('nan'), True, False, Decimal('0.5'), '0.5', [0, 1], (0, 1), {0}, None, 1+0j]:
    run(f'{v!r} in X', lambda v=v: v in X1, lambda v=v: v in X2)
# numpy scalars
import numpy as np
for v in [np.float64(0.5), np.int64(1), np.float32(2.5), np.bool_(True)]:
    run(f'{v!r} in X', lambda v=v: v in X1, lambda v=v: v in X2)
# unbounded
U1 = M1(1, inf, end_closed=False); U2 = M2(1, inf, end_closed=False)
run('inf in [1,inf)', lambda: inf in U1, lambda: inf in U2)
run('10**400 in [1,inf)', lambda: 10 ** 400 in U1, lambda: 10 ** 400 in U2)
# overlapping with scalar infinities
run('[1,inf).overlapping(inf)', lambda: str(U1.overlapping(inf)), lambda: str(M2().union(*[p for p in U2 if p.overlaps(M2(inf))])))
run('[1,inf).overlapping(inf, adj)', lambda: str(U1.overlapping(inf, or_adjacent=True)), lambda: str(M2().union(*[p for p in U2 if p.overlaps(M2(inf)) or p.adjoins(M2(inf))])))
L1 = M1(-inf, 0, start_closed=False).union(M1(5)); L2 = M2(-inf, 0, start_closed=False) | M2(5)
run('(-inf,0]u[5].overlapping(-inf, adj)', lambda: str(L1.overlapping(-inf, or_adjacent=True)), lambda: str(M2().union(*[p for p in L2 if p.overlaps(M2(-inf)) or p.adjoins(M2(-inf))])))
run('(-inf,0]u[5].overlapping(inf, adj)', lambda: str(L1.overlapping(inf, or_adjacent=True)), lambda: str(M2().union(*[p for p in L2 if p.overlaps(M2(inf)) or p.adjoins(M2(inf))])))
run('overlapping(empty)', lambda: str(X1.overlapping(M1())), lambda: str(M2().union(*[p for p in X2 if p.overlaps(M2())])))
run('overlapping("1")', lambda: str(X1.overlapping('1')), lambda: 'n/a')
run('overlapping(nan)', lambda: str(X1.overlapping(float('nan'))), lambda: 'n/a')
# overlapping adjacency against an inner piece of B: does p.adjoins(B) (whole-set) miss it?
A1 = M1(1, 2, start_closed=False, end_closed=False); A2 = M2(1, 2, start_closed=False, end_closed=False)
B1 = M1(0, 1).union(M1(5, 6)); B2 = M2(0, 1) | M2(5, 6)
run('(1,2).overlapping([0,1]u[5,6], adj)', lambda: str(A1.overlapping(B1, or_adjacent=True)),
    lambda: str(M2().union(*[p for p in A2 if p.overlaps(B2) or any(p.adjoins(q) for q in B2)])))
run('  naive p.adjoins(B) spelling', lambda: str(A1.overlapping(B1, or_adjacent=True)),
    lambda: str(M2().union(*[p for p in A2 if p.overlaps(B2) or p.adjoins(B2)])))
B1 = M1(-5, -4).union(M1(0, 1)).union(M1(5, 6)); B2 = M2(-5, -4) | M2(0, 1) | M2(5, 6)
run('(1,2).overlapping([-5,-4]u[0,1]u[5,6], adj)', lambda: str(A1.overlapping(B1, or_adjacent=True)),
    lambda: str(M2().union(*[p for p in A2 if p.overlaps(B2) or any(p.adjoins(q) for q in B2)])))
run('  naive p.adjoins(B) spelling', lambda: str(A1.overlapping(B1, or_adjacent=True)),
    lambda: str(M2().union(*[p for p in A2 if p.overlaps(B2) or p.adjoins(B2)])))
