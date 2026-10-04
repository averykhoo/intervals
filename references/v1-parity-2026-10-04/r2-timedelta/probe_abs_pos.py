"""abs(T) and +T: v1 TimeDeltaInterval has neither (TypeError); v1 composition out.interval = abs(A.interval) /
A.copy(); v2 abs(A), +A. 500 seeded random sets, exact structure + oracle (x in abs(A) iff x >= 0 and (x in A or -x in A))"""
import sys
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-timedelta']
import io, contextlib, random
from fractions import Fraction
with contextlib.redirect_stdout(io.StringIO()):
    import probe_rdiv_neg as P
T1, T2 = P.T1, P.T2
rng = random.Random(99)
fails = checks = 0
for i in range(500):
    pieces = P.rand_pieces(rng)
    a1, a2 = P.build1(pieces), P.build2(pieces)
    m1 = T1()
    m1.interval = abs(a1.interval)
    r2 = abs(a2)
    c1, c2 = P.canon_mi1(m1.interval), P.canon_mi2(r2.seconds)
    sp = [(Fraction(a), sc, Fraction(b), ec) for a, b, sc, ec in pieces]
    ok = P.same_structure(c1, c2) and type(r2) is T2 and all(
        P.member(c2, x) == (x >= 0 and (P.member(sp, x) or P.member(sp, -x))) for x in (Fraction(k, 4) for k in range(-30, 31)))
    ok = ok and (+a2) == a2 and P.same_structure(P.canon_mi1(a1.copy().interval), P.canon_mi2((+a2).seconds))
    checks += 1
    if not ok:
        fails += 1
        if fails < 5:
            print('MISMATCH', pieces, c1, c2)
print('sabotaged expectation caught:', not (abs(P.T2(-P.H, P.H)) == P.T2(-P.H, P.H)))
print(f'probe_abs_pos: {checks} cases, {fails} mismatches')
