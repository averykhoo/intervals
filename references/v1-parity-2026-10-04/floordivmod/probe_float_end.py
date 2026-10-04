"""two float cases from probe_mod_inf_float.py: is v2's rounded end flag right?"""
from common import *
from intervals import modulo
for A, B in [([(7.5, False, 7.9, True)], [(1.1, True, 1.1 * 3, True)]), ([(7.5, False, 7.9, True)], [(1.1, True, 1.1 * 3, False)]),
             ([(3.75, False, 7.9, True), (10.2, False, 10.8, True)], [(0.2, True, 0.2 * 3, True)]),
             ([(3.75, False, 7.9, True), (10.2, False, 10.8, True)], [(0.2, True, 0.2 * 3, False)])]:
    Ax = [(Fraction(a), ac, Fraction(b), bc) for a, ac, b, bc in A]
    Bx = [(Fraction(a), ac, Fraction(b), bc) for a, ac, b, bc in B]
    r2 = mk2(A) % mk2(B)
    rx = mk2(Ax) % mk2(Bx)
    ro = v2.OutwardMultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in A]) % v2.OutwardMultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in B])
    s1, r1 = run(lambda: mk1(A) % mk1(B))
    print('A', A, 'B', B)
    print('  v2 float   ', r2, show(v2_pieces(r2)))
    print('  v2 exact   ', show(v2_pieces(rx)), [float(p[2]) for p in v2_pieces(rx)])
    print('  v2 outward ', ro)
    print('  v1         ', show(v1_pieces(r1)) if s1 == 'ok' else r1)
    for lo, lc, hi, hc in v2_pieces(r2):
        print('    float end', hi, 'attained exactly (oracle):', attained_mod(hi, Ax, Bx), '| in exact result:', contains(v2_pieces(rx), hi))
