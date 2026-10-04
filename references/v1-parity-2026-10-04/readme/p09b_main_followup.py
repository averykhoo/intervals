# follow-up of p09: closed_hull mismatches, reciprocal modulo float rounding, in-subset mismatches only at the empty set
from common import *
warnings.simplefilter('ignore')
random.seed(2026)
def flt(M):
    return MI.from_pieces((float(p.inf), float(p.sup), p.inf_closed, p.sup_closed) for p in M.pieces)
ch = []; rc = [0, 0]; sub = [0, 0, 0]
for _ in range(300):
    i = v1.random_multi_interval(-100, 100, random.randint(0, 5), 0)
    j = v1.random_multi_interval(-100, 100, random.randint(0, 5), 0)
    I, J = conv(i), conv(j)
    ch1 = i.closed_hull
    if (MI() if ch1 is None else conv(ch1)) != I.closed_hull: ch.append((str(I), str(ch1), str(I.closed_hull)))
    if 0 not in I:
        rc[0] += 1; rc[1] += conv(i.reciprocal()) == flt(I.reciprocal())
    U, X, D = I | J, I & J, I.difference(J)
    for a1, b1, A, B in [(i.union(j), j, U, J), (j, i.union(j), J, U), (i.intersection(j), i, X, I), (i.intersection(j), j, X, J), (i.difference(j), i, D, I), (i.difference(j), j, D, J)]:
        if (a1 in b1) != (A in B):
            sub[0] += 1; sub[1] += A.is_empty
            # exact: A subset of B?
            sub[2] += (A in B) == all(p in B for p in probes_of(A))
print('closed_hull mismatches:', ch)
print(f'reciprocal (0 not in i), v2 rounded to float: {rc[1]}/{rc[0]} equal to v1')
print(f'in-subset mismatches {sub[0]}, of which left operand empty {sub[1]}, v2 consistent with point probes {sub[2]}')
print('v1: empty in empty ->', V1() in V1(), '; empty in [1,2] ->', V1() in V1(1, 2), '| v2:', MI() in MI(), MI() in MI(1, 2))
