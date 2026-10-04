"""v1 merge is a classmethod: A.merge(B) ignores A. v2 has no merge; union/intersection are methods
taking self. compare as sets via exact membership at ends, just inside/outside, midpoints."""
import sys, random, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import intervals as v2

def pieces_v1(A):
    e = A.endpoints
    return [(e[i][0], e[i][1] == 0, e[i+1][0], e[i+1][1] == 0) for i in range(0, len(e), 2)]

def mk(ps):
    a = v1.MultiInterval(); b = v2.MultiInterval()
    for (s, sc, t, tc) in ps:
        if s == t:
            a = a.union(v1.MultiInterval(s)); b = b | v2.MultiInterval(s)
        else:
            a = a.union(v1.MultiInterval(s, t, start_closed=sc, end_closed=tc))
            b = b | v2.MultiInterval(s, t, start_closed=sc, end_closed=tc)
    return a, b

def probes(*sets):
    pts = set()
    for ps in sets:
        for (s, sc, t, tc) in ps:
            for x in (s, t):
                pts |= {F(x), F(x) - F(1, 7), F(x) + F(1, 7)}
            pts.add((F(s) + F(t)) / 2)
    return sorted(pts)

def same(a1, a2, pts):
    for p in pts:
        m1 = float(p) in a1 if p.denominator == 1 else p in a1
        m2 = p in a2
        if m1 != m2:
            return p
    return None

def rand_ps(rng):
    out = []
    for _ in range(rng.randint(0, 3)):
        s = rng.randint(-5, 5); t = s + rng.randint(0, 4)
        sc, tc = (True, True) if s == t else (rng.random() < .5, rng.random() < .5)
        out.append((s, sc, t, tc))
    return out

rng = random.Random(20261004)
n = bad_inst = bad_cls = bad_inter = ign = 0
for _ in range(400):
    pa, pb, pc = rand_ps(rng), rand_ps(rng), rand_ps(rng)
    A1, A2 = mk(pa); B1, B2 = mk(pb); C1, C2 = mk(pc)
    pts = probes(pa, pb, pc)
    n += 1
    # v1 instance call: A.merge(B, C) is merge(B, C) only
    r1 = A1.merge(B1, C1)
    if same(r1, B2 | C2, pts) is not None: bad_inst += 1
    if same(r1, A2 | B2 | C2, pts) is not None: ign += 1     # counts cases where A is ignored visibly
    # v1 class call == v2 unbound union / instance union
    r1c = v1.MultiInterval.merge(A1, B1, C1)
    if same(r1c, v2.MultiInterval.union(A2, B2, C2), pts) is not None: bad_cls += 1
    if same(r1c, A2.union(B2, C2), pts) is not None: bad_cls += 1
    # n_overlaps=k via instance: still ignores A. intersection (k=number of args) vs v2 intersection
    r1i = v1.MultiInterval.merge(A1, B1, C1, n_overlaps=3)
    if same(r1i, A2.intersection(B2, C2), pts) is not None: bad_inter += 1
print('cases', n, 'instance-merge != union(B,C):', bad_inst, '| instance-merge != A|B|C (A ignored, visible):', ign)
print('class merge != v2 union:', bad_cls, '| merge(n_overlaps=3) != intersection:', bad_inter)

# hand cases
A1, A2 = mk([(0, True, 1, True)]); B1, B2 = mk([(5, True, 6, False)])
print('v1 A.merge(B):', A1.merge(B1), ' v1 A.union(B):', A1.union(B1), ' v2 A.union(B):', A2.union(B2))
print('v1 A.merge():', repr(str(A1.merge())), ' v2 A.union():', A2.union())
print('v1 MultiInterval.merge():', str(v1.MultiInterval.merge()))
try: print('v2 MultiInterval.union():', v2.MultiInterval.union())
except Exception as e: print('v2 MultiInterval.union():', type(e).__name__, e)
try: print('v2 MultiInterval.union(1, A2):', v2.MultiInterval.union(1, A2))
except Exception as e: print('v2 MultiInterval.union(1, A2):', type(e).__name__, e)
print('v2 MultiInterval(1).union(A2):', v2.MultiInterval(1).union(A2), ' v1 merge(1, A1):', v1.MultiInterval.merge(1, A1))
print('v2 hasattr merge:', hasattr(v2.MultiInterval, 'merge'))
# sabotage: a wrong expectation must be caught. instance-merge claimed equal to A|B|C must fail somewhere.
assert ign > 0, 'sabotage: probe cannot see that A is ignored'
A1, A2 = mk([(0, True, 1, True)])
assert same(A1, v2.MultiInterval(0, 1, end_closed=False), probes([(0, True, 1, True)])) == 1, 'sabotage: membership compare is blind'
print('sabotage checks caught')
