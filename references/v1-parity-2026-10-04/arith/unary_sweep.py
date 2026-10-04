"""random + hand differential sweep of reciprocal, -A, +A, abs(A), ~A (v1 vs v2), compared as sets"""
import sys, os, random, warnings, math, collections
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
warnings.simplefilter('ignore')

def brute_recip_member(A2, z):
    """exact: is z = 1/x for some finite nonzero x in A (z finite nonzero), or z == 0 never (x finite)"""
    if z == 0:
        return False
    return (1 / z) in A2

def brute_abs_member(A2, z):
    return z >= 0 and (z in A2 or -z in A2)

def brute_neg_member(A2, z):
    return -z in A2

def brute_compl_member(A2, z):
    return z not in A2

UN = {
    'reciprocal': (lambda m: m.reciprocal(), lambda m: m.reciprocal(), brute_recip_member),
    'neg': (lambda m: -m, lambda m: -m, brute_neg_member),
    'pos': (lambda m: +m, lambda m: +m, lambda A, z: z in A),
    'abs': (lambda m: abs(m), lambda m: abs(m), brute_abs_member),
    'invert': (lambda m: ~m, lambda m: ~m, brute_compl_member),
}

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1788)
stats = collections.Counter(); ex = collections.defaultdict(list)
cases = [rand_pieces(rng) for _ in range(400)]
cases += [[], [(F(0), F(0), True, True)], [(F(0), F(1), True, True)], [(F(0), F(1), False, True)], [(F(-1), F(0), True, False)],
          [(F(-1), F(1), True, True)], [(-INF, INF, False, False)], [(F(0), INF, False, False)], [(-INF, F(0), False, True)],
          [(F(1), F(2), True, False)], [(F(-2), F(-1), False, True), (F(1), F(2), True, False)], [(F(-1), F(0), True, True), (F(1), F(2), False, False)]]
for pa in cases:
    a1, a2 = mk1(pa), mk2(pa)
    assert not finite_diff(a1, a2)
    for name, (f1, f2, brute) in UN.items():
        key = fmt_pieces(pa)
        try:
            c1 = f1(a1); c1._consistency_check(); e1 = None
        except Exception as e:
            c1, e1 = None, f'{type(e).__name__}: {e}'
        try:
            c2 = f2(a2); e2 = None
        except Exception as e:
            c2, e2 = None, f'{type(e).__name__}: {e}'
        if e1 or e2:
            cat = f'{name} raise v1={bool(e1)} v2={bool(e2)}'
            stats[cat] += 1; ex[cat].append(f'{key}: v1 {e1 or c1} v2 {e2 or c2}'); continue
        pts = probe_points(v1_ends(c1), v2_ends(c2), [e for p in pa for e in p[:2]])
        # brute force oracle on every probe point
        bad1 = [x for x in pts if (x in c1) != brute(a2, x)]
        bad2 = [x for x in pts if (x in c2) != brute(a2, x)]
        idf = inf_diff(c1, c2)
        if not bad1 and not bad2:
            cat = f'{name} agree (both match brute force)' + (' except at +-inf' if idf else '')
        elif bad1 and not bad2:
            cat = f'{name} v1 wrong, v2 right'
        elif bad2 and not bad1:
            cat = f'{name} V2 WRONG, v1 right'
        else:
            cat = f'{name} both wrong'
        stats[cat] += 1
        if 'agree' not in cat or idf:
            ex[cat].append(f'{key}: v1 {c1}  v2 {c2}  v1-bad {[str(x) for x in bad1[:3]]}  v2-bad {[str(x) for x in bad2[:3]]} infdiff {idf}')
for k in sorted(stats):
    print(f'{stats[k]:5d}  {k}')
for k in sorted(ex):
    print('==', k)
    for e in ex[k][:5]:
        print('   ', e)
# self-check: a deliberately wrong oracle must be caught
wrong = [x for x in probe_points([0, 1]) if (x in -mk2([(F(0), F(1), True, True)])) != (x in mk2([(F(0), F(1), True, True)]))]
print('self-check (neg vs identity disagree somewhere):', bool(wrong))
