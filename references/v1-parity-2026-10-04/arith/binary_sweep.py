"""random differential sweep of + - * / (both operands sets, and number on either side), v1 vs v2, compared as sets"""
import sys, os, random, warnings, operator, math, collections
sys.path.insert(0, os.path.dirname(__file__))
from common import *

warnings.simplefilter('ignore')
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

def witness(op, A, B, z):
    """an exact x in A, y in B (finite) with x op y == z, searched over candidate x"""
    cands = [x for x in probe_points(v2_ends(A), v2_ends(B), [z]) if x in A]
    for x in cands:
        ys = []
        if op == '+': ys = [z - x]
        elif op == '-': ys = [x - z]
        elif op == '*':
            if x != 0: ys = [z / x]
            elif z == 0: ys = [y for y in probe_points(v2_ends(B)) if y in B][:1]
        elif op == '/':
            if z != 0: ys = [x / z]
            elif x == 0: ys = [y for y in probe_points(v2_ends(B)) if y in B and y != 0][:1]
        for y in ys:
            if y in B and not (op == '/' and y == 0):
                return (x, y)
    return None

def run(seed, n):
    rng = random.Random(seed)
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for i in range(n):
        pa, pb = rand_pieces(rng), rand_pieces(rng)
        mode = rng.choice(['ss', 'ss', 'sn', 'ns'])
        if mode == 'sn':
            pb = [(v, v, True, True) for v in [rng.choice(VALS[1:-1])]]
        if mode == 'ns':
            pa = [(v, v, True, True) for v in [rng.choice(VALS[1:-1])]]
        a1, b1, a2, b2 = mk1(pa), mk1(pb), mk2(pa), mk2(pb)
        # input sanity: same sets
        assert not finite_diff(a1, a2) and not finite_diff(b1, b2), (pa, pb)
        for name, op in OPS.items():
            l1 = pa[0][0] if mode == 'ns' else a1
            r1 = pb[0][0] if mode == 'sn' else b1
            l2 = pa[0][0] if mode == 'ns' else a2
            r2 = pb[0][0] if mode == 'sn' else b2
            key = f'{fmt_pieces(pa)} {name} {fmt_pieces(pb)} ({mode})'
            try:
                c1 = op(l1, r1); c1._consistency_check(); e1 = None
            except Exception as e:
                c1, e1 = None, f'{type(e).__name__}: {e}'
            try:
                c2 = op(l2, r2); e2 = None
            except Exception as e:
                c2, e2 = None, f'{type(e).__name__}: {e}'
            if e1 or e2:
                cat = f'{name} raise v1={bool(e1)} v2={bool(e2)}'
                stats[cat] += 1
                examples[cat].append(f'{key}: v1 {e1 or c1}  v2 {e2 or c2}')
                continue
            fd = finite_diff(c1, c2)
            idf = inf_diff(c1, c2)
            if not fd and not idf:
                stats[f'{name} agree'] += 1
                continue
            if not fd:
                stats[f'{name} differ only at +-inf'] += 1
                examples[f'{name} differ only at +-inf'].append(f'{key}: v1 {c1} v2 {c2}')
                continue
            # classify the finite differences
            only2 = [x for x, i1, i2 in fd if i2]
            only1 = [x for x, i1, i2 in fd if i1]
            zero_div = name == '/' and (0 in b2)
            nanny = any(isinstance(e, float) and math.isnan(e) for e in v1_ends(c1))
            w2 = [witness(name, a2, b2, z) for z in only2]
            w1 = [witness(name, a2, b2, z) for z in only1]
            if zero_div:
                cat = f'{name} differ: divisor holds 0'
            elif nanny:
                cat = f'{name} differ: v1 has nan endpoints'
            elif all(w is not None for w in w2) and all(w is None for w in w1):
                cat = f'{name} differ: v2 right (witness for every v2-only point, none for v1-only)'
            else:
                cat = f'{name} differ: UNRESOLVED'
            stats[cat] += 1
            examples[cat].append(f'{key}: v1 {c1}  v2 {c2}  v2-only {[str(z) for z in only2[:3]]} wit {w2[:2]}  v1-only {[str(z) for z in only1[:3]]} wit {w1[:2]}')
    return stats, examples

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1788
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    stats, ex = run(seed, n)
    for k in sorted(stats):
        print(f'{stats[k]:5d}  {k}')
    for k in sorted(ex):
        print('==', k)
        for e in ex[k][:4]:
            print('   ', e)
