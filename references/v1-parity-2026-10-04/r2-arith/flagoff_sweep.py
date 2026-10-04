"""v1 with INFINITY_IS_NOT_FINITE = False (closed +-inf ends, [inf] points legal) vs v2, for + - * / and reciprocal,
set op set, set op number, number op set; numbers include +-inf and 0. each result is checked against the exact oracle
(oracle.py) at every test point incl. +-inf. usage: flagoff_sweep.py SEED N [sabotage]"""
import sys, os, random, collections, operator
sys.path.insert(0, os.path.dirname(__file__))
from oracle import *
v1.INFINITY_IS_NOT_FINITE = False
warnings.simplefilter('ignore')
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
VALS = [-INF, F(-3), F(-2), F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(2), F(3), INF]
SAB = len(sys.argv) > 3

def rand_pieces(rng):
    out = []
    for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
        a, b = sorted(rng.sample(VALS, 2))
        if rng.random() < 0.18:
            v = rng.choice(VALS); out.append((v, v, True, True)); continue
        out.append((a, b, rng.random() < .5, rng.random() < .5))
    return out

def run_op(f, l, r):
    try:
        c = f(l, r)
        if c is NotImplemented: return None, 'NotImplemented'
        if hasattr(c, '_consistency_check'):
            c._consistency_check()
            e = c.endpoints
            if any(e[i][1] not in (0, 1) or e[i + 1][1] not in (0, -1) for i in range(0, len(e), 2)):
                return None, f'MALFORMED endpoints {e}'
        return c, None
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:60]}'

def classify(name, c1, e1, c2, e2, A, B, extra, mem_fn):
    if e2: return 'v2 RAISES', None
    pts = test_points(ends2(c2), extra, *( [ends1(c1)] if c1 is not None else []))
    v2_bad = [z for z in pts if (z in c2) != mem_fn(z, True)]
    if v2_bad: return 'v2 WRONG vs oracle', v2_bad[:3]
    if e1: return 'v1 raises, v2 = oracle', None
    d = [z for z in pts if v1_mem(c1, z) != (z in c2)]
    if not d: return 'agree (= oracle)', None
    v1_strict_ok = all(v1_mem(c1, z) == mem_fn(z, False) for z in pts)
    if v1_strict_ok: return 'v1 = strict oracle (no pole), v2 = pole convention', d[:3]
    fin_d = [z for z in d if not math.isinf(z)]
    tag = 'differ at finite points' if fin_d else 'differ only at +-inf'
    return f'v1 WRONG ({tag}), v2 = oracle', d[:3]

def main(seed, n):
    rng = random.Random(seed)
    stats = collections.Counter(); ex = collections.defaultdict(list)
    for i in range(n):
        pa, pb = rand_pieces(rng), rand_pieces(rng)
        mode = rng.choice(['ss', 'ss', 'sn', 'ns'])
        if mode == 'sn': pb = [(v, v, True, True) for v in [rng.choice(VALS)]]
        if mode == 'ns': pa = [(v, v, True, True) for v in [rng.choice(VALS)]]
        try:
            a1, b1 = mk1(pa), mk1(pb)
        except Exception as e:
            stats['v1 cannot build input'] += 1; ex['v1 cannot build input'].append((fmt(pa), fmt(pb), repr(e))); continue
        a2, b2 = mk2(pa), mk2(pb)
        A, B = pieces2(a2), pieces2(b2)
        allp = test_points(ends2(a2), ends2(b2))
        assert all(v1_mem(a1, z) == (z in a2) for z in allp) and all(v1_mem(b1, z) == (z in b2) for z in allp), (pa, pb)
        for name, f in OPS.items():
            l1 = pa[0][0] if mode == 'ns' else a1; r1 = pb[0][0] if mode == 'sn' else b1
            l2 = pa[0][0] if mode == 'ns' else a2; r2 = pb[0][0] if mode == 'sn' else b2
            c1, e1 = run_op(f, l1, r1); c2, e2 = run_op(f, l2, r2)
            if SAB and c2 is not None and i % 7 == 0 and name == '+':
                c2 = c2 | V2(F(1, 3))      # wrong on purpose
            cat, d = classify(name, c1, e1, c2, e2, A, B, ends2(a2) + ends2(b2),
                              lambda z, pole: member(name, A, B, z, pole))
            z0 = name == '/' and ((mode == 'sn' and pb[0][0] == 0) or (mode != 'sn' and F(0) in b2))
            key = f'{name} {mode} {cat}' + (' [divisor holds 0]' if z0 and 'agree' not in cat else '')
            stats[key] += 1
            ex[key].append(f'{fmt(pa)} {name} {fmt(pb)}: v1 {e1 or c1}  v2 {e2 or c2}  diff@ {[str(z) for z in d] if d else ""}')
        # reciprocal of A
        c1, e1 = run_op(lambda x, _: x.reciprocal(), a1, None); c2, e2 = run_op(lambda x, _: x.reciprocal(), a2, None)
        cat, d = classify('recip', c1, e1, c2, e2, A, None, ends2(a2), lambda z, pole: recip_member(A, z, pole))
        key = f'recip {cat}' + (' [A holds 0]' if F(0) in a2 and 'agree' not in cat else '')
        stats[key] += 1
        ex[key].append(f'1/({fmt(pa)}): v1 {e1 or c1}  v2 {e2 or c2}  diff@ {[str(z) for z in d] if d else ""}')
    for k in sorted(stats): print(f'{stats[k]:5d}  {k}')
    for k in sorted(ex):
        if 'agree' in k: continue
        print('==', k)
        for e in ex[k][:4]: print('   ', e)
    return stats

if __name__ == '__main__':
    st = main(int(sys.argv[1]) if len(sys.argv) > 1 else 1788, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    if SAB: assert any('v2 WRONG' in k for k in st), 'sabotage not caught'; print('SABOTAGE CAUGHT')
