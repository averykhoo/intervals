"""float ends: x / A (number / set), A / x (set / number) and A.reciprocal(), v1 (default flag) vs v2 MultiInterval,
EXACT structure compared (end values bit for bit, flags). where they differ, each side's end is checked against the
correctly rounded exact quotient of a corner (Fraction arithmetic, rounded once by float(Fraction)).
usage: float_div_sweep.py SEED N [sabotage]"""
import sys, os, math, random, collections
sys.path.insert(0, os.path.dirname(__file__))
from oracle import *
warnings.simplefilter('ignore')
SAB = len(sys.argv) > 3

def struct1(a):
    e = a.endpoints
    return [(float(e[i][0]), e[i][1] == 0, float(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]
def struct2(b):
    return [(float(p.inf), bool(p.inf_closed), float(p.sup), bool(p.sup_closed)) for p in b.pieces]

def rfloat(rng):
    k = rng.random()
    if k < .3: return round(rng.uniform(-5, 5), 1)
    if k < .5: return rng.uniform(-1e3, 1e3)
    if k < .6: return rng.choice([0.0, 1.0, -1.0, 0.5, 3.0, 0.1])
    if k < .65: return rng.uniform(-1, 1) * 1e-300
    if k < .7: return rng.uniform(-1, 1) * 1e300
    return rng.uniform(-10, 10)

def rpieces_nozero(rng):
    """float pieces all on one side of 0 (so 0 is in no piece): the divisor of x / A and the operand of reciprocal()"""
    sign = rng.choice([1, -1]); out = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, b = sorted((abs(rfloat(rng)) + rng.choice([0.0, 0.25]), abs(rfloat(rng)) + rng.choice([0.0, 0.25])))
        if a == 0: a = 0.5
        if b == 0: b = 0.75
        a, b = sorted((a * sign, b * sign))
        if a == b or rng.random() < .1: out.append((a, a, True, True)); continue
        out.append((a, b, rng.random() < .5, rng.random() < .5))
    return out

def rpieces_any(rng):
    out = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, b = sorted((rfloat(rng), rfloat(rng)))
        if a == b or rng.random() < .1: out.append((a, a, True, True)); continue
        out.append((a, b, rng.random() < .5, rng.random() < .5))
    return out

def judge(r1, r2, corners):
    """for each differing end value: which side equals a correctly rounded exact corner"""
    t = collections.Counter()
    if len(r1) != len(r2): t['piece count differs'] += 1; return t
    for p1, p2 in zip(r1, r2):
        for k in (0, 2):
            if p1[k] != p2[k] or p1[k + 1] != p2[k + 1]:
                if p1[k] == p2[k]: t['flag only'] += 1; continue
                in1, in2 = p1[k] in corners, p2[k] in corners
                t['v2 correctly rounded, v1 not' if in2 and not in1 else 'v1 correctly rounded, v2 not' if in1 and not in2 else 'neither/both'] += 1
                ulps = abs(p1[k] - p2[k]) / math.ulp(p2[k]) if p2[k] not in (0.0,) and math.isfinite(p2[k]) else None
                if ulps is not None and ulps != 1.0: print('   ulp note', p1[k], p2[k], ulps, file=sys.stderr)
                t[f'ulps apart {ulps}'] += 1
    return t

def main(seed, n):
    rng = random.Random(seed)
    st = collections.Counter(); ulp = collections.Counter(); ex = collections.defaultdict(list)
    for i in range(n):
        x = rng.choice([rfloat(rng), rfloat(rng), rng.randint(-9, 9)])
        A = rpieces_nozero(rng); G = rpieces_any(rng)
        a1, a2, g1, g2 = mk1(A), mk2(A), mk1(G), mk2(G)
        ends_A = [F(v) for p in A for v in p[:2]]; ends_G = [F(v) for p in G for v in p[:2]]
        def rnd(q):
            try: return float(q)
            except OverflowError: return INF if q > 0 else -INF
        cases = {
            'x / A': (lambda: x / a1, lambda: x / a2, {rnd(F(x) / e) for e in ends_A}),
            'reciprocal(A)': (lambda: a1.reciprocal(), lambda: a2.reciprocal(), {rnd(1 / e) for e in ends_A}),
            'G / x': (lambda: g1 / x, lambda: g2 / x, {rnd(e / F(x)) for e in ends_G} if x != 0 else set()),
            'G / A (critic covered A op B; repeated as a control)': (lambda: g1 / a1, lambda: g2 / a2, {rnd(g / e) for g in ends_G for e in ends_A}),
        }
        for name, (f1, f2, corners) in cases.items():
            try: c1 = f1(); c1._consistency_check(); s1 = struct1(c1); e1 = None
            except Exception as e: s1, e1 = None, f'{type(e).__name__}: {str(e)[:60]}'
            try: c2 = f2(); s2 = struct2(c2); e2 = None
            except Exception as e: s2, e2 = None, f'{type(e).__name__}: {str(e)[:60]}'
            if SAB and s2 and i % 50 == 1 and name == 'reciprocal(A)':
                s2 = s2[:-1] + [(s2[-1][0], s2[-1][1], math.nextafter(s2[-1][2], INF), s2[-1][3])]
            key = f'{name}: '
            if e1 or e2:
                key += f'v1 raises={bool(e1)} v2 raises={bool(e2)}'; st[key] += 1
                ex[key].append(f'x={x!r} A={fmt(A)} G={fmt(G)}: v1 {e1 or s1} v2 {e2 or c2}'); continue
            if s1 == s2: st[key + 'same structure'] += 1; continue
            if name == 'G / x' and x == 0:
                st[key + 'divisor 0: v1 (-inf, inf), v2 empty'] += 1; continue
            j = judge(s1, s2, corners)
            for k, v in j.items():
                if k.startswith('ulps'): ulp[f'{name} {k}'] += v
                else: st[key + k] += v
            if not any(k.startswith('ulps') or k == 'v2 correctly rounded, v1 not' for k in j) or 'v1 correctly rounded, v2 not' in j or 'neither/both' in j or 'flag only' in j or 'piece count differs' in j:
                ex[key + 'OTHER'].append(f'x={x!r} A={fmt(A)} G={fmt(G)}: v1 {s1} v2 {s2}')
            ex[key + 'differs'].append(f'x={x!r} A={fmt(A)} G={fmt(G)}: v1 {s1} v2 {s2}')
            st[key + 'cases differing'] += 1
    for k in sorted(st): print(f'{st[k]:6d}  {k}')
    for k in sorted(ulp): print(f'{ulp[k]:6d}  {k}')
    for k in sorted(ex):
        print('==', k)
        for e in ex[k][:4]: print('   ', e[:420])
    return st

if __name__ == '__main__':
    st = main(int(sys.argv[1]), int(sys.argv[2]))
    if SAB: assert any('v1 correctly rounded, v2 not' in k for k in st), 'sabotage missed'; print('SABOTAGE CAUGHT')
