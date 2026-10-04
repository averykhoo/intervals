"""re-check arith row 'float rounding of + - * / (to nearest) [EQUAL]': its evidence is hand cases only (the random
sweeps used Fraction ends). here: random FLOAT ends, exact structure compared (end values bit-for-bit and closedness)."""
import sys, math, random, warnings
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/arith']
warnings.simplefilter('ignore')
from common import V1, V2, mk1, mk2
SABOTAGE = len(sys.argv) > 1

def struct1(a):
    e = a.endpoints
    return [(float(e[i][0]), e[i][1] == 0, float(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]

def struct2(b):
    return [(float(p.inf), bool(p.inf_closed), float(p.sup), bool(p.sup_closed)) for p in b]

def rfloat(rng):
    k = rng.random()
    if k < .3: return round(rng.uniform(-5, 5), 1)          # 0.1-grid: inexact binary
    if k < .5: return rng.uniform(-1e3, 1e3)
    if k < .6: return rng.choice([0.0, 1.0, -1.0, 0.5])
    if k < .7: return rng.uniform(-1, 1) * 1e-300
    return rng.uniform(-10, 10)

def rpieces(rng, positive_only=False):
    out = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, b = sorted((rfloat(rng), rfloat(rng)))
        if positive_only:
            a, b = abs(a) + 0.25, abs(b) + 0.25
            a, b = min(a, b), max(a, b)
        if a == b or rng.random() < .1:
            out.append((a, a, True, True)); continue
        out.append((a, b, rng.random() < .5, rng.random() < .5))
    return out

rng = random.Random(4242)
ops = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}
agree = {k: 0 for k in ops}; differ = {k: [] for k in ops}; raised = {k: 0 for k in ops}
for i in range(700):
    pa = rpieces(rng)
    for name, f in ops.items():
        pb = rpieces(rng, positive_only=(name == '/'))
        try:
            a1, b1, a2, b2 = mk1(pa), mk1(pb), mk2(pa), mk2(pb)
            r1 = struct1(f(a1, b1)); r2 = struct2(f(a2, b2))
        except Exception as e:
            raised[name] += 1; continue
        if SABOTAGE and i == 5:
            r2 = r2[:-1] + [(r2[-1][0], r2[-1][1], math.nextafter(r2[-1][2], math.inf), r2[-1][3])]
        if r1 == r2: agree[name] += 1
        else: differ[name].append((pa, pb, r1, r2))
print('agree', agree); print('raised', raised)
for k, v in differ.items():
    print(k, 'differ', len(v))
    for d in v[:3]: print('   ', d)
assert not any(differ.values()), 'differences found'
