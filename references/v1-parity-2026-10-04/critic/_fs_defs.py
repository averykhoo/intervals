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

