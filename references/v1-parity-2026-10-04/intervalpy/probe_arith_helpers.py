from common import *

def run(f):
    try:
        return 'ok', f()
    except Exception as e:
        return type(e).__name__, str(e)[:70]

def exact_val(rng):
    r = rng.random()
    if r < 0.5:
        return rng.randint(-5, 5)
    return F(rng.randint(-20, 20), rng.randint(1, 5))

def exact_interval(rng):
    while True:
        a, b = sorted([exact_val(rng), exact_val(rng)])
        so, ec = rng.random() < 0.5, rng.random() < 0.5
        if rng.random() < 0.12:
            a, so = -INF, True
        if rng.random() < 0.12:
            b, ec = INF, False
        if a == b:
            so, ec = False, True
        try:
            return I(a, so, b, ec)
        except ValueError:
            pass
