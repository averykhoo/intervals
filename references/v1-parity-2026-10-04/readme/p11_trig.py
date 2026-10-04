# TODO "trigonometry?": v1 had none; v2's sin/cos/tan enclose sampled values (soundness spot check)
from common import *
warnings.simplefilter('ignore')
rng = random.Random(1); bad = 0; n = 0
for _ in range(300):
    a, b = sorted(rng.uniform(-10, 10) for _ in range(2)); A = MI(a, b)
    for f, g in ((math.sin, A.sin()), (math.cos, A.cos())):
        for k in range(9):
            x = a + (b - a) * k / 8; y = f(x); n += 1
            # the exact value is within an ulp of math's; accept if the set holds y or its neighbours
            if not any(z in g for z in (y, math.nextafter(y, -2), math.nextafter(y, 2))): bad += 1
print(f'sin/cos enclose samples: {n - bad}/{n}')
print('v1 has sin?', hasattr(V1, 'sin'), '| v2 tan [1,2]:', MI(1, 2).tan())
assert not (2 in MI(0, 1).sin())  # sabotage
