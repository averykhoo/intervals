from common import *
rng = random.Random(3)
v1right = v2right = diff = 0
for _ in range(2000):
    x = sorted([rng.uniform(-10, 10) for _ in range(2)])
    a = I(x[0], False, x[1], True); A = to_v2(a)
    fr = F(rng.randint(1, 10**6), 3)
    r1, R = a + fr, A + fr
    for v1e, v2e, xe in [(r1.start, R.inf, x[0]), (r1.end, R.sup, x[1])]:
        if v1e != v2e:
            diff += 1
            correct = float(F(xe) + fr)  # Fraction.__float__ rounds the exact sum once, to nearest
            v1right += v1e == correct; v2right += v2e == correct
print('ends differing', diff, 'v1 correctly rounded', v1right, 'v2 correctly rounded', v2right)
check('SELFTEST expected mismatch', v1right == diff)
check('v2 always correctly rounded', v2right == diff)
report_end(__file__)
