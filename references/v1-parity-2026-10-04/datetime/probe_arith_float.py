"""why v1's `A - dt` misses its own ends: float timestamps"""
from common import *
D = dt.datetime; TD = dt.timedelta
A1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1)); A2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1))
x1 = A1 - D(2024, 1, 1); x2 = A2 - D(2024, 1, 1)
print('v1 endpoints', x1.interval.endpoints, ' v2 seconds', x2.seconds)
for p in (TD(hours=9, microseconds=1), TD(hours=10, microseconds=1)):
    print(p, 'in v1', p in x1, 'in v2', p in x2, ' total_seconds()', repr(p.total_seconds()))
    check((p in x2), 'v2 holds its exact end')
print('exact error of v1 start:', Fraction(x1.interval.endpoints[0][0]) - Fraction(32400000001, 10 ** 6))
print('exact error of v1 end:  ', Fraction(x1.interval.endpoints[1][0]) - Fraction(36000000001, 10 ** 6))
# SELFTEST: an expectation that v1 holds both ends must fail for at least one end
selftest(lambda: all(p in x1 for p in (TD(hours=9, microseconds=1), TD(hours=10, microseconds=1))))
report('arith_float')
