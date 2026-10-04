import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime as dt, random
import multi_interval as v1m
import time_interval as v1t
import intervals as v2
from intervals.time_interval import TimeDeltaInterval as T2, DateTimeInterval as D2
T1 = v1t.TimeDeltaInterval; D1 = v1t.DateTimeInterval
td = lambda s: dt.timedelta(seconds=s)
print("v1 TD has __bool__:", '__bool__' in vars(T1), "__len__:", '__len__' in vars(T1))
print("v1 DT has __bool__:", '__bool__' in vars(D1), "__len__:", '__len__' in vars(D1))
print("v1 MI __bool__ on empty:", bool(v1m.MultiInterval()), "on [1,2]:", bool(v1m.MultiInterval(1,2)))
cases = {}
# empty constructions on v1
e1 = T1(); cases['T1()'] = e1
e1b = T1(td(1), td(2)); e1b = e1b.intersection(T1(td(3), td(4))); cases['T1 [1,2]&[3,4]'] = e1b
e1c = T1(td(1), td(2), start_closed=False).intersection(T1(td(0), td(1)))
cases['T1 (1,1]'] = e1c
for k, v in cases.items():
    print(k, "is_empty", v.is_empty, "bool", bool(v), "| v1 inner MI bool", bool(v.interval))
# v2 side
e2 = T2(); print("T2() is_empty", e2.is_empty, "bool", bool(e2))
e2b = T2(td(1), td(2)) & T2(td(3), td(4)); print("T2 [1,2]&[3,4] is_empty", e2b.is_empty, "bool", bool(e2b))
print("D1() bool", bool(D1()), "is_empty", D1().is_empty, "| D2() bool", bool(D2()), "is_empty", D2().is_empty)
# sweep: bool vs not is_empty, both sides
random.seed(65); mism_v1 = 0; mism_v2 = 0; agree_nonempty = 0; n = 0; diffs = 0
for _ in range(400):
    a, b, c, d = sorted(random.sample(range(-5, 6), 2)) + sorted(random.sample(range(-5, 6), 2))
    sc, ec = random.random() < .5, random.random() < .5
    x1 = T1(td(a), td(b), start_closed=sc, end_closed=ec).intersection(T1(td(c), td(d)))
    x2 = T2(td(a), td(b), start_closed=sc, end_closed=ec) & T2(td(c), td(d))
    n += 1
    assert x1.is_empty == x2.is_empty, (a, b, c, d, sc, ec)
    if bool(x1) != (not x1.is_empty): mism_v1 += 1
    if bool(x2) != (not x2.is_empty): mism_v2 += 1
    if bool(x1) != bool(x2): diffs += 1
print(f"sweep n={n}: v1 bool!=nonempty {mism_v1}, v2 bool!=nonempty {mism_v2}, v1/v2 bool differ {diffs}")
# the v1-reproducing spelling in v2: always True -> no spelling needed; truthiness as emptiness: not A.is_empty / bool(A)
# sanity: a deliberately wrong expectation must be caught
try:
    assert bool(e1) == bool(e2), "caught: v1 bool(empty) != v2 bool(empty)"
    print("SANITY FAIL: probe could not fail")
except AssertionError as ex:
    print("sanity ok:", ex)
