from common import *
D = dt.datetime; d_ = dt.date
UTC = dt.timezone.utc
r = random.Random(1)


def show(x):
    return x[1] if x[0] == 'raise' else str(x[1])


def cmp(label, f1, f2, expect_same=True, extra=()):
    a = safe(f1); b = safe(f2)
    if a[0] != 'ok' or b[0] != 'ok':
        print(f'{label}: v1 {a[0]} {show(a)} | v2 {b[0]} {show(b)}')
        return None
    ok, diff = same_set(a[1], b[1], extra)
    print(f'{label}: same_set={ok} v1={a[1]} v2={b[1]}' + (f' diff={diff}' if diff else ''))
    if expect_same is not None:
        check(ok == expect_same, label)
    return ok


# SELFTEST: a 10:00:00.5 point vs a 10:00:00.6 point must differ
selftest(lambda: same_set(V1D(D(2024, 1, 1, 10, 0, 0, 500000)), V2D(D(2024, 1, 1, 10, 0, 0, 600000)))[0])

print('--- empty and points')
cmp('empty', lambda: V1D(), lambda: V2D())
cmp('point dt', lambda: V1D(D(2024, 1, 1, 10, 30)), lambda: V2D(D(2024, 1, 1, 10, 30)))
cmp('point midnight dt', lambda: V1D(D(2024, 1, 1)), lambda: V2D(D(2024, 1, 1)))
cmp('point date = day (us grid)', lambda: V1D(d_(2024, 1, 1)), lambda: V2D(d_(2024, 1, 1)),
    extra=[D(2024, 1, 1, 23, 59, 59, 999999), D(2024, 1, 2)])
print('--- explicit None end')
cmp('dt, None', lambda: V1D(D(2024, 1, 1, 10, 30), None), lambda: V2D(D(2024, 1, 1, 10, 30), None))
cmp('date, None', lambda: V1D(d_(2024, 1, 1), None), lambda: V2D(d_(2024, 1, 1), None))
print('--- two datetimes, us != 0 (no snap)')
for sc in (True, False):
    for ec in (True, False):
        cmp(f'dt,dt flags {sc},{ec}',
            lambda: V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 17, 0, 0, 7), start_closed=sc, end_closed=ec),
            lambda: V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 17, 0, 0, 7), start_closed=sc, end_closed=ec))
print('--- snaps of a datetime end (v1)')
for e in (D(2024, 1, 2), D(2024, 1, 1, 17), D(2024, 1, 1, 17, 30), D(2024, 1, 1, 17, 30, 15)):
    cmp(f'snap end {e}', lambda: V1D(D(2024, 1, 1, 9, 0, 0, 1), e), lambda: V2D(D(2024, 1, 1, 9, 0, 0, 1), e),
        expect_same=False)
cmp('snap makes reversed valid: (10:30, 10:00)', lambda: V1D(D(2024, 1, 1, 10, 30), D(2024, 1, 1, 10)),
    lambda: V2D(D(2024, 1, 1, 10, 30), D(2024, 1, 1, 10)), expect_same=None)
print('--- two dates')
cmp('date,date', lambda: V1D(d_(2024, 1, 1), d_(2024, 1, 7)), lambda: V2D(d_(2024, 1, 1), d_(2024, 1, 7)),
    extra=[D(2024, 1, 7, 23, 59, 59, 999999), D(2024, 1, 8)])
for sc in (True, False):
    for ec in (True, False):
        cmp(f'date,date flags {sc},{ec}',
            lambda: V1D(d_(2024, 1, 1), d_(2024, 1, 7), start_closed=sc, end_closed=ec),
            lambda: V2D(d_(2024, 1, 1), d_(2024, 1, 7), start_closed=sc, end_closed=ec), expect_same=None,
            extra=[D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 7), D(2024, 1, 7, 23, 59, 59, 999999), D(2024, 1, 8)])
print('--- single date with flags')
for sc, ec in ((False, True), (True, False), (False, False)):
    cmp(f'date single flags {sc},{ec}', lambda: V1D(d_(2024, 1, 1), start_closed=sc, end_closed=ec),
        lambda: V2D(d_(2024, 1, 1), start_closed=sc, end_closed=ec), expect_same=None)
cmp('workaround day minus midnight', lambda: V1D(d_(2024, 1, 1), start_closed=False),
    lambda: V2D(d_(2024, 1, 1)).difference(V2D(D(2024, 1, 1))),
    extra=[D(2024, 1, 1), D(2024, 1, 1, 0, 0, 0, 1), D(2024, 1, 1, 23, 59, 59, 999999), D(2024, 1, 2)])
cmp('workaround day minus end us', lambda: V1D(d_(2024, 1, 1), end_closed=False),
    lambda: V2D(D(2024, 1, 1), D(2024, 1, 1, 23, 59, 59, 999999), end_closed=False),
    extra=[D(2024, 1, 1), D(2024, 1, 1, 23, 59, 59, 999999)])
cmp('workaround open day both ends', lambda: V1D(d_(2024, 1, 1), start_closed=False, end_closed=False),
    lambda: V2D(D(2024, 1, 1), D(2024, 1, 1, 23, 59, 59, 999999), start_closed=False, end_closed=False),
    extra=[D(2024, 1, 1), D(2024, 1, 1, 23, 59, 59, 999999)])
print('--- mixed date / datetime')
cmp('date, dt noon same day', lambda: V1D(d_(2024, 1, 2), D(2024, 1, 2, 12, 0, 0, 5)),
    lambda: V2D(d_(2024, 1, 2), D(2024, 1, 2, 12, 0, 0, 5)))
cmp('dt noon, date same day', lambda: V1D(D(2024, 1, 2, 12, 0, 0, 5), d_(2024, 1, 2)),
    lambda: V2D(D(2024, 1, 2, 12, 0, 0, 5), d_(2024, 1, 2)), extra=[D(2024, 1, 2, 23, 59, 59, 999999), D(2024, 1, 3)])
cmp('dt, date (later)', lambda: V1D(D(2024, 1, 2, 12, 0, 0, 5), d_(2024, 1, 5)),
    lambda: V2D(D(2024, 1, 2, 12, 0, 0, 5), d_(2024, 1, 5)), extra=[D(2024, 1, 5, 23, 59, 59, 999999), D(2024, 1, 6)])
print('--- reversed')
cmp('wed, mon', lambda: V1D(d_(2024, 1, 3), d_(2024, 1, 1)), lambda: V2D(d_(2024, 1, 3), d_(2024, 1, 1)), expect_same=None)
cmp('tue, mon', lambda: V1D(d_(2024, 1, 2), d_(2024, 1, 1)), lambda: V2D(d_(2024, 1, 2), d_(2024, 1, 1)), expect_same=None)
cmp('dt reversed', lambda: V1D(D(2024, 1, 2, 0, 0, 0, 5), D(2024, 1, 1, 0, 0, 0, 5)),
    lambda: V2D(D(2024, 1, 2, 0, 0, 0, 5), D(2024, 1, 1, 0, 0, 0, 5)), expect_same=None)
cmp('dt start after date end', lambda: V1D(D(2024, 1, 2, 12), d_(2024, 1, 1)),
    lambda: V2D(D(2024, 1, 2, 12), d_(2024, 1, 1)), expect_same=None)
print('--- equal ends with an open flag')
t = D(2024, 1, 1, 10, 0, 0, 5)
for sc, ec in ((False, True), (True, False), (False, False)):
    cmp(f't,t flags {sc},{ec}', lambda: V1D(t, t, start_closed=sc, end_closed=ec),
        lambda: V2D(t, t, start_closed=sc, end_closed=ec), expect_same=None)
print('--- None / NaT / nan')
cmp('None, t', lambda: V1D(None, t), lambda: V2D(None, t), expect_same=None)
cmp('NaT, t', lambda: V1D(pd.NaT, t), lambda: V2D(pd.NaT, t), expect_same=None)
cmp('t, NaT', lambda: V1D(t, pd.NaT), lambda: V2D(t, pd.NaT), expect_same=None)
cmp('date, NaT', lambda: V1D(d_(2024, 1, 1), pd.NaT), lambda: V2D(d_(2024, 1, 1), pd.NaT), expect_same=None)
cmp('nan, t', lambda: V1D(float('nan'), t), lambda: V2D(float('nan'), t), expect_same=None)
cmp('t, nan', lambda: V1D(t, float('nan')), lambda: V2D(t, float('nan')), expect_same=None)
cmp('NaT', lambda: V1D(pd.NaT), lambda: V2D(pd.NaT), expect_same=None)
cmp('np.datetime64', lambda: V1D(pd.Timestamp(t).to_datetime64()), lambda: V2D(pd.Timestamp(t).to_datetime64()), expect_same=None)
print('--- foreign types')
for x in ('2024-01-01', '01/01/2024 10:00', 1704067200, 1704067200.0, Fraction(1), dt.time(10), dt.timedelta(1)):
    print(f'type {type(x).__name__} {x!r}: v1', show(safe(lambda: V1D(x))), '| v2', show(safe(lambda: V2D(x))))
print('--- infinities')
for x in (float('inf'), -float('inf'), NEG_INF, POS_INF, dt.datetime.max, dt.datetime.min):
    print(f'start {x!r}: v1', show(safe(lambda: V1D(x, t))), '| v2', show(safe(lambda: V2D(x, t))))
    print(f'end   {x!r}: v1', show(safe(lambda: V1D(t, x))), '| v2', show(safe(lambda: V2D(t, x))))
print('--- pandas Timestamp')
ts = pd.Timestamp('2024-01-01 10:00:00.000005')
cmp('Timestamp point', lambda: V1D(ts), lambda: V2D(ts))
cmp('Timestamp,Timestamp', lambda: V1D(ts, pd.Timestamp('2024-01-02 10:00:00.000007')),
    lambda: V2D(ts, pd.Timestamp('2024-01-02 10:00:00.000007')))
cmp('Timestamp end at midnight (snap)', lambda: V1D(ts, pd.Timestamp('2024-01-02')),
    lambda: V2D(ts, pd.Timestamp('2024-01-02')), expect_same=False)
tsn = pd.Timestamp('2024-01-01 10:00:00.000005001')
a = safe(lambda: V1D(tsn)); b = safe(lambda: V2D(tsn))
print('ns Timestamp point: v1', show(a), '| v2', show(b))
if a[0] == 'ok' and b[0] == 'ok':
    print('  v1 seconds', a[1].interval.endpoints, ' v2 seconds', b[1].seconds, ' v2 holds ns ts:', tsn in b[1],
          ' v1 holds ns ts:', safe(lambda: tsn in a[1]), ' v1 holds the us-truncated:', D(2024, 1, 1, 10, 0, 0, 5) in a[1])
print('--- aware datetimes')
sgt = dt.timezone(dt.timedelta(hours=8))
ta = D(2024, 1, 1, 2, 0, 0, 5, tzinfo=UTC)
a = safe(lambda: V1D(ta, ta + dt.timedelta(hours=3))); b = safe(lambda: V2D(ta, ta + dt.timedelta(hours=3)))
print('aware utc: v1', show(a), '| v2', show(b))
print('  v1 inf', repr(a[1].infimum), ' v2 inf', repr(b[1].inf))
print('  aware 03:00Z in v1:', (ta + dt.timedelta(hours=1)) in a[1], ' in v2:', (ta + dt.timedelta(hours=1)) in b[1])
print('  naive local 11:00 (= 03:00Z) in v1:', D(2024, 1, 1, 11) in a[1], ' in v2:', show(safe(lambda: D(2024, 1, 1, 11) in b[1])))
print('mixed naive/aware: v1', show(safe(lambda: V1D(D(2024, 1, 1, 9), ta))), '| v2', show(safe(lambda: V2D(D(2024, 1, 1, 9), ta))))
print('mixed zones: v1', show(safe(lambda: V1D(ta, D(2024, 1, 1, 12, tzinfo=sgt)))), '| v2',
      show(safe(lambda: V2D(ta, D(2024, 1, 1, 12, tzinfo=sgt)))))
tsa = pd.Timestamp('2024-01-01 02:00:00.000005', tz='UTC')
a = safe(lambda: V1D(tsa)); b = safe(lambda: V2D(tsa))
print('aware Timestamp: v1', show(a), a[1].interval.endpoints if a[0] == 'ok' else '', '| v2', show(b))
print('--- before 1970 / far range')
for x in (d_(1960, 1, 1), D(1969, 12, 31, 12, 0, 0, 5), d_(1, 1, 1), d_(9999, 12, 31), D(2100, 1, 1, 0, 0, 0, 5)):
    a = safe(lambda: V1D(x)); b = safe(lambda: V2D(x))
    print(f'{x!r}: v1 {show(a)} | v2 {show(b)}')
print('--- random sweep: dt,dt and date,date and mixed (us grid)')
n = 0
for i in range(400):
    kind = r.choice(['dtdt', 'dd', 'ddt', 'dtd', 'pt', 'day'])
    a = rand_dt(r); b = rand_dt(r)
    if a > b:
        a, b = b, a
    args = {'dtdt': (a, b), 'dd': (a.date(), b.date()), 'ddt': (a.date(), b), 'dtd': (a, b.date()),
            'pt': (a,), 'day': (a.date(),)}[kind]
    sc = r.random() < .5 if kind == 'dtdt' else True
    ec = r.random() < .5 if kind == 'dtdt' else True
    x1 = safe(lambda: V1D(*args, start_closed=sc, end_closed=ec))
    x2 = safe(lambda: V2D(*args, start_closed=sc, end_closed=ec))
    if x1[0] != x2[0]:
        check(False, f'raise mismatch {args} {x1} {x2}')
        continue
    if x1[0] == 'raise':
        continue
    extra = [D.combine(x, dt.time()) + dt.timedelta(days=1) for x in args if not isinstance(x, D)]
    extra += [e - US for e in extra]
    ok, diff = same_set(x1[1], x2[1], extra=extra)
    check(ok, f'{kind} {args} {sc} {ec} {diff}')
    n += 1
print('sweep compared', n)
report('constructor')
