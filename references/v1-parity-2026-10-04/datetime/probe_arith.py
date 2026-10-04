from common import *
from gen import rand_pair
D = dt.datetime; d_ = dt.date; TD = dt.timedelta
r = random.Random(41)
UTC = dt.timezone.utc


def td_pts(t2):
    out = []
    for p in t2:
        for v in (p.inf, p.sup):
            out.append(v)
    return out


def td_cands(*pts):
    base = sorted(set(pts))
    out = set()
    for p in base:
        for d in (TD(0), US, -US, TD(seconds=1), -TD(seconds=1)):
            out.add(p + d)
    for a, b in zip(base, base[1:]):
        out.add(a + (b - a) // 2)
    return sorted(out)


def same_td(t1, t2, extra=()):
    pts = td_cands(*td_pts(t2), *[TD(seconds=float(x)) for x, _ in t1.interval.endpoints], *extra)
    m1 = [p in t1 for p in pts]; m2 = [p in t2 for p in pts]
    if m1 != m2:
        return False, [(p, x, y) for p, x, y in zip(pts, m1, m2) if x != y][:3]
    return True, None


def rand_td_pair(r, n_max=2):
    t1, t2 = V1T(), V2T()
    for _ in range(r.randrange(1, n_max + 1)):
        a = TD(seconds=r.randrange(-86400, 86400), microseconds=r.randrange(1, 10 ** 6))
        if r.random() < .3:
            t1 = t1.union(V1T(a)); t2 = t2.union(V2T(a))
        else:
            b = a + TD(seconds=r.randrange(1, 86400), microseconds=r.randrange(1, 10 ** 6))
            sc, ec = r.random() < .5, r.random() < .5
            t1 = t1.union(V1T(a, b, start_closed=sc, end_closed=ec)); t2 = t2.union(V2T(a, b, start_closed=sc, end_closed=ec))
    return t1, t2


def exact_sum(A2, T2, sign=1):
    """oracle: union over piece pairs of [a.inf + s*b, a.sup + s*b'] by datetime arithmetic, flags closed iff both closed"""
    out = V2D()
    for a in A2:
        for b in T2:
            if sign == 1:
                lo, hi, lc, hc = a.inf + b.inf, a.sup + b.sup, a.inf_closed and b.inf_closed, a.sup_closed and b.sup_closed
            else:
                lo, hi, lc, hc = a.inf - b.sup, a.sup - b.inf, a.inf_closed and b.sup_closed, a.sup_closed and b.inf_closed
            out = out | V2D(lo, hi, start_closed=lc, end_closed=hc)
    return out


def exact_diff(A2, B2):
    out = V2T()
    for a in A2:
        for b in B2:
            out = out | V2T(a.inf - b.sup, a.sup - b.inf, start_closed=a.inf_closed and b.sup_closed, end_closed=a.sup_closed and b.inf_closed)
    return out


# SELFTEST: shifting by 1 h vs 2 h must be caught
A1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1)); A2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1))
selftest(lambda: same_set(A1 + TD(hours=1), A2 + TD(hours=2))[0])

print('--- hand cases')
h = TD(hours=1)
for label, f1, f2 in [
    ('A + td', lambda: A1 + h, lambda: A2 + h), ('td + A', lambda: h + A1, lambda: h + A2),
    ('A + pd.Timedelta', lambda: A1 + pd.Timedelta(hours=1), lambda: A2 + pd.Timedelta(hours=1)),
    ('pd.Timedelta + A', lambda: pd.Timedelta(hours=1) + A1, lambda: pd.Timedelta(hours=1) + A2),
    ('A + TDI', lambda: A1 + V1T(h, 2 * h), lambda: A2 + V2T(h, 2 * h)), ('TDI + A', lambda: V1T(h, 2 * h) + A1, lambda: V2T(h, 2 * h) + A2),
    ('A - td', lambda: A1 - h, lambda: A2 - h), ('A - TDI', lambda: A1 - V1T(h, 2 * h), lambda: A2 - V2T(h, 2 * h)),
    ('A - pd.Timedelta', lambda: A1 - pd.Timedelta(hours=1), lambda: A2 - pd.Timedelta(hours=1)),
]:
    x1, x2 = safe(f1), safe(f2)
    if x1[0] == 'ok' and x2[0] == 'ok':
        ok = same_set(x1[1], x2[1])[0]; check(ok, label)
        print(f'{label}: same={ok} v1={x1[1]} ({type(x1[1]).__name__}) v2={x2[1]} ({type(x2[1]).__name__})')
    else:
        print(f'{label}: v1 {x1} | v2 {x2}')
for label, f1, f2 in [
    ('A - dt', lambda: A1 - D(2024, 1, 1), lambda: A2 - D(2024, 1, 1)),
    ('A - date', lambda: A1 - d_(2024, 1, 1), lambda: A2 - d_(2024, 1, 1)),
    ('A - Timestamp', lambda: A1 - pd.Timestamp('2024-01-01'), lambda: A2 - pd.Timestamp('2024-01-01')),
    ('A - DTI', lambda: A1 - V1D(D(2024, 1, 1), D(2024, 1, 1, 1, 0, 0, 1)), lambda: A2 - V2D(D(2024, 1, 1), D(2024, 1, 1, 1, 0, 0, 1))),
    ('dt - A', lambda: D(2024, 1, 2) - A1, lambda: D(2024, 1, 2) - A2),
    ('date - A', lambda: d_(2024, 1, 2) - A1, lambda: d_(2024, 1, 2) - A2),
    ('Timestamp - A', lambda: pd.Timestamp('2024-01-02') - A1, lambda: pd.Timestamp('2024-01-02') - A2),
]:
    x1, x2 = safe(f1), safe(f2)
    if x1[0] == 'ok' and x2[0] == 'ok':
        ok = same_td(x1[1], x2[1])[0]; check(ok or 'date' in label, label)
        print(f'{label}: same={ok} v1={x1[1]} ({type(x1[1]).__name__}) v2={x2[1]} ({type(x2[1]).__name__})')
    else:
        print(f'{label}: v1 {x1} | v2 {x2}')
print('--- refusals')
for label, f1, f2 in [('A + dt', lambda: A1 + D(2024, 1, 1), lambda: A2 + D(2024, 1, 1)),
                      ('A + DTI', lambda: A1 + A1, lambda: A2 + A2),
                      ('A + 5', lambda: A1 + 5, lambda: A2 + 5), ('5 + A', lambda: 5 + A1, lambda: 5 + A2),
                      ('A - 5', lambda: A1 - 5, lambda: A2 - 5), ('A * 2', lambda: A1 * 2, lambda: A2 * 2),
                      ('A + NaT', lambda: A1 + pd.NaT, lambda: A2 + pd.NaT), ('A - NaT', lambda: A1 - pd.NaT, lambda: A2 - pd.NaT),
                      ('A + np.timedelta64', lambda: A1 + pd.Timedelta(hours=1).to_timedelta64(), lambda: A2 + pd.Timedelta(hours=1).to_timedelta64())]:
    print(f'{label}: v1 {safe(f1)} | v2 {safe(f2)}')
print('--- empty operands')
for label, f1, f2 in [('A + empty TDI', lambda: A1 + V1T(), lambda: A2 + V2T()), ('empty + td', lambda: V1D() + h, lambda: V2D() + h),
                      ('A - empty DTI', lambda: A1 - V1D(), lambda: A2 - V2D()), ('empty - dt', lambda: V1D() - D(2024, 1, 1), lambda: V2D() - D(2024, 1, 1)),
                      ('dt - empty', lambda: D(2024, 1, 1) - V1D(), lambda: D(2024, 1, 1) - V2D()), ('TDI + empty DTI', lambda: V1T(h) + V1D(), lambda: V2T(h) + V2D())]:
    print(f'{label}: v1 {safe(f1)} | v2 {safe(f2)}')
print('--- aware')
B1 = V1D(D(2024, 1, 1, 1, 0, 0, 1, tzinfo=UTC), D(2024, 1, 1, 2, 0, 0, 1, tzinfo=UTC)); B2 = V2D(D(2024, 1, 1, 1, 0, 0, 1, tzinfo=UTC), D(2024, 1, 1, 2, 0, 0, 1, tzinfo=UTC))
print('aware - aware dt: v1', B1 - D(2024, 1, 1, tzinfo=UTC), ' v2', B2 - D(2024, 1, 1, tzinfo=UTC))
print('aware + td: v1', B1 + h, ' v2', B2 + h)
print('aware - naive: v1', safe(lambda: B1 - D(2024, 1, 1)), ' v2', safe(lambda: B2 - D(2024, 1, 1)))

print('--- random sweep')
st = {}
for i in range(400):
    a1, a2, _ = rand_pair(r, max_pieces=3, days=3)
    t1, t2 = rand_td_pair(r)
    if a1.is_empty:
        continue
    for label, f1, f2, oracle in [
        ('A + TDI', lambda: a1 + t1, lambda: a2 + t2, lambda: exact_sum(a2, t2)),
        ('TDI + A', lambda: t1 + a1, lambda: t2 + a2, lambda: exact_sum(a2, t2)),
        ('A - TDI', lambda: a1 - t1, lambda: a2 - t2, lambda: exact_sum(a2, t2, -1)),
    ]:
        x1, x2 = safe(f1), safe(f2)
        s = st.setdefault(label, [0, 0, 0, 0])
        s[0] += 1
        if x1[0] != 'ok' or x2[0] != 'ok':
            check(False, f'{label} raised {x1} {x2}'); continue
        s[1] += same_set(x1[1], x2[1])[0]
        s[2] += x2[1] == oracle()
        s[3] += same_set(x1[1], oracle())[0]
        check(x2[1] == oracle(), f'{label} v2 vs oracle {a2} {t2}')
    b1, b2, _ = rand_pair(r, max_pieces=2, days=3)
    if not b1.is_empty:
        x1, x2 = a1 - b1, a2 - b2
        s = st.setdefault('A - B', [0, 0, 0, 0]); s[0] += 1
        s[1] += same_td(x1, x2)[0]; s[2] += x2 == exact_diff(a2, b2); s[3] += same_td(x1, exact_diff(a2, b2))[0]
        check(x2 == exact_diff(a2, b2), 'A - B oracle')
        p = candidates(*v2_points(b2))[r.randrange(5)]
        y1, y2 = p - a1, p - a2
        s = st.setdefault('dt - A', [0, 0, 0, 0]); s[0] += 1
        s[1] += same_td(y1, y2)[0]; s[2] += y2 == exact_diff(V2D(p), a2); s[3] += same_td(y1, exact_diff(V2D(p), a2))[0]
        check(y2 == exact_diff(V2D(p), a2), 'dt - A oracle')
print('[cases, v1==v2 at us grid, v2==exact oracle, v1==exact oracle at us grid]')
for k, v in st.items():
    print('  ', k, v)
report('arith')
