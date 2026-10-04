"""structural pattern matching: dataclass __match_args__ on v1 Interval vs v2 attribute patterns"""
from common import *  # noqa

print('v1 __match_args__:', V1.__match_args__, ' v2:', getattr(MI, '__match_args__', None))


def v1_match(iv):
    match iv:
        case V1(s, so, e, ec):
            return (s, so, e, ec)


def v2_match(A):
    match A:
        case MI(is_contiguous=True, inf=s, inf_closed=sc, sup=e, sup_closed=ec):
            return (s, not sc, e, ec)


def v2_positional(A):
    match A:
        case MI(s, so, e, ec):
            return 'matched'


n = 0
for iv in [V1(0, False, 1, True), V1(-math.inf, True, Fraction(1, 3), False), V1(2, False, 2, True), V1(0.5, True, math.inf, False)]:
    r1, r2 = v1_match(iv), v2_match(v1_to_v2(iv))
    check('match', r1 == r2, (iv, r1, r2)); n += 1
print('keyword-attribute pattern agrees on', n, 'cases;', 'v2 two-piece set does not match:', v2_match(MI(0, 1) | MI(2, 3)))
try:
    v2_positional(MI(0, 1))
except TypeError as ex:
    print('v2 positional pattern: TypeError', ex)
check('SABOTAGE', v2_match(MI(0, 1)) == (0, True, 1, True))
print('sabotage caught', len(MISMATCHES), 'of 1'); del MISMATCHES[:1]
report('probe_match_args')
