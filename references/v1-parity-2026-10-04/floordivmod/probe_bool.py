"""v1 takes a bool as the Real it subclasses; v2 refuses bool. workaround int(b): v1 A % True / A // True vs v2 A % 1 / A // 1"""
from common import *
import common
_ex = common.ex
common.ex = ex = lambda v: Fraction(int(v)) if isinstance(v, bool) else _ex(v)
rng = random.Random(3)
eq = neq = 0; v2raise = 0; ex1 = None; kinds = {}
for _ in range(200):
    A = rand_pieces(rng, lo=0, hi=12)
    for op in ('%', '//'):
        r1 = mk1(A) % True if op == '%' else mk1(A) // True
        try:
            (mk2(A) % True) if op == '%' else (mk2(A) // True)
        except TypeError:
            v2raise += 1
        r2 = mk2(A) % int(True) if op == '%' else mk2(A) // int(True)
        p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
        if op == '//':  # v1's // is a hull of floors; compare the integers it holds against v2's
            p1 = canon([(Fraction(n), True, Fraction(n), True) for n in range(-1, 14) if contains(p1, Fraction(n))])
        if p1 == p2: eq += 1
        else:
            neq += 1; ex1 = ex1 or (op, show(A), show(p1), show(p2)); kinds[op] = kinds.get(op, 0) + 1
print(f'v2 raised TypeError for a bool: {v2raise}/400; v1(bool) vs v2(int(bool)) sets equal {eq}, differ {neq}', ex1 or '')
print('v1 [1,2] % True endpoints:', (mk1([(1, True, 2, True)]) % True).endpoints)
print('differences by op:', kinds)
