"""what argument types v1's union/intersection/difference/symmetric_difference (and _update) accept, vs v2"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *

def run(label, f1, f2, expect_same=True):
    try: r1 = str(f1())
    except Exception as e: r1 = f'raise {type(e).__name__}: {str(e)[:40]}'
    try: r2 = str(f2())
    except Exception as e: r2 = f'raise {type(e).__name__}: {str(e)[:40]}'
    flag = '' if r1 == r2 else '   <-- DIFF'
    print(f'{label:52s} v1={r1:42.42s} v2={r2:42.42s}{flag}')

A1 = M1(0, 2); A2 = M2(0, 2)
args = [('{1, 3}', {1, 3}, None), ('[1, 3] list', [1, 3], M2(1, 3)), ('(1, 3) tuple', (1, 3), M2(1, 3, start_closed=False, end_closed=False)),
        ('"[1, 3)" str', '[1, 3)', M2.parse('[1, 3)')), ('"{5}" str', '{5}', M2.parse('{ [5] }')), ('True', True, M2(1)),
        ('nan', float('nan'), None), ('inf', inf, M2(inf)), ('-inf', -inf, M2(-inf)), ('Decimal', __import__('decimal').Decimal(1), None),
        ('np.float64(1.5)', __import__('numpy').float64(1.5), M2(1.5))]
for op in ['union', 'intersection', 'difference', 'symmetric_difference']:
    for name, a, v2_eq in args:
        run(f'[0,2].{op}({name})', lambda: getattr(A1, op)(a), lambda: getattr(A2, op)(a))
        if v2_eq is not None:
            run(f'   v2 explicit: [0,2].{op}(<{name} as MI>)', lambda: getattr(A1, op)(a), lambda: getattr(A2, op)(v2_eq))
print()
# the {1,3} set: v2 spelling union of points
run('[0,2].intersection({1,3}) vs v2 & (M(1)|M(3))', lambda: A1.intersection({1, 3}), lambda: A2 & (M2(1) | M2(3)))
run('[0,2].intersection({1,3}) vs v2 & from_pieces', lambda: A1.intersection({1, 3}), lambda: A2 & M2.from_pieces([(1, 1), (3, 3)]))
# empty self with one number
run('empty.union(3)', lambda: M1().union(3), lambda: M2().union(3))
run('empty.update(3) (add)', lambda: M1().update(3), lambda: M2() | 3)
run('[0,2].difference(1)', lambda: A1.difference(1), lambda: A2.difference(1))
run('[0,2].difference(1, 5)', lambda: A1.difference(1, 5), lambda: A2.difference(1, 5))
run('[0,2].difference()', lambda: A1.difference(), lambda: A2.difference())
run('[0,2].symmetric_difference()', lambda: A1.symmetric_difference(), lambda: A2.symmetric_difference())
run('[0,2].intersection()', lambda: A1.intersection(), lambda: A2.intersection())
run('[0,2].union()', lambda: A1.union(), lambda: A2.union())
run('[0,2] ^ [1,3] ^ [1.5,4] (3 operands)', lambda: A1.symmetric_difference(M1(1, 3), M1(1.5, 4)), lambda: A2.symmetric_difference(M2(1, 3), M2(1.5, 4)))
run('   chained 2-arg v1', lambda: A1.symmetric_difference(M1(1, 3)).symmetric_difference(M1(1.5, 4)), lambda: A2 ^ M2(1, 3) ^ M2(1.5, 4))
# aliasing in-place
x1 = M1(0, 1).union(M1(2, 3)); x2 = M2(0, 1) | M2(2, 3)
run('x.update(x)', lambda: x1.copy().update(x1), lambda: x2 | x2)
y = x1.copy(); run('x.difference_update(x) (self alias)', lambda: y.difference_update(y), lambda: x2.difference(x2))
y = x1.copy(); run('x.symmetric_difference_update(x)', lambda: y.symmetric_difference_update(y), lambda: x2 ^ x2)
y = x1.copy(); run('x.intersection_update(x)', lambda: y.intersection_update(y), lambda: x2 & x2)
# operators: v1 has none (| raises); v2 | & ^ ~ and their augmented rebinding forms
run('A | B operator', lambda: A1 | M1(5), lambda: A2 | M2(5))
z = A2; z |= 5; z &= M2(1, 9); z ^= M2(1.5, 1.5); z -= 0
print('v2 rebinding |= &= ^= ->', z, '| original untouched:', A2, '(note: -= is arithmetic subtraction, not difference)')
# sabotage
print('sabotage caught:', str(A1.union(M1(5))) != str(A2 & M2(5)))
