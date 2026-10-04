import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
from common import *
a = mk1([(1, 2, True, False), (3, 3, True, True)])
print(a, pieces1(a))
b = mk2([(1, 2, True, False), (3, 3, True, True)])
print(b, pieces2(b))
print(compare(pieces1(a), pieces2(b)))
# sabotage: a wrong expectation must be caught
print('sabotage', compare(pieces1(a), [(1, 2, True, True), (3, 3, True, True)]))
