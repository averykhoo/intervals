import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random
rng = random.Random(1)
for _ in range(500):
    p = rand_pieces(rng)
    a, b = twin(p)
    d = same_set_v1_v2(a, b)
    assert d is None, (p, d)
# sabotage: a wrong expectation is caught
a, b = twin([(0, 1, True, False)])
assert same_set_v1_v2(a, v2_from([(0, 1, True, True)])) is not None
assert same_set_v1_v2(a, v2_from([(0, 1, True, False), (1,1,True,True)])) is not None
print('common ok')
