# doctest-like usage in v1's docstrings/comments: the string forms of MultiInterval.merge (multi_interval.py:302-305)
# "e.g. [1, 2] or [1,2] / [0] or {0} / {} or [] or () or (123) / { [1, 2) | [3, 4) } or {[1,2),[3,4)} or even [1,2)[3,4)"
from common import *
def run(f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always'); r = f()
        return r, ([type(x.message).__name__ for x in w])
    except Exception as e: return f'{type(e).__name__}: {str(e)[:60]}', []
forms = ['[1, 2]', '[1,2]', '[0]', '{0}', '{}', '[]', '()', '(123)', '{ [1, 2) | [3, 4) }', '{[1,2),[3,4)}', '[1,2)[3,4)',
         '[1, inf)', '(-inf, 0]', '[1e-3, 2.5]', '{1, 2, 3}', '[- 5, 5]', '{1; 2}', '[1; 2]', '(1)', '[1, 2', 'hello [1, 2] world', '[2, 1]']
agree = 0
for s in forms:
    r1, w1 = run(lambda: V1.merge(s)); r2, w2 = run(lambda: MI.parse(s))
    if isinstance(r1, V1): r1c = conv(r1); same = isinstance(r2, MI) and r1c == r2; r1 = str(r1)
    else: same = False
    agree += same
    print(f'{s!r:24s} v1: {str(r1)+(" "+str(w1) if w1 else ""):42s} v2: {str(r2)+(" "+str(w2) if w2 else ""):30s} {"SAME" if same else "DIFF"}')
print('agree', agree, 'of', len(forms))
# sabotage: a wrong expectation is caught
assert conv(V1.merge('[1,2)[3,4)')) != MI.parse('[1, 4)')
