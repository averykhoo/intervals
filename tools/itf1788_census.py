"""
the itf1788 counts quoted in HANDOFF.md, v2-plan.md "ieee 1788" and the plan's M13 records: files,
vectors, interval-valued vectors, ops, divergence keys and vectors by category, vectors per file,
skipped statements by op, and (M16b) the 1788 layer pass's rows. read from the adapter and the pass
themselves, so they cannot drift from what the gate runs.
not collected by pytest (`tools/` is not in `testpaths`). from the repo root:

    C:/Users/user/anaconda3/envs/intervals/python.exe tools/itf1788_census.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'archive' / 'v1')]

from tests.itf1788 import test_itf1788 as t  # noqa: E402


print('files', len(t.FILES))
print('vectors', len(t.VECTORS), 'interval-valued', len(t.INTERVAL_VECTORS), 'test items', len(t.VECTORS) + len(t.INTERVAL_VECTORS))
print('ops in OPS', len(t.OPS), 'ops with vectors', len({v.op for v in t.VECTORS}))


def category(reason):
    """the REASONS entry a row's reason starts with (a category may hold a colon: 'no NaI: ...')"""
    return next(c for c in t.REASONS if reason.startswith(c))


cats = Counter(category(r) for r in t.DIVERGENCES.values())
print('rows (keys) by category', dict(cats), 'total', len(t.DIVERGENCES), 'listed', len(t.LISTED))
hit = Counter(category(t.DIVERGENCES[t.key(v)]) for v in t.VECTORS if t.key(v) in t.DIVERGENCES)
print('vectors under a row by category', dict(hit))
hito = Counter(category(t.DIVERGENCES[t.key(v)]) for v in t.INTERVAL_VECTORS if t.key(v) in t.DIVERGENCES)
print('interval vectors under a row by category', dict(hito))
# M13g part 3: the decorated vectors, and the rows on a decoration alone (plain pass only)
print('plain-only rows by category', dict(Counter(category(r) for r in t.PLAIN_ONLY.values())))
dec = [v for v in t.VECTORS if t.is_decorated(v)]
print('decorated vectors', len(dec), 'propagated', sum(v.op in t.PROPAGATED for v in dec),
      'bare part', sum(v.op in t.BARE_PART for v in dec), 'decorated ops', sum(v.op in t.DECORATED for v in dec),
      'propagated ops with one', len({v.op for v in dec if v.op in t.PROPAGATED}))
# M13's merge: the rows on a decoration alone in both passes (their set must match), and the reverse
# ops' decorated vectors, run on DecoratedInterval operands (those with a [nai] operand are rows)
print('decoration-only rows by category', dict(Counter(category(r) for r in t.DECORATION_ONLY.values())))
rev = [v for v in dec if v.op in t.REVERSE]
print('decorated reverse vectors', len(rev), 'with a [nai] operand', sum(t._has_nai(v) for v in rev),
      'pairs', sum(v.op in t.PAIRS for v in rev))
per_file = Counter(v.source.split(':')[0] for v in t.VECTORS)
print('vectors per file', dict(per_file))
sk = Counter()
for name, c in t.SKIPPED.items():
    sk.update(c)
print('skipped statements', sum(sk.values()), 'ops', len(sk))
print('skipped by op', dict(sk.most_common()))
# M16b: the 1788 layer's pass (tests/itf1788/test_ieee1788.py), exact, with its own rows: the
# adapter's under three categories only
from tests.itf1788 import test_ieee1788 as layer  # noqa: E402

print('layer pass rows (keys) by category', dict(Counter(category(r) for r in layer.ROWS.values())),
      'total', len(layer.ROWS), 'vectors under a row', sum(1 for v in t.VECTORS if t.key(v) in layer.ROWS))
