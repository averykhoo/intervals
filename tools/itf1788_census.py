"""
the itf1788 counts quoted in HANDOFF.md, v2-plan.md "ieee 1788" and the plan's M13 records: files,
vectors, interval-valued vectors, ops, divergence keys and vectors by category, vectors per file,
skipped statements by op. read from the adapter itself, so it cannot drift from what the gate runs.
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
cats = Counter(r.split(':')[0] for r in t.DIVERGENCES.values())
print('rows (keys) by category', dict(cats), 'total', len(t.DIVERGENCES), 'listed', len(t.LISTED))
hit = Counter(t.DIVERGENCES[t.key(v)].split(':')[0] for v in t.VECTORS if t.key(v) in t.DIVERGENCES)
print('vectors under a row by category', dict(hit))
hito = Counter(t.DIVERGENCES[t.key(v)].split(':')[0] for v in t.INTERVAL_VECTORS if t.key(v) in t.DIVERGENCES)
print('interval vectors under a row by category', dict(hito))
per_file = Counter(v.source.split(':')[0] for v in t.VECTORS)
print('vectors per file', dict(per_file))
sk = Counter()
for name, c in t.SKIPPED.items():
    sk.update(c)
print('skipped statements', sum(sk.values()), 'ops', len(sk))
print('skipped by op', dict(sk.most_common()))
