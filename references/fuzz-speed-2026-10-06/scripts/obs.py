"""count test cases and distinct inputs per property from hypothesis observability JSONL.

usage: obs.py LABEL DIR [DIR...]   each DIR is a HYPOTHESIS_STORAGE_DIRECTORY (reads DIR/observed/*.jsonl)
prints, per property: runs (distinct run_start), cases by how_generated, distinct representations
(all and generated-only), and the overlap: generated cases whose input another run also generated.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv):
    label, dirs = argv[0], argv[1:]
    cases = defaultdict(list)  # property -> [(run_key, how, hash)]
    for d in dirs:
        for p in sorted(Path(d, 'observed').glob('*.jsonl')):
            with open(p, encoding='utf-8') as f:
                for ln in f:
                    try:
                        o = json.loads(ln)
                    except ValueError:
                        continue
                    if o.get('type') != 'test_case':
                        continue
                    h = hashlib.sha1(o['representation'].encode('utf-8')).hexdigest()
                    cases[o['property']].append(((d, o['run_start']), o['how_generated'], o['status'], h))
    for prop, rows in sorted(cases.items()):
        runs = {r[0] for r in rows}
        how = Counter(r[1] for r in rows)
        status = Counter(r[2] for r in rows)
        gen = [r for r in rows if r[1].startswith('during generate')]
        distinct_all = len({r[3] for r in rows})
        distinct_gen = len({r[3] for r in gen})
        valid_gen = [r for r in gen if r[2] == 'passed']
        distinct_valid = len({r[3] for r in valid_gen})
        # inputs generated in more than one run
        per_run = defaultdict(set)
        for r in gen:
            per_run[r[0]].add(r[3])
        seen = Counter(h for s in per_run.values() for h in s)
        shared = sum(1 for c in seen.values() if c > 1)
        print(f'[{label}] {prop}: runs={len(runs)} cases={len(rows)} status={dict(status)}')
        print(f'    how={dict(how)}')
        print(f'    generated={len(gen)} distinct_generated={distinct_gen} passed_generated={len(valid_gen)} '
              f'distinct_passed_generated={distinct_valid} distinct_all={distinct_all} '
              f'inputs_in_more_than_one_run={shared}')


if __name__ == '__main__':
    main(sys.argv[1:])
