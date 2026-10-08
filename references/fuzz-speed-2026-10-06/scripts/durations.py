"""aggregate pytest --durations=0 output: per item (setup+call+teardown) and per test function.

usage: durations.py LOG [LOG...] [--top N] [--scale K]
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^\s*([\d.]+)s (setup|call|teardown)\s+(\S.*?)\s*$')


def load(paths):
    item = defaultdict(float)
    for p in paths:
        with open(p, encoding='utf-8', errors='replace') as f:
            for ln in f:
                m = LINE.match(ln)
                if m:
                    item[m.group(3)] += float(m.group(1))
    return item


def main(argv):
    top = 15
    paths = []
    it = iter(argv)
    for a in it:
        if a == '--top':
            top = int(next(it))
        else:
            paths.append(a)
    item = load(paths)
    func = defaultdict(float)
    nparams = defaultdict(int)
    for k, v in item.items():
        f = k.split('[', 1)[0]
        func[f] += v
        nparams[f] += 1
    total = sum(item.values())
    print(f'items with a duration line: {len(item)}, functions: {len(func)}, total {total:.1f} s')
    for label, d in (('item', item), ('function', func)):
        ranked = sorted(d.values(), reverse=True)
        for n in (1, 10, 50):
            print(f'  top {n:>2} {label}s: {sum(ranked[:n]):8.1f} s = {100 * sum(ranked[:n]) / total:5.1f}%')
    print(f'\ntop {top} items:')
    for k, v in sorted(item.items(), key=lambda kv: -kv[1])[:top]:
        print(f'  {v:8.2f}  {k}')
    print(f'\ntop {top} functions (sum over params):')
    for k, v in sorted(func.items(), key=lambda kv: -kv[1])[:top]:
        print(f'  {v:8.2f}  {k}  ({nparams[k]} items)')


if __name__ == '__main__':
    main(sys.argv[1:])
