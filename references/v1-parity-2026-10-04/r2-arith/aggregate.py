import collections, re, sys
tot = collections.Counter()
for fn in sys.argv[1:]:
    for line in open(fn, encoding='utf-8'):
        if line.startswith('=='): break
        m = re.match(r'^\s*(\d+)  (.*)$', line.rstrip())
        if m: tot[m.group(2)] += int(m.group(1))
for k in sorted(tot): print(f'{tot[k]:5d}  {k}')
print('total', sum(tot.values()))
