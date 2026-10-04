# every line of v1 README's "redo illustrations" subtree must appear verbatim in the carried-over todo file
v1 = open('archive/v1/README.md', encoding='utf-8').read().splitlines()
todo = open('references/todo-from-v1-readme.md', encoding='utf-8').read().splitlines()
start = next(i for i, l in enumerate(v1) if l.startswith('* redo illustrations'))
block = [v1[start]]
for l in v1[start + 1:]:
    if not l.startswith('  '):
        break
    block.append(l)
todo_set = set(todo)
for n, l in enumerate(block, start + 1):
    print(f'{"KEPT" if l in todo_set else "LOST"} v1:{n}: {l}')
# control: a line that surely is in the todo file, and one surely not
assert '* redo illustrations with negative and positive bits' in todo_set
assert '  * zoom into x axis a bit' not in todo_set
