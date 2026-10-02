"""
CORE-MATH's binary64 worst cases against `intervals.elementary`, a manual tool (never run by CI)

CORE-MATH (https://gitlab.inria.fr/core-math/core-math, MIT) keeps, per function, a file of inputs that
are hard to round (`src/binary64/<f>/<f>.wc`: hex floats, `x,y` for two operands, `#` lines heading
blocks). the files hold inputs only; the expected values here come from MPFR (gmpy2, 53 bits, the
binary64 exponent range, subnormals), which is CORE-MATH's own reference. what the cases catch, and what
they cannot, is `references/test-vector-sources.md` §3h.

    $PY tools/coremath.py fetch              # download the pinned files into the cache, check sha256
    $PY tools/coremath.py sample [--check]   # (re)write tests/coremath/*.tsv, or check they are current
    $PY tools/coremath.py check [--jobs N] [--only exp,log]   # every input, three directions
    $PY tools/coremath.py status             # what changed since the last full check
    $PY tools/coremath.py pin [<commit>]     # move the pin (master by default): rewrites the manifest

the cache is `.scratch/coremath-cache/<commit>/` (gitignored, kept between sessions: `CLAUDE.md`). the
pin and every file's sha256 are `tests/coremath/MANIFEST.tsv`; a file whose bytes differ is refused.
the gate runs only the vendored sample (`tests/test_coremath.py`); `check` runs everything and appends
its verdict to `references/coremath-runs.tsv`, which `status` reads.
"""
import argparse
import ast
import hashlib
import math
import multiprocessing
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'tests' / 'coremath'
MANIFEST = DATA / 'MANIFEST.tsv'
RUNS = ROOT / 'references' / 'coremath-runs.tsv'  # prose to the ledger: a check's row stales no gate run
CACHE = ROOT / '.scratch' / 'coremath-cache'
REPO = 'https://gitlab.inria.fr/core-math/core-math'
API = 'https://gitlab.inria.fr/api/v4/projects/35719'

# our functions with a CORE-MATH binary64 file. the three of two operands go through the scalar calls
# the set layer uses: atan2 through `rounded_angle`, hypot through sqrt of the exact x*x + y*y, pow
# through `rounded_pow`
UNARY = ('exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'log1p', 'sin', 'cos', 'tan', 'asin', 'acos',
         'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'cbrt')
BINARY = ('atan2', 'hypot', 'pow')
FUNCTIONS = UNARY + BINARY

# the sample: every row of a block up to WHOLE rows, PER_BLOCK seeded rows of a bigger one, and these
# blocks whole whatever their size (atan's ±2^e block is the one that drives the loop past 512 bits)
WHOLE = 300
PER_BLOCK = 200
WHOLE_BLOCKS = {'atan': ('the following are +/-2^e',)}


# ---- the files ---------------------------------------------------------------------------------------

def read_manifest():
    """(commit, {name: (sha256, symmetric)})"""
    commit, files = None, {}
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        if line.startswith('# commit '):
            commit = line.split()[2]
        elif line and not line.startswith('#'):
            name, digest, symmetric = line.split('\t')
            files[name] = (digest, symmetric == 'symmetric')
    return commit, files


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _raw(commit: str, path: str) -> bytes:
    return _get(f'{REPO}/-/raw/{commit}/{path}')


def cached(name: str) -> Path:
    """the pinned file of `name`, from the cache, its sha256 checked (fetched if absent)"""
    commit, files = read_manifest()
    path = CACHE / commit / f'{name}.wc'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _raw(commit, f'src/binary64/{name}/{name}.wc')
        path.with_suffix('.part').write_bytes(data)
        path.with_suffix('.part').replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != files[name][0]:
        raise SystemExit(f'{path}: sha256 {digest} is not the manifest\'s {files[name][0]}; delete it to refetch')
    return path


def blocks(name: str):
    """[(label, [x or (x, y)]), ...] in file order: finite inputs, each block headed by its `#` lines"""
    pair = name in BINARY
    out, rows, label, heading = [], [], '(no heading)', []
    for line in cached(name).read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if text.startswith('#'):
            body = text[1:].strip()
            if body and _parse(body.split('#')[0], pair) is None:  # not an empty comment nor a commented-out input
                heading.append(body)
            continue
        value = _parse(text.split('#')[0], pair)
        if value is None:
            continue
        if heading:  # the first row after `#` lines starts a block
            if rows:
                out.append((label, rows))
            rows, label, heading = [], ' / '.join(heading), []
        rows.append(value)
    if rows:
        out.append((label, rows))
    return out


def _parse(text: str, pair: bool):
    """the operand of a data line: its first hex float (a count of bits may follow it), or for a function
    of two operands its first two, comma-separated; None for anything else or a non-finite value"""
    if pair:
        parts = [p.strip() for p in text.split(',')[:2]]
        if len(parts) < 2:
            return None
    else:
        parts = text.split()[:1]
    try:
        values = tuple(float.fromhex(p) for p in parts if p)
    except (ValueError, OverflowError):
        return None
    if len(values) != (2 if pair else 1) or not all(math.isfinite(v) for v in values):
        return None
    return values if pair else values[0]


# ---- the oracle ----------------------------------------------------------------------------------------

def _oracle():
    import gmpy2
    from intervals.rounding import DOWN, NEAREST, UP
    modes = {DOWN: gmpy2.RoundDown, NEAREST: gmpy2.RoundToNearest, UP: gmpy2.RoundUp}
    contexts = {d: gmpy2.context(precision=53, emin=-1073, emax=1024, subnormalize=True, round=r)
                for d, r in modes.items()}
    unary = {name: getattr(gmpy2, name) for name in UNARY}
    binary = {'atan2': lambda x, y: gmpy2.atan2(y, x), 'hypot': gmpy2.hypot, 'pow': lambda x, y: x ** y}

    def value(name, operand, direction):
        with contexts[direction]:
            if name in BINARY:
                x, y = operand
                return float(binary[name](gmpy2.mpfr(x), gmpy2.mpfr(y)))
            return float(unary[name](gmpy2.mpfr(operand)))
    return value


def inside(name: str, operand) -> bool:
    """whether the scalar call takes this operand: the set layer's cases (a domain's outside, a zero
    of atan2, a base <= 0 of pow) are not the scalar's"""
    from intervals import elementary
    if name == 'atan2':
        return operand[0] != 0 and operand[1] != 0
    if name == 'hypot':
        return True
    if name == 'pow':
        return operand[0] > 0
    try:
        elementary._check_domain(name, Fraction(operand), None)
    except ValueError:
        return False
    return True


def ours(name: str, operand, direction: int) -> float:
    """the library's scalar answer: the call the set layer makes for an end"""
    from intervals import elementary
    if name == 'atan2':
        x, y = operand
        q = Fraction(y) / Fraction(x)
        return elementary.rounded_angle(q, 0 if x > 0 else 2 if y > 0 else -2, direction)
    if name == 'hypot':
        x, y = operand
        return elementary.rounded('sqrt', Fraction(x) ** 2 + Fraction(y) ** 2, direction)
    if name == 'pow':
        return elementary.rounded_pow(Fraction(operand[0]), Fraction(operand[1]), direction)
    return elementary.rounded(name, Fraction(operand), direction)


def inputs(name: str):
    """[(block index, label, operands)] of the scalar's inputs, mirrored where CORE-MATH mirrors them"""
    symmetric = read_manifest()[1][name][1] and name in UNARY  # hypot's sign is the square's
    out = []
    for i, (label, rows) in enumerate(blocks(name)):
        if symmetric:
            rows = rows + [-x for x in rows]
        seen, kept = set(), []
        for row in rows:
            if row not in seen and inside(name, row):
                seen.add(row)
                kept.append(row)
        out.append((i, label, kept))
    return out


# ---- the sample ------------------------------------------------------------------------------------------

def _hex(operand) -> str:
    return ','.join(v.hex() for v in operand) if isinstance(operand, tuple) else operand.hex()


def sample_text(name: str) -> str:
    from intervals.rounding import DOWN, NEAREST, UP
    value = _oracle()
    commit, files = read_manifest()
    lines = [f'# CORE-MATH {commit} src/binary64/{name}/{name}.wc (sha256 {files[name][0]}), MIT: LICENSE.core-math',
             f'# written by tools/coremath.py sample: every row of a block of <= {WHOLE}, {PER_BLOCK} seeded rows of a '
             f'bigger one{", sign mirrored" if files[name][1] else ""}; expected values from MPFR',
             '# do not edit: tools/coremath.py sample --check']
    rows = []
    for i, label, kept in inputs(name):
        whole = len(kept) <= WHOLE or any(w in label for w in WHOLE_BLOCKS.get(name, ()))
        chosen = kept if whole else random.Random(f'{name}:{i}').sample(kept, PER_BLOCK)
        if not chosen:
            continue
        lines.append(f'# block {i} ({len(chosen)} of {len(kept)}): {label[:150]}')
        rows += [(operand, i) for operand in chosen]
    lines.append('# x[,y]\tdown\tnearest\tup\tblock')
    for operand, i in rows:
        down, nearest, up = (value(name, operand, d) for d in (DOWN, NEAREST, UP))
        lines.append(f'{_hex(operand)}\t{down.hex()}\t{nearest.hex()}\t{up.hex()}\t{i}')
    return '\n'.join(lines) + '\n'


def sample(check: bool) -> int:
    stale = []
    for name in FUNCTIONS:
        text, path = sample_text(name), DATA / f'{name}.tsv'
        if check:
            if not path.exists() or path.read_text(encoding='utf-8') != text:
                stale.append(name)
        else:
            path.write_text(text, encoding='utf-8', newline='\n')
            print(f'{path.relative_to(ROOT)}: {text.count(chr(10)) - text.count(chr(10) + "#") - 1} rows')
    if stale:
        print(f'the sample is not what the pinned files and the oracle give: {", ".join(stale)}')
    return 1 if stale else 0


# ---- the full check ------------------------------------------------------------------------------------------

def _check_chunk(job):
    name, operands = job
    from intervals import backend
    from intervals.rounding import DOWN, NEAREST, UP
    value, bad, slow = _oracle(), [], []
    with backend._use('python'):
        for operand in operands:
            for direction in (DOWN, NEAREST, UP):
                start = time.perf_counter()
                try:
                    got = ours(name, operand, direction)
                except Exception as exc:  # noqa: BLE001 - every failure is a finding
                    bad.append(f'{name} {_hex(operand)} {direction}: raises {exc!r}')
                    continue
                spent = time.perf_counter() - start
                if spent > 1:
                    slow.append(f'{name} {_hex(operand)} {direction}: {spent:.1f} s')
                want = value(name, operand, direction)
                if got != want:
                    bad.append(f'{name} {_hex(operand)} {direction}: ours {got.hex()}, MPFR {want.hex()}')
    return name, len(operands), bad, slow


def check(jobs: int, only) -> int:
    names = only or FUNCTIONS
    started, t0 = datetime.now(), time.perf_counter()
    work = []
    for name in names:
        operands = [o for _, _, kept in inputs(name) for o in kept]
        work += [(name, operands[k:k + 2000]) for k in range(0, len(operands), 2000)]
    totals, bad, slow = {}, [], []
    with multiprocessing.Pool(jobs) as pool:
        for k, (name, n, b, s) in enumerate(pool.imap_unordered(_check_chunk, work), 1):
            totals[name] = totals.get(name, 0) + n
            bad += b
            slow += s
            if k % 50 == 0 or k == len(work):
                print(f'  {k} of {len(work)} chunks, {sum(totals.values())} inputs, {len(bad)} mismatches, '
                      f'{time.perf_counter() - t0:.0f} s', flush=True)
    seconds = time.perf_counter() - t0
    for name in names:
        print(f'{name}: {totals.get(name, 0)} inputs x3')
    for line in bad[:50] + slow[:20]:
        print('  ' + line)
    inputs_n = sum(totals.values())
    print(f'{inputs_n} inputs, {3 * inputs_n} calls, {len(bad)} mismatches, {len(slow)} over 1 s, {seconds:.0f} s')
    head = _git('rev-parse', '--short', 'HEAD')
    dirty = 'dirty' if _git('status', '--porcelain', '--', *closure()) else 'clean'
    commit = read_manifest()[0]
    with RUNS.open('a', encoding='utf-8', newline='\n') as out:
        out.write(f'{started:%Y-%m-%d %H:%M}\t{head}\t{dirty}\t{commit[:12]}\t{",".join(names) if only else "all"}\t'
                  f'{inputs_n}\t{len(bad)}\t{len(slow)}\t{seconds:.0f}\t{jobs}\n')
    return 1 if bad else 0


# ---- status ----------------------------------------------------------------------------------------------------

def closure(start: str = 'intervals/elementary.py'):
    """the files the scalar evaluator imports, transitively, read from the source (importing
    `intervals` would pull in the whole package through its __init__)"""
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path in seen or not (ROOT / path).exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse((ROOT / path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] == 'intervals':
                parts = node.module.split('.')[1:]
                if parts:
                    todo.append('intervals/' + '/'.join(parts) + '.py')
                else:
                    todo += [f'intervals/{alias.name}.py' for alias in node.names]
            elif isinstance(node, ast.Import):
                todo += ['intervals/' + '/'.join(a.name.split('.')[1:]) + '.py' for a in node.names
                         if a.name.startswith('intervals.')]
    return sorted(seen)


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def status() -> int:
    commit, _ = read_manifest()
    print(f'pinned: CORE-MATH {commit[:12]}')
    try:
        line = subprocess.run(['git', 'ls-remote', f'{REPO}.git', 'refs/heads/master'], capture_output=True,
                              text=True, timeout=60).stdout.split()
        upstream = line[0] if line else None
    except (OSError, subprocess.TimeoutExpired):
        upstream = None
    if upstream is None:
        print('upstream: unknown (offline)')
    elif upstream == commit:
        print('upstream: master is the pin')
    else:
        try:
            import json
            diff = json.loads(_get(f'{API}/repository/compare?from={commit}&to={upstream}'))
            touched = sorted({d['new_path'] for d in diff['diffs'] for name in FUNCTIONS
                              if d['new_path'] == f'src/binary64/{name}/{name}.wc'})
            print(f'upstream: master {upstream[:12]}, {len(diff["commits"])} commits past the pin; '
                  f'our .wc files changed: {", ".join(touched) or "none"}')
        except Exception as exc:  # noqa: BLE001
            print(f'upstream: master {upstream[:12]} differs from the pin ({exc!r} comparing them)')
    files = closure()
    runs = [line.split('\t') for line in RUNS.read_text(encoding='utf-8').splitlines()
            if line and not line.startswith('#')] if RUNS.exists() else []
    full = [r for r in runs if r[4] == 'all']
    if not full:
        print('last full check: never')
        return 1
    last = full[-1]
    print(f'last full check: {last[0]} at {last[1]} ({last[2]}), CORE-MATH {last[3]}: {last[5]} inputs, '
          f'{last[6]} mismatches, {last[8]} s')
    log = _git('log', '--oneline', f'{last[1]}..HEAD', '--', *files)
    stat = _git('diff', '--shortstat', last[1], '--', *files)
    print(f'the scalar evaluator ({", ".join(Path(f).stem for f in files)}): '
          f'{len(log.splitlines())} commits since, {stat or "no change"}')
    return 0


# ---- pin ------------------------------------------------------------------------------------------------------

def pin(commit: str) -> int:
    import json
    if commit == 'master':
        commit = json.loads(_get(f'{API}/repository/commits/master'))['id']
    lines = [f'# commit {commit}', f'# {REPO}, src/binary64/<name>/<name>.wc, sha256; symmetric: CORE-MATH defines '
             'WORST_SYMMETRIC for it, so its worst cases are checked at -x too', '# name\tsha256\tsymmetric']
    for name in FUNCTIONS:
        data = _raw(commit, f'src/binary64/{name}/{name}.wc')
        path = CACHE / commit / f'{name}.wc'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        header = _raw(commit, f'src/binary64/{name}/function_under_test.h').decode('utf-8')
        symmetric = any(line.strip().startswith('#define WORST_SYMMETRIC') for line in header.splitlines())
        lines.append(f'{name}\t{hashlib.sha256(data).hexdigest()}\t{"symmetric" if symmetric else "-"}')
        print(f'{name}: {len(data)} bytes{", symmetric" if symmetric else ""}', flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('fetch')
    s = sub.add_parser('sample')
    s.add_argument('--check', action='store_true')
    c = sub.add_parser('check')
    c.add_argument('--jobs', type=int, default=1)
    c.add_argument('--only', type=lambda t: t.split(','), default=None)
    sub.add_parser('status')
    p = sub.add_parser('pin')
    p.add_argument('commit', nargs='?', default='master')
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        for name in FUNCTIONS:
            print(f'{name}: {cached(name).stat().st_size} bytes, sha256 ok', flush=True)
        return 0
    if args.command == 'sample':
        return sample(args.check)
    if args.command == 'check':
        return check(args.jobs, args.only)
    if args.command == 'status':
        return status()
    return pin(args.commit)


if __name__ == '__main__':
    sys.exit(main())
