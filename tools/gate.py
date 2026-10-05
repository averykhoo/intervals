"""the run ledger: which gate and fuzz runs passed, and on what code.

    $PY tools/gate.py run gate:itf          # tests/itf1788, the gate's first call
    $PY tools/gate.py run gate:rest         # the rest, its second call
    $PY tools/gate.py run docs              # the collected READMEs' doctests only
    $PY tools/gate.py run fuzz-x10:itf      # the fuzz profile at x10, as fuzz.yml (prepush runs these)
    $PY tools/gate.py run fuzz-x10:rest
    $PY tools/gate.py run gate:gmpy2        # the whole suite with MULTIINTERVAL_BACKEND=gmpy2 forced, as ci.yml's job
    $PY tools/gate.py status                # what is green on the code in front of you
    $PY tools/gate.py status --require commit   # exit 1 unless the gate is green on this code
    $PY tools/gate.py status --require push     # exit 1 unless a push needs nothing more
    $PY tools/gate.py plan                  # nothing, or docs / fuzz / gmpy2 joined by + (tools/prepush.sh reads it)
    $PY tools/gate.py tree-id

why: "did the gate run on this code, and did the fuzz run?" was answered from memory and from logs
in `.scratch/`, which name a commit, not the code that ran. adapted from the sibling repo
graph-reachability-zanzibar-index (`scripts/gate_status.py` and `formal/verify.sh`'s ledger,
2026-08-16..09-10), whose lessons are kept here.

each `run` appends one row to `.gate-runs/ledger.tsv` and keeps the run's whole output beside it
(both gitignored). a row is about the CODE, not the commit: its ids are content addresses over the
files the run could read, so a run made before `git commit` still counts after it, and a commit
that changes no content changes nothing. the recorder and the reader are this one file, so they
cannot disagree about what "the same code" means.

two scopes, each an id over the tracked and untracked-not-ignored files' bytes:

* `src` (`s:`): everything but markdown and `references/`, exactly the paths fuzz.yml's
  `paths-ignore` skips (tests/test_gate_ledger.py pins the match). a fuzz verdict is keyed by it.
* `code` (`c:`): `src` plus every `README.md`, which pytest collects as doctests
  (`--doctest-glob=README.md`). a gate or docs verdict is keyed by it.

so a HANDOFF or plan edit stales nothing; a README edit stales the doctests only (the `docs` phase
re-earns them in seconds, as the owner's docs-only rule of 2026-09-30 allows); anything else
stales everything. an exclusion is a fail-open surface: it rests on a survey (2026-10-01: no test
reads a markdown file other than the collected READMEs, and none reads `references/`), and
tests/test_gate_ledger.py::test_no_source_names_a_prose_path re-checks it mechanically.

the backend (owner, Q16(e), 2026-10-03): every phase but `gate:gmpy2` runs the pure path, with
MULTIINTERVAL_BACKEND removed, as CI's gate and fuzz jobs do; `gate:gmpy2` runs the whole suite with it
set to `gmpy2`, as ci.yml's `gate-gmpy2` job does. forced, `import multiinterval` raises unless gmpy2
is taken, so that run cannot pass on the pure path. it never covers the gate (a commit needs the
pure path); a push needs it, keyed by src like the fuzz, only when a file of BACKEND_FILES changed
since the base.

what a row does not see: gitignored files (`.hypothesis/`, so the fuzz database a run replayed),
the installed packages and python (recorded in the row's facts, never matched), and anything that
changes while a run is in progress, which is why the ids are taken again at the end and a run
whose code moved is recorded MOVED and counts for nothing either way.

a verdict is keyed by (phase, id), never by phase alone (zanzibar 2026-08-17): re-running a phase
on other code must not erase the green earned on this code, and a phase re-run red on the same code
is red. a run killed before it finishes (the 10-minute tool limit) leaves a log and no row; `status`
lists such logs as killed or still running, never as passed.

the ids hash the working tree's bytes, not git's: a checkout that rewrites line endings
(core.autocrlf) changes them, which over-invalidates (safe). `.gate-runs/` must stay gitignored:
an un-ignored ledger sits inside its own hash and every row is born stale.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = '.gate-runs'
LEDGER = 'ledger.tsv'
COLUMNS = ('started', 'dur_s', 'phase', 'status', 'code', 'src', 'facts', 'log')
ALGO = 'intervals-gate-ledger/1'
# fuzz.yml's default FUZZ_MULTIPLIER; a fuzz run below it does not clear a push
PUSH_MULTIPLIER = 10
# fuzz.yml's paths-ignore, written as globs there; `classify` is the rule
SRC_IGNORED = ('**.md', 'references/**')
UNKNOWN = 'unknown'
PARTS = {'itf': ['tests/itf1788'], 'rest': ['--ignore=tests/itf1788']}
PHASE_RE = re.compile(r'^(?:gate:(itf|rest)|fuzz-x([1-9]\d*):(itf|rest)|docs|(gate:gmpy2))$')
GMPY2 = 'gate:gmpy2'
# the modules that pick a double through the backend (the dispatch sites and the backend itself);
# tests/test_gate_ledger.py::test_the_backend_files_are_the_modules_naming_it keeps the list whole
BACKEND_FILES = ('multiinterval/_gmpy2.py', 'multiinterval/backend.py', 'multiinterval/elementary.py', 'multiinterval/ops.py')
COUNT_RE = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b')
SUMMARY_RE = re.compile(r'\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b.* in [\d.]+s')


class TreeIdError(RuntimeError):
    """the tree could not be read. never fall back to a guess: a guessed id can match a green row"""


def classify(path):
    """'prose' (in neither scope), 'readme' (code only) or 'src' (both)"""
    p = path.replace('\\', '/')
    if p.startswith('references/'):
        return 'prose'
    if p.endswith('.md'):
        return 'readme' if p.rsplit('/', 1)[-1] == 'README.md' else 'prose'
    return 'src'


def _git(repo, *args):
    """git's stdout as bytes, or TreeIdError; never an empty string standing for a failure"""
    try:
        done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TreeIdError(f'git {" ".join(args)} could not run: {exc}') from exc
    if done.returncode:
        tail = done.stderr.decode('utf-8', 'replace').strip().splitlines()
        raise TreeIdError(f'git {" ".join(args)} exited {done.returncode}' + (f': {tail[-1]}' if tail else ''))
    return done.stdout


def tree_ids(repo=REPO):
    """{'code': 'c:<12 hex>', 'src': 's:<12 hex>'} over the files' bytes; see the module docstring.

    a path in the index but gone from the worktree is skipped, not hashed as absent: a committed
    deletion then hashes as the pending one did (zanzibar's 2026-09-05b hole: committing a rename
    moved the id though no byte had changed)
    """
    listed = _git(repo, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    digests = {'code': hashlib.sha256(), 'src': hashlib.sha256()}
    for scope, h in digests.items():
        h.update(f'{ALGO}/{scope}\0'.encode())
    counts = dict.fromkeys(digests, 0)
    for rel in sorted({p for p in listed.split(b'\0') if p}):
        kind = classify(os.fsdecode(rel))
        if kind == 'prose':
            continue
        try:
            data = (Path(repo) / os.fsdecode(rel)).read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TreeIdError(f'cannot read {os.fsdecode(rel)}: {exc}') from exc
        entry = rel + b'\0' + hashlib.sha256(data).hexdigest().encode() + b'\0'
        for scope in ('code', 'src') if kind == 'src' else ('code',):
            digests[scope].update(entry)
            counts[scope] += 1
    if not counts['src']:
        raise TreeIdError('no source file in the tree; an id over nothing certifies nothing')
    return {scope: f'{scope[0]}:{h.hexdigest()[:12]}' for scope, h in digests.items()}


def head(repo=REPO):
    """a label for people (the row's facts, the report's header); never used to match a row"""
    try:
        sha = _git(repo, 'rev-parse', '--short', 'HEAD').decode().strip()
        dirty = bool(_git(repo, 'status', '--porcelain').strip())
    except TreeIdError:
        return 'no-head'
    return sha + ('+dirty' if dirty else '')


def runs_dir(repo=REPO):
    return Path(repo) / RUNS


def read_ledger(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) == len(COLUMNS):  # a torn append is skipped, not guessed at
            rows.append(dict(zip(COLUMNS, parts)))
    return rows


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        if new:
            f.write('# ' + '\t'.join(COLUMNS) + '\n')
        f.write('\t'.join(str(row[c]).replace('\t', ' ').replace('\n', ' ') for c in COLUMNS) + '\n')


# ---- running a phase

def phase_spec(phase, repo=REPO):
    """(pytest arguments, environment changes) for a phase name, or ValueError. a change of None removes
    the variable: MULTIINTERVAL_BACKEND is removed for every phase but gate:gmpy2, which sets it"""
    m = PHASE_RE.match(phase or '')
    if not m:
        raise ValueError(f'unknown phase {phase!r}: gate:itf, gate:rest, docs, fuzz-x<N>:itf, fuzz-x<N>:rest, '
                         f'{GMPY2}')
    gate_part, multiplier, fuzz_part, gmpy2 = m.groups()
    pure = {'MULTIINTERVAL_BACKEND': None, 'HYPOTHESIS_PROFILE': None, 'FUZZ_MULTIPLIER': None}
    if multiplier:
        return PARTS[fuzz_part], {**pure, 'HYPOTHESIS_PROFILE': 'fuzz', 'FUZZ_MULTIPLIER': multiplier}
    if gate_part:
        return PARTS[gate_part], pure
    if gmpy2:
        return [], {**pure, 'MULTIINTERVAL_BACKEND': 'gmpy2'}  # no arguments: the whole suite, as ci.yml runs it
    return collected_readmes(repo), pure


def collected_readmes(repo=REPO):
    """the README.md files under pyproject's testpaths: what pytest runs as doctests"""
    config = tomllib.loads((Path(repo) / 'pyproject.toml').read_text(encoding='utf-8'))
    paths = config['tool']['pytest']['ini_options']['testpaths']
    listed = _git(repo, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    found = []
    for rel in sorted({os.fsdecode(p) for p in listed.split(b'\0') if p}):
        if classify(rel) == 'readme' and any(rel == t or rel.startswith(t.rstrip('/') + '/') for t in paths):
            found.append(rel)
    return found


def _counts(text):
    """pytest's summary line, as {'passed': n, ...}, or None if the run printed none"""
    lines = [ln for ln in text.splitlines() if SUMMARY_RE.search(ln)]
    if not lines:
        return None
    counts = {}
    for n, word in COUNT_RE.findall(lines[-1]):
        counts['errors' if word.startswith('error') else word] = int(n)
    return counts


def _versions():
    out = [f'py={sys.version.split()[0]}']
    for name in ('hypothesis', 'pytest', 'numpy', 'gmpy2'):
        try:
            out.append(f'{name}={metadata.version(name)}')
        except metadata.PackageNotFoundError:
            out.append(f'{name}=none')
    return out


def run_phase(phase, repo=REPO, command=None):
    """run one phase, append its row, return 0 iff it is recorded PASSED.

    `command` replaces the pytest command (the ledger's own tests use a fake one). the backend
    variable is the phase's (`phase_spec`): removed for the pure phases, as CI's gate and fuzz jobs
    run the pure path, so a verdict on MULTIINTERVAL_BACKEND=gmpy2 never stands for one on the default;
    set to gmpy2 for gate:gmpy2, whatever the caller's environment says
    """
    args, changes = phase_spec(phase, repo)
    env = dict(os.environ)
    if env.get('MULTIINTERVAL_BACKEND') and changes['MULTIINTERVAL_BACKEND'] is None:
        print('gate: MULTIINTERVAL_BACKEND removed for this run (this phase runs the pure path)')
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    if command is None:
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *args]
    try:
        ids = tree_ids(repo)
    except TreeIdError as exc:
        print(f'gate: WARNING: cannot identify the tree ({exc}); this run will be recorded against '
              f'{UNKNOWN!r}, which matches nothing, so it will not count however it ends')
        ids = {'code': UNKNOWN, 'src': UNKNOWN}
    now = datetime.now().astimezone()
    out = runs_dir(repo)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f'{now:%Y%m%d-%H%M%S}-{phase.replace(":", "-")}-{os.getpid()}.log'
    label = head(repo)
    header = (f'# {phase} on {ids["code"]} {ids["src"]} (HEAD {label}), started {now.isoformat()}\n'
              f'# {" ".join(command)}\n').encode()
    with open(log, 'wb') as f:
        f.write(header)
        f.flush()
        print(f'gate: {phase} on {ids["code"]} {ids["src"]}, log {log.relative_to(repo).as_posix()}', flush=True)
        status = 'INTERRUPTED'
        rc = None
        try:
            rc = subprocess.run(command, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        finally:
            elapsed = int((datetime.now().astimezone() - now).total_seconds())
            # the run's own output only: the header echoes the command, which is not a verdict
            counts = _counts(log.read_bytes()[len(header):].decode('utf-8', 'replace'))
            if rc is not None:
                if rc == 0 and counts and counts.get('passed') and not counts.get('failed') and not counts.get('errors'):
                    status = 'PASSED'
                elif rc == 0:
                    status = 'INCONSISTENT'  # exit 0 without a passing summary: never green
                else:
                    status = 'FAILED'
            try:
                moved = tree_ids(repo) != ids
            except TreeIdError:
                moved = True
            if moved and ids['code'] != UNKNOWN:
                status = 'MOVED'  # the code changed under the run: it certifies neither tree
            facts = [f'rc={rc}'] + [f'{k}={v}' for k, v in (counts or {}).items()] + [f'head={label}'] + _versions()
            append_row(out / LEDGER, {'started': now.strftime('%Y-%m-%dT%H:%M:%S%z'), 'dur_s': elapsed,
                                      'phase': phase, 'status': status, 'code': ids['code'],
                                      'src': ids['src'], 'facts': ' '.join(facts), 'log': log.name})
    summary = ', '.join(f'{v} {k}' for k, v in (counts or {}).items()) or 'no pytest summary'
    print(f'gate: {phase} {status} ({summary}, {elapsed} s, rc={rc})')
    return 0 if status == 'PASSED' else 1


# ---- reading the ledger

def _fuzz(phase):
    m = PHASE_RE.match(phase)
    return (int(m.group(2)), m.group(3)) if m and m.group(2) else None


def _key(phase):
    """the id a phase's verdict is keyed by: src for the fuzz and gate:gmpy2, code for the rest"""
    return 'src' if _fuzz(phase) or phase == GMPY2 else 'code'


def green(rows, key, here, accept):
    """the phases accepted by `accept` whose last run with rows[key] == here PASSED"""
    if here == UNKNOWN:
        return set()
    last = {}
    for r in rows:  # appended in order: the last run per (phase, id) wins
        if r[key] == here and accept(r['phase']):
            last[r['phase']] = r
    return {p for p, r in last.items() if r['status'] == 'PASSED'}


def gate_covered(rows, ids):
    """{part: the phases that cover it} on this code; a fuzz run at any multiplier runs every test.
    gate:gmpy2 never covers the gate: CLAUDE.md's gate is the pure path"""
    ok = green(rows, 'code', ids['code'], lambda p: p.startswith(('gate:', 'fuzz-')) and p != GMPY2)
    return {part: sorted(p for p in ok if p.endswith(':' + part)) for part in PARTS}


def fuzz_covered(rows, ids):
    """{part: the fuzz phases at >= PUSH_MULTIPLIER green on this src}"""
    ok = green(rows, 'src', ids['src'], lambda p: (_fuzz(p) or (0,))[0] >= PUSH_MULTIPLIER)
    return {part: sorted(p for p in ok if p.endswith(':' + part)) for part in PARTS}


def gmpy2_covered(rows, ids):
    """whether gate:gmpy2's last run on this src passed"""
    return GMPY2 in green(rows, 'src', ids['src'], lambda p: p == GMPY2)


def readmes_covered(rows, ids):
    if 'docs' in green(rows, 'code', ids['code'], lambda p: p == 'docs'):
        return True
    return all(gate_covered(rows, ids).values())


def dirty_paths(repo=REPO):
    """uncommitted changes, untracked files included, that a push would not carry; prose aside"""
    lines = _git(repo, 'status', '--porcelain', '-z').decode('utf-8', 'replace').split('\0')
    out = []
    skip = False
    for ln in lines:
        if skip:
            skip = False
            continue
        if not ln:
            continue
        if ln[0] in 'RC':  # a rename's NUL-separated source follows
            skip = True
        path = ln[3:]
        if classify(path) != 'prose' and not path.startswith(RUNS + '/'):
            out.append(path)
    return out


def changed_since(base, repo=REPO):
    """paths changed between `base` and HEAD (three-dot, as prepush and github compare), or None"""
    try:
        return [p for p in _git(repo, 'diff', '--name-only', '-z', f'{base}...HEAD').decode().split('\0') if p]
    except TreeIdError:
        return None


def plan(rows, ids, changed, dirty):
    """(word, reasons): what a push of HEAD still needs. word is dirty, nothing, or the runs still
    needed joined by '+' in this order: docs or fuzz, then gmpy2 (gate:gmpy2, when a file of
    BACKEND_FILES changed since the base; owner, Q16(e), 2026-10-03).

    the src of `base` is taken as fuzzed and as run on gmpy2: every push to master is watched to the
    end of its fuzz run and of ci.yml's gmpy2 job (CLAUDE.md "push"), so src unchanged since
    origin/master needs no local fuzz run, and backend files unchanged since it no local gmpy2 run
    """
    if dirty:
        return 'dirty', [f'uncommitted: {", ".join(dirty[:5])}' + (' ...' if len(dirty) > 5 else '')]
    if changed is None:
        changed_kinds = {'src', 'readme'}
        reasons = ['the base could not be compared with HEAD: everything counts as changed']
    else:
        changed_kinds = {classify(p) for p in changed}
        reasons = [f'{len(changed)} file(s) changed since the base: '
                   + (', '.join(f'{k} {sum(classify(p) == k for p in changed)}' for k in ('src', 'readme', 'prose')
                                if any(classify(p) == k for p in changed)) or 'none')]
    need_fuzz = need_docs = need_gmpy2 = False
    if changed is None or any(p in BACKEND_FILES for p in changed):
        if gmpy2_covered(rows, ids):
            reasons.append(f'gmpy2: {GMPY2} green on {ids["src"]}')
        else:
            need_gmpy2 = True
            reasons.append(f'gmpy2: a backend file changed since the base, no {GMPY2} run green on {ids["src"]}')
    if 'src' in changed_kinds:
        fuzz = fuzz_covered(rows, ids)
        if all(fuzz.values()):
            reasons.append(f'fuzz: green on {ids["src"]} ({", ".join(p for v in fuzz.values() for p in v)})')
        else:
            need_fuzz = True
            missing = [part for part, v in fuzz.items() if not v]
            reasons.append(f'fuzz: no fuzz-x{PUSH_MULTIPLIER}+ run green on {ids["src"]} for {", ".join(missing)}')
    else:
        reasons.append('fuzz: src unchanged since the base, whose push was fuzzed on CI')
    if changed_kinds & {'src', 'readme'}:
        if readmes_covered(rows, ids):
            reasons.append(f'doctests: green on {ids["code"]}')
        elif not need_fuzz:
            need_docs = True
            reasons.append(f'doctests: no docs or gate run green on {ids["code"]}')
    words = ['fuzz'] * need_fuzz + ['docs'] * need_docs + ['gmpy2'] * need_gmpy2
    return '+'.join(words) or 'nothing', reasons


def _age(started):
    try:
        when = datetime.strptime(started, '%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        return '?'
    secs = (datetime.now(timezone.utc) - when).total_seconds()
    for unit, size in (('d', 86400), ('h', 3600), ('m', 60)):
        if secs >= size:
            return f'{int(secs // size)}{unit} ago'
    return f'{max(int(secs), 0)}s ago'


def report(repo=REPO, base='origin/master', require=None):
    try:
        ids = tree_ids(repo)
        err = None
    except TreeIdError as exc:
        ids, err = {'code': UNKNOWN, 'src': UNKNOWN}, str(exc)
    rows = read_ledger(runs_dir(repo) / LEDGER)
    print(f'tree: {ids["code"]} (code)  {ids["src"]} (src)  HEAD {head(repo)}')
    print(f'ledger: {RUNS}/{LEDGER}, {len(rows)} row(s)')
    if err:
        print(f'WARNING: cannot identify the tree: {err}. nothing below counts as green here')
    try:
        if _git(repo, 'status', '--porcelain', '--', RUNS).strip():
            print(f'WARNING: {RUNS}/ is not gitignored: the ledger is inside its own hash, every row born stale')
    except TreeIdError:
        pass

    phases = ['gate:itf', 'gate:rest', 'docs', f'fuzz-x{PUSH_MULTIPLIER}:itf', f'fuzz-x{PUSH_MULTIPLIER}:rest', GMPY2]
    phases += sorted({r['phase'] for r in rows} - set(phases))
    width = max(map(len, phases))
    print()
    for phase in phases:
        key = _key(phase)
        mine = [r for r in rows if r['phase'] == phase and r[key] == ids[key]]
        if mine:
            r = mine[-1]
            print(f'  {phase:{width}}  {r["status"]:12} {r["started"][:16].replace("T", " ")} '
                  f'({_age(r["started"])}, {r["dur_s"]} s) on this {key}  {r["facts"].split(" head=")[0]}')
            continue
        others = [r for r in rows if r['phase'] == phase]
        if others:
            r = others[-1]
            print(f'  {phase:{width}}  not run on this {key} (last: {r["status"]} {_age(r["started"])} on {r[key]})')
        else:
            print(f'  {phase:{width}}  never run')

    logged = {r['log'] for r in rows}
    orphans = sorted(p.name for p in runs_dir(repo).glob('*.log') if p.name not in logged)
    if orphans:
        print('\nlogs with no row (killed, or still running):')
        for name in orphans[-5:]:
            print(f'  {name}')

    covered = gate_covered(rows, ids)
    gate_ok = all(covered.values())
    print('\ncommit: the gate is ' + ('green on this code (' + '; '.join(
        f'{part} by {", ".join(v)}' for part, v in covered.items()) + ')' if gate_ok else
        'NOT green on this code (no green run for ' + ', '.join(p for p, v in covered.items() if not v) + ')'))
    try:
        dirty = dirty_paths(repo)
    except TreeIdError as exc:
        dirty = [f'(git status failed: {exc})']
    word, reasons = plan(rows, ids, changed_since(base, repo), dirty)
    say = {'dirty': 'commit first', 'nothing': 'nothing to run', 'docs': 'run the docs phase',
           'fuzz': f'run the fuzz at x{PUSH_MULTIPLIER}', 'gmpy2': f'run {GMPY2}'}
    print(f'push (base {base}): ' + ', then '.join(say[w] for w in word.split('+'))
          + (' (tools/prepush.sh does)' if word not in ('dirty', 'nothing') else ''))
    for reason in reasons:
        print(f'  {reason}')
    if require == 'commit':
        return 0 if gate_ok else 1
    if require == 'push':
        return 0 if word == 'nothing' else 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run', help='run one phase and record it')
    run.add_argument('phase')
    st = sub.add_parser('status', help='what is green on this code')
    st.add_argument('--require', choices=('commit', 'push'))
    st.add_argument('--base', default='origin/master')
    pl = sub.add_parser('plan', help='print what a push still needs: nothing, or docs/fuzz/gmpy2 joined by + (or dirty)')
    pl.add_argument('--base', default='origin/master')
    sub.add_parser('tree-id', help='print the two ids')
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        try:
            return run_phase(args.phase)
        except ValueError as exc:
            print(f'gate: {exc}', file=sys.stderr)
            return 2
    if args.cmd == 'status':
        return report(base=args.base, require=args.require)
    try:
        ids = tree_ids()
    except TreeIdError as exc:
        print(f'gate: cannot identify the tree: {exc}', file=sys.stderr)
        return 2
    if args.cmd == 'tree-id':
        print(ids['code'], ids['src'])
        return 0
    word, reasons = plan(read_ledger(runs_dir() / LEDGER), ids, changed_since(args.base), dirty_paths())
    for reason in reasons:
        print(f'gate: {reason}', file=sys.stderr)
    print(word)
    return 2 if word == 'dirty' else 0


if __name__ == '__main__':
    sys.exit(main())
