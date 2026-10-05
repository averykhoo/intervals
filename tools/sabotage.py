"""the sabotage engine: break a private copy of the repo one exact replacement at a time, and see each go red.

    $PY tools/sabotage.py run TABLE --name NAME              # snapshot HEAD, run the table's breaks on it
    $PY tools/sabotage.py run TABLE --name NAME --ref REV    # snapshot another commit (git archive)
    $PY tools/sabotage.py run TABLE --name NAME --worktree   # snapshot the working tree (tracked + untracked, not ignored)
    $PY tools/sabotage.py run TABLE --name NAME --dry-run    # snapshot and report each row's match count; no pytest
    $PY tools/sabotage.py stop NAME                          # kill that run by the PID it recorded, remove its copy

why: M13's sub-tasks wrote the same ~30-line loop nine times (HANDOFF T1, 2026-09-27), and three
hazards came out of those copies; the engine is built around them.

1. stale bytecode (M16c): a same-size break restored within the same second as the broken write left
   python running the broken `.pyc`. so every pytest child runs with PYTHONDONTWRITEBYTECODE=1, every
   `__pycache__` in the copy (and its `.hypothesis/`) is removed before AND after each break, the
   restore is `shutil.copy2` from a pristine copy (its mtime too), verified with
   `filecmp.cmp(shallow=False)` (cache cleared) and against the pristine bytes held in memory, and a
   control run on the intact copy comes first and again last (the closing control: the copy must come
   back green after every restore, or the run is void, exit 2).
2. stopped by command-line match (M16b): two harnesses shared the path `.scratch/sabotage.py`, and
   stopping one by matching it killed the other mid-break, so its `finally` never restored. every run
   has a `--name`; it writes its PID and the process's creation time to `.scratch/sabotage/<name>/pid`
   at launch, refuses to start while that PID is alive, and `stop <name>` kills only that PID's tree
   (and the pytest child's, from `child`), after checking the creation time so a reused PID is never
   killed. nothing here ever matches a process by name or command line.
3. a shared tree (the pown-huge review, 2026-09-29): the sabotage lens broke files in the worktree the
   soundness lens was probing. the engine never writes outside `.scratch/sabotage/<name>/`: it breaks a
   snapshot in `tree/` there (`git archive <ref>`, default HEAD, or `--worktree`), so a stop that skips
   the `finally` leaves a broken file only in a private copy, which `stop` removes and every run rebuilds.

the copy must import ITSELF. pyproject's `pythonpath = ["."]` is relative to the ini's rootdir, so a
whole snapshot (pyproject included) run with cwd = the copy imports the copy's package; a partial copy
fell through to the live package and "passed" vacuously (2026-10-04). that is checked mechanically, not
trusted: each pytest child loads a small plugin (`guard/_sabotage_guard.py`, written by the run) that,
at session end, lists every imported module whose file is under the live repo but outside this run's
directory, or whose top-level name is one the copy provides but whose file is not in the copy (an
editable install or a PYTHONPATH pointing at another checkout). a leak in a control aborts the run
(exit 2); in a break it is the verdict LEAK.

the break table (TOML, or JSON if it ends in .json):

    select = ["tests/test_fmt.py"]        # default pytest selection: node ids, files, "-k", "expr", ...
    timeout = 300                          # default seconds per pytest run (else --timeout, else 600)

    [[break]]
    id = "fmt-drop-sign"                   # unique; names the row's log file
    target = "multiinterval/fmt.py"        # repo-relative, '/' separated
    old = '''text that must occur exactly once in the target'''
    new = '''what it becomes'''
    select = ["tests/test_fmt.py::test_x"] # optional: this row's selection
    timeout = 60                           # optional
    expect = "green"                       # optional: a placebo, a break the selection should NOT see

`old` and `new` are matched as UTF-8 bytes. if the target has CRLF line ends, a newline in either is
written CRLF first (the table's own line ends do not matter); occurrences are counted overlapping, so
`aa` in `aaa` is AMBIGUOUS.

verdicts, one TSV line each in `.scratch/sabotage/<name>/verdicts.tsv`, appended as each row finishes
(pytest's output per row in `logs/<id>.log`):

* RED: the selection failed (pytest exit 1 with a failed test): caught
* GREEN: it passed: the break survived. exit 1, unless the row is a placebo (`expect = "green"`)
* TIMEOUT: it ran past the timeout; its process tree was killed. caught, listed separately
* ERROR: pytest exit 2-5 or another code, or exit 1 with errors and no failure (a collection or
  import error, a crash): caught, but weakly (an import error proves nothing about the check), flagged
* NOMATCH / AMBIGUOUS: `old` occurs 0 / more than 1 time (or the target is missing): the row did not
  run. exit 1: a row that cannot apply is reported, never skipped silently
* LEAK: the copy imported a module from outside itself (above). exit 1
* the control rows: PASSED or FAILED (pytest must exit 0 with a passed test, and the guard must
  report no leak); a failed control (first or closing) aborts with exit 2, and so does a restore that
  does not verify (RESTORE-FAILED)

exit: 0 every break caught (and every placebo GREEN); 1 a break survived, a row did not apply, a leak,
or a placebo was not GREEN; 2 the run proves nothing (bad table, control failed, restore failed, a
live run of that name, the snapshot failed).

each pytest child is `python -m pytest -q -x -p no:cacheprovider -p _sabotage_guard <selection>`, cwd
the copy, with the caller's environment plus PYTHONDONTWRITEBYTECODE=1 and the guard's directory
prepended to PYTHONPATH (so HYPOTHESIS_PROFILE etc. pass through). `-x`: one failure is a catch. a
timed-out child is killed with its whole tree: `taskkill /T /F` on Windows (psutil is not in the env,
2026-10-05), a new session and `killpg` on posix. the tree is removed at the end of a run unless
`--keep-tree` or a restore failed. tests/test_sabotage_tool.py pins all of it on toy repos.
"""
import argparse
import filecmp
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path('.scratch') / 'sabotage'
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
DEFAULT_TIMEOUT = 600
COLUMNS = ('finished', 'row', 'verdict', 'expect', 'rc', 'dur_s', 'counts', 'target', 'select', 'note')
COUNT_RE = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b')
SUMMARY_RE = re.compile(r'\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b.* in [\d.]+s')
GUARD_MODULE = '_sabotage_guard'
GUARD_SOURCE = '''\
"""written by tools/sabotage.py for one run: at the end of the pytest session, list every imported
module that did not come from the copy under test (see the engine's docstring)"""
import os
import sys


def _norm(p):
    return os.path.normcase(os.path.realpath(p))


def _under(p, root):
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def pytest_sessionfinish(session, exitstatus):
    live, run, tree = (_norm(os.environ[k]) for k in ('SABOTAGE_LIVE', 'SABOTAGE_RUN', 'SABOTAGE_TREE'))
    provided = set()
    for entry in os.listdir(tree):
        full = os.path.join(tree, entry)
        if entry.endswith('.py'):
            provided.add(entry[:-3])
        elif not entry.startswith('.') and os.path.isdir(full) and any(f.endswith('.py') for f in os.listdir(full)):
            provided.add(entry)
    leaks = []
    for name, module in list(sys.modules.items()):
        f = getattr(module, '__file__', None)
        if not f:
            continue
        p = _norm(f)
        if _under(p, tree) or _under(p, run):
            continue
        if _under(p, live) or name.split('.')[0] in provided:
            leaks.append(name + ' ' + f)
    with open(os.environ['SABOTAGE_LEAKS'], 'w', encoding='utf-8') as fh:
        fh.write(''.join(line + '\\n' for line in sorted(leaks)))
'''


class TableError(ValueError):
    """the break table is malformed: nothing runs"""


class Abort(RuntimeError):
    """the run proves nothing from here on (exit 2)"""


# ---- processes: by recorded PID only, never by name or command line

def process_token(pid):
    """a string naming the process `pid` (its creation time), or None if no such process is running.
    two processes with one PID at different times have different tokens, so a recorded (pid, token)
    never names a newcomer"""
    if pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            err = ctypes.get_last_error()
            return None if err == 87 else f'unreadable-{err}'  # 87: no such process
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return f'unreadable-{ctypes.get_last_error()}'
            return str(times[0].dwHighDateTime << 32 | times[0].dwLowDateTime)
        finally:
            k32.CloseHandle(handle)
    stat = Path(f'/proc/{pid}/stat')
    if stat.parent.exists():
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except OSError:
            return None
        return None if fields[0] in 'ZX' else fields[19]  # state; starttime is field 22
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return 'alive'  # no /proc (macOS): the PID alone


def kill_tree(pid):
    """kill `pid` and every process under it. on posix `pid` must lead its own session (the engine
    starts its pytest child so)"""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True)
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def read_pid(path):
    """(pid, token) from a pid file, or None if there is none or it is torn"""
    try:
        parts = path.read_text(encoding='utf-8').split()
        return int(parts[0]), parts[1]
    except (OSError, ValueError, IndexError):
        return None


def write_pid(path, pid):
    path.write_text(f'{pid} {process_token(pid)}\n', encoding='utf-8')


def _wait_gone(pid, token, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if process_token(pid) != token:
            return True
        time.sleep(0.1)
    return False


def _rmtree(path):
    """remove a tree; on Windows a just-killed process can hold a file for a moment, so retry"""
    def onexc(func, p, exc):
        os.chmod(p, 0o700)
        func(p)
    for attempt in range(20):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=onexc)
            return
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.25)


# ---- the table

def load_table(path, timeout=None):
    """the rows, each a dict with id, target, old, new, select (list), timeout, expect; TableError if malformed"""
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8')) if path.suffix == '.json' else tomllib.loads(raw.decode('utf-8'))
    except (OSError, ValueError) as exc:
        raise TableError(f'cannot read {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise TableError('the table must be a mapping with a `break` list')
    unknown = set(data) - {'select', 'timeout', 'break'}
    if unknown:
        raise TableError(f'unknown top-level keys: {sorted(unknown)}')
    default_select = data.get('select')
    default_timeout = data.get('timeout', timeout if timeout is not None else DEFAULT_TIMEOUT)
    rows = data.get('break')
    if not isinstance(rows, list) or not rows:
        raise TableError('no rows: the table needs at least one [[break]]')
    seen = set()
    out = []
    for i, row in enumerate(rows):
        where = f'row {i + 1}' + (f' ({row.get("id")!r})' if isinstance(row, dict) and 'id' in row else '')
        if not isinstance(row, dict):
            raise TableError(f'{where}: not a table')
        unknown = set(row) - {'id', 'target', 'old', 'new', 'select', 'timeout', 'expect'}
        if unknown:
            raise TableError(f'{where}: unknown keys {sorted(unknown)}')
        for key in ('id', 'target', 'old', 'new'):
            if not isinstance(row.get(key), str):
                raise TableError(f'{where}: `{key}` must be a string')
        if not NAME_RE.match(row['id']):
            raise TableError(f'{where}: id must match {NAME_RE.pattern} (it names a log file)')
        if row['id'] in seen or row['id'].startswith('control'):
            raise TableError(f'{where}: duplicate or reserved id')
        seen.add(row['id'])
        if not row['old']:
            raise TableError(f'{where}: `old` is empty')
        if row['old'] == row['new']:
            raise TableError(f'{where}: `new` is `old`: not a break')
        target = row['target'].replace('\\', '/')
        if target.startswith('/') or '..' in target.split('/') or ':' in target:
            raise TableError(f'{where}: target must be a relative path inside the repo')
        select = row.get('select', default_select)
        if not isinstance(select, list) or not select or not all(isinstance(s, str) and s for s in select):
            raise TableError(f'{where}: `select` must be a non-empty list of strings (here or at the top)')
        limit = row.get('timeout', default_timeout)
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0:
            raise TableError(f'{where}: `timeout` must be a positive number of seconds')
        expect = row.get('expect', 'red')
        if expect not in ('red', 'green'):
            raise TableError(f'{where}: `expect` is "red" (default) or "green" (a placebo)')
        out.append({'id': row['id'], 'target': target, 'old': row['old'], 'new': row['new'],
                    'select': list(select), 'timeout': limit, 'expect': expect})
    return out, default_timeout


def occurrences(data, needle):
    """how many times `needle` occurs in `data`, overlapping ones counted"""
    count, start = 0, data.find(needle)
    while start != -1:
        count += 1
        start = data.find(needle, start + 1)
    return count


def to_target_endings(text, data):
    """`text` as bytes, its newlines written as the target's: CRLF if the target has any"""
    text = text.replace('\r\n', '\n')
    if b'\r\n' in data:
        text = text.replace('\n', '\r\n')
    return text.encode('utf-8')


# ---- the copy

def snapshot(repo, tree, ref=None, worktree=False):
    """fill `tree` with the repo at `ref` (git archive) or the working tree; returns a label"""
    tree.mkdir(parents=True)
    if worktree:
        done = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                              capture_output=True)
        if done.returncode:
            raise Abort(f'git ls-files failed: {done.stderr.decode("utf-8", "replace").strip()}')
        listed = done.stdout
        count = 0
        for rel in sorted({os.fsdecode(p) for p in listed.split(b'\0') if p}):
            if rel.startswith(SCRATCH.as_posix() + '/'):
                continue  # never copy a run into itself, should .scratch/ not be ignored
            src = repo / rel
            if not src.is_file():
                continue  # deleted in the working tree
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        return f'working tree ({count} files) on {head.stdout.strip() or "no HEAD"}'
    ref = ref or 'HEAD'
    sha = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', '--short', f'{ref}^{{commit}}'],
                         capture_output=True, text=True)
    if sha.returncode:
        raise Abort(f'not a commit: {ref!r}')
    archive = subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', sha.stdout.strip()],
                             capture_output=True)
    if archive.returncode:
        raise Abort(f'git archive {ref} failed: {archive.stderr.decode("utf-8", "replace").strip()}')
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(tree, filter='data')
    return f'{ref} = {sha.stdout.strip()} (git archive)'


def clear_caches(tree):
    """remove every __pycache__ in the copy and its .hypothesis/: no bytecode or example database
    from another version of the code may be read by the next run"""
    for dirpath, dirnames, _ in os.walk(tree):
        for d in list(dirnames):
            if d == '__pycache__':
                _rmtree(Path(dirpath) / d)
                dirnames.remove(d)
    _rmtree(tree / '.hypothesis')


def restore(pristine, target):
    shutil.copy2(pristine, target)


def verify_restore(pristine, target, pristine_bytes):
    """the restored target is the pristine bytes, by content (filecmp's cache keys on stat, so cleared)"""
    filecmp.clear_cache()
    return filecmp.cmp(pristine, target, shallow=False) and target.read_bytes() == pristine_bytes


# ---- one pytest run

def _counts(text):
    lines = [ln for ln in text.splitlines() if SUMMARY_RE.search(ln)]
    counts = {}
    for n, word in COUNT_RE.findall(lines[-1] if lines else ''):
        counts['errors' if word.startswith('error') else word] = int(n)
    return counts


class Run:
    def __init__(self, repo, name):
        self.repo = Path(repo).resolve()
        self.name = name
        self.dir = self.repo / SCRATCH / name
        self.tree = self.dir / 'tree'
        self.pristine = self.dir / 'pristine'
        self.logs = self.dir / 'logs'
        self.guard = self.dir / 'guard'
        self.tsv = self.dir / 'verdicts.tsv'

    def env(self, leaks):
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(self.guard)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        env.update(SABOTAGE_LIVE=str(self.repo), SABOTAGE_RUN=str(self.dir), SABOTAGE_TREE=str(self.tree),
                   SABOTAGE_LEAKS=str(leaks))
        return env

    def pytest(self, row_id, select, timeout):
        """run the selection on the copy: {'rc', 'timeout', 'counts', 'leaks' (None: the guard did not
        report), 'dur'}"""
        log = self.logs / f'{row_id}.log'
        leaks = self.guard / f'{row_id}.leaks'
        leaks.unlink(missing_ok=True)
        command = [sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-p', GUARD_MODULE, *select]
        start = time.monotonic()
        timed_out = False
        with open(log, 'wb') as out:
            out.write(f'# cwd {self.tree}\n# {" ".join(command)}\n'.encode())
            out.flush()
            child = subprocess.Popen(command, cwd=self.tree, env=self.env(leaks), stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=(os.name != 'nt'))
            write_pid(self.dir / 'child', child.pid)
            try:
                rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(child.pid)
                rc = child.wait()
            finally:
                if child.poll() is None:  # interrupted while waiting: never leave the child running
                    kill_tree(child.pid)
                    child.wait()
                (self.dir / 'child').unlink(missing_ok=True)
        text = log.read_bytes().decode('utf-8', 'replace')
        found = leaks.read_text(encoding='utf-8').splitlines() if leaks.exists() else None
        return {'rc': rc, 'timeout': timed_out, 'counts': _counts(text), 'leaks': found,
                'dur': time.monotonic() - start}

    def write(self, row, verdict, result=None, expect='', target='', select=(), note=''):
        result = result or {}
        counts = ','.join(f'{k}={v}' for k, v in result.get('counts', {}).items())
        line = {'finished': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'row': row, 'verdict': verdict,
                'expect': expect, 'rc': '' if result.get('rc') is None else result['rc'],
                'dur_s': f'{result["dur"]:.1f}' if 'dur' in result else '', 'counts': counts, 'target': target,
                'select': ' '.join(select), 'note': note}
        with open(self.tsv, 'a', encoding='utf-8', newline='\n') as f:
            f.write('\t'.join(str(line[c]).replace('\t', ' ').replace('\n', ' ') for c in COLUMNS) + '\n')
        shown = f'  {verdict:10} {row}' + (f'  ({counts}, {line["dur_s"]} s)' if result else '') + (f'  {note}' if note else '')
        print(shown, flush=True)


def control_ok(result):
    """a control passes: exit 0, something passed, nothing failed, and the guard saw no leak"""
    c = result['counts']
    return (not result['timeout'] and result['rc'] == 0 and c.get('passed', 0) > 0 and not c.get('failed')
            and not c.get('errors') and result['leaks'] == [])


def verdict_of(result):
    """(verdict, note) for a break's pytest result"""
    c = result['counts']
    if result['timeout']:
        return 'TIMEOUT', 'killed with its process tree'
    if result['leaks']:
        return 'LEAK', f'imported from outside the copy: {"; ".join(result["leaks"][:3])}'
    rc = result['rc']
    if rc == 0:
        return 'GREEN', '' if c.get('passed') else 'exit 0 with no test passed'
    if rc == 1 and c.get('failed'):
        return 'RED', ''
    if rc == 1:
        return 'ERROR', 'errors without a failed test (setup or import): a weak catch'
    meaning = {2: 'interrupted: a collection or import error', 3: 'internal error', 4: 'usage error',
               5: 'no test collected'}.get(rc, 'a crash or an unexpected exit code')
    return 'ERROR', f'pytest exit {rc}, {meaning}: a weak catch'


def _controls(run, selections, timeout, label):
    for i, select in enumerate(selections):
        row_id = label if len(selections) == 1 else f'{label}-{i + 1}'
        result = run.pytest(row_id, select, timeout)
        ok = control_ok(result)
        note = ''
        if result['leaks']:
            note = f'imported from outside the copy: {"; ".join(result["leaks"][:3])}'
        elif result['leaks'] is None:
            note = 'the import guard did not report'
        elif result['timeout']:
            note = 'timed out'
        run.write(row_id, 'PASSED' if ok else 'FAILED', result, select=select, note=note)
        if not ok:
            raise Abort(f'{row_id} {" ".join(select)} is not green on the intact copy ({note or "see its log"}): '
                        f'every break would be meaningless; logs/{row_id}.log')


def run_table(table, name, repo=REPO, ref=None, worktree=False, timeout=None, keep_tree=False, dry_run=False):
    """run every row on a private copy; returns the exit code (see the module docstring)"""
    try:
        rows, default_timeout = load_table(table, timeout)
    except TableError as exc:
        print(f'sabotage: bad table: {exc}', file=sys.stderr)
        return 2
    if not NAME_RE.match(name):
        print(f'sabotage: --name must match {NAME_RE.pattern}', file=sys.stderr)
        return 2
    run = Run(repo, name)
    pidfile = run.dir / 'pid'
    held = read_pid(pidfile)
    if held and process_token(held[0]) == held[1]:
        print(f'sabotage: run {name!r} is live (pid {held[0]}); `stop {name}` it or pick another --name', file=sys.stderr)
        return 2
    if run.dir.exists():
        _rmtree(run.dir)  # this name's previous run, dead: its copy is ours to rebuild
    run.dir.mkdir(parents=True)
    write_pid(pidfile, os.getpid())
    keep = keep_tree
    try:
        for d in (run.pristine, run.logs, run.guard):
            d.mkdir()
        (run.guard / f'{GUARD_MODULE}.py').write_text(GUARD_SOURCE, encoding='utf-8', newline='\n')
        source = snapshot(run.repo, run.tree, ref, worktree)
        with open(run.tsv, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'# sabotage run {name} of {Path(table).name} on {source}, started {datetime.now().isoformat()}\n')
            f.write('# ' + '\t'.join(COLUMNS) + '\n')
        print(f'sabotage: {name}: {len(rows)} row(s) on a copy of {source}', flush=True)
        print(f'sabotage: copy {run.tree}, verdicts {run.tsv}', flush=True)
        if dry_run:
            for row in rows:
                path = run.tree / row['target']
                n = occurrences(path.read_bytes(), to_target_endings(row['old'], path.read_bytes())) if path.is_file() else 0
                print(f'  {row["id"]}: {n} match(es) in {row["target"]}' + ('' if path.is_file() else ' (no such file)'))
            return 0
        selections = []
        for row in rows:
            if row['select'] not in selections:
                selections.append(row['select'])
        clear_caches(run.tree)
        _controls(run, selections, default_timeout, 'control')
        outcomes = []
        for row in rows:
            outcomes.append((row, break_row(run, row)))
        clear_caches(run.tree)
        _controls(run, selections, default_timeout, 'control-closing')
        return summarize(run, outcomes)
    except Abort as exc:
        print(f'sabotage: ABORTED: {exc}', file=sys.stderr, flush=True)
        keep = True  # evidence: the copy as it was
        return 2
    finally:
        if not keep and run.tree.exists():
            _rmtree(run.tree)
        if read_pid(pidfile) and read_pid(pidfile)[0] == os.getpid():
            pidfile.unlink()


def break_row(run, row):
    """apply one break to the copy, run its selection, restore; returns the verdict"""
    target = run.tree / row['target']
    common = {'expect': row['expect'], 'target': row['target'], 'select': row['select']}
    if not target.is_file():
        run.write(row['id'], 'NOMATCH', note='no such file in the copy', **common)
        return 'NOMATCH'
    data = target.read_bytes()
    old, new = to_target_endings(row['old'], data), to_target_endings(row['new'], data)
    n = occurrences(data, old)
    if n != 1:
        verdict = 'NOMATCH' if n == 0 else 'AMBIGUOUS'
        run.write(row['id'], verdict, note=f'`old` occurs {n} times', **common)
        return verdict
    pristine = run.pristine / row['target']
    if pristine.exists():
        if pristine.read_bytes() != data:
            raise Abort(f'{row["target"]} in the copy differs from its pristine copy before row {row["id"]}')
    else:
        pristine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, pristine)
    applied = run.dir / 'applied'
    applied.write_text(f'{row["id"]} {row["target"]}\n', encoding='utf-8')
    result = None
    try:
        target.write_bytes(data.replace(old, new))
        clear_caches(run.tree)
        result = run.pytest(row['id'], row['select'], row['timeout'])
    finally:
        restore(pristine, target)
        if not verify_restore(pristine, target, data):
            run.write(row['id'], 'RESTORE-FAILED', result, note='the restored target is not the pristine bytes', **common)
            raise Abort(f'row {row["id"]}: {row["target"]} did not restore byte for byte; the copy is kept')
        clear_caches(run.tree)
        applied.unlink()
    verdict, note = verdict_of(result)
    if row['expect'] == 'green':
        note = (note + '; ' if note else '') + ('placebo, as expected' if verdict == 'GREEN' else 'PLACEBO NOT GREEN')
    run.write(row['id'], verdict, result, note=note, **common)
    return verdict


def summarize(run, outcomes):
    tally = {}
    for _, verdict in outcomes:
        tally[verdict] = tally.get(verdict, 0) + 1
    print('sabotage: ' + ', '.join(f'{n} {v}' for v, n in sorted(tally.items())) + f'  ({run.tsv})')
    bad = []
    for row, verdict in outcomes:
        if row['expect'] == 'green':
            if verdict != 'GREEN':
                bad.append(f'placebo {row["id"]} came out {verdict}: the selection sees more than it should')
        elif verdict == 'GREEN':
            bad.append(f'{row["id"]} SURVIVED: {row["target"]} broken, {" ".join(row["select"])} still green')
        elif verdict in ('NOMATCH', 'AMBIGUOUS', 'LEAK'):
            bad.append(f'{row["id"]} {verdict}: the row did not test what it says')
    weak = [row['id'] for row, verdict in outcomes if verdict == 'ERROR']
    hung = [row['id'] for row, verdict in outcomes if verdict == 'TIMEOUT']
    if hung:
        print(f'sabotage: timed out (caught, killed): {", ".join(hung)}')
    if weak:
        print(f'sabotage: ERROR, a weak catch (collection, import or crash): {", ".join(weak)}')
    for line in bad:
        print(f'sabotage: {line}')
    return 1 if bad else 0


# ---- stop

def stop(name, repo=REPO):
    """kill the run `name` by the PID it recorded (and its pytest child's tree), then remove its copy"""
    if not NAME_RE.match(name):
        print(f'sabotage: not a run name: {name!r}', file=sys.stderr)
        return 2
    run = Run(repo, name)
    pidfile = run.dir / 'pid'
    held = read_pid(pidfile)
    if held is None:
        print(f'sabotage: no pid file at {pidfile}: nothing to stop. a run is stopped only by the PID it '
              f'recorded, never by matching a command line', file=sys.stderr)
        return 2
    pid, token = held
    now = process_token(pid)
    if now is None:
        print(f'sabotage: run {name!r} (pid {pid}) is not running; its pid file is stale and is removed')
        pidfile.unlink(missing_ok=True)
        return 0
    if now != token:
        print(f'sabotage: pid {pid} is now another process (created {now}, the run recorded {token}): '
              f'refusing to kill it', file=sys.stderr)
        return 2
    applied = (run.dir / 'applied').read_text(encoding='utf-8').strip() if (run.dir / 'applied').exists() else ''
    child = read_pid(run.dir / 'child')
    # the engine first: killed second, it saw its child die, recorded a false RED, restored and ran on
    # (found 2026-10-05, the M4 row of the engine's own sabotage run). then the child's tree, which on
    # posix is its own session and outlives the engine
    kill_tree(pid)
    if not _wait_gone(pid, token):
        print(f'sabotage: pid {pid} survived the kill', file=sys.stderr)
        return 2
    if child and process_token(child[0]) == child[1]:
        kill_tree(child[0])
        _wait_gone(*child)
    _rmtree(run.tree)
    pidfile.unlink(missing_ok=True)
    (run.dir / 'child').unlink(missing_ok=True)
    if run.tsv.exists():
        with open(run.tsv, 'a', encoding='utf-8', newline='\n') as f:
            f.write(f'# stopped {datetime.now().isoformat()} by `stop {name}`'
                    + (f' with {applied} applied' if applied else '') + '; the copy was removed\n')
    print(f'sabotage: stopped {name!r} (pid {pid})' + (f'; row {applied} was applied' if applied else '')
          + f'; removed its copy {run.tree}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repo', default=str(REPO), help=argparse.SUPPRESS)  # the tests' toy repos
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run a break table on a private copy')
    r.add_argument('table', help='a TOML (or .json) break table')
    r.add_argument('--name', required=True, help='this run: .scratch/sabotage/<name>/, and what `stop` takes')
    src = r.add_mutually_exclusive_group()
    src.add_argument('--ref', help='the commit to snapshot (default HEAD)')
    src.add_argument('--worktree', action='store_true', help='snapshot the working tree instead')
    r.add_argument('--timeout', type=float, help=f'seconds per pytest run when the table names none ({DEFAULT_TIMEOUT})')
    r.add_argument('--keep-tree', action='store_true', help='keep the copy after the run')
    r.add_argument('--dry-run', action='store_true', help='snapshot and count each row\'s matches; run nothing')
    s = sub.add_parser('stop', help='kill a run by the PID it recorded, and remove its copy')
    s.add_argument('name')
    args = ap.parse_args(argv)
    if args.cmd == 'stop':
        return stop(args.name, args.repo)
    return run_table(args.table, args.name, args.repo, args.ref, args.worktree, args.timeout, args.keep_tree,
                     args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
