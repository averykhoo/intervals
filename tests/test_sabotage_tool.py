"""
the sabotage engine (tools/sabotage.py), on toy repos in tmp_path: a package `toypkg` whose `core.py` has
CRLF line ends, and a toy test file. one run of a mixed table pins the verdicts (RED, GREEN, a placebo,
NOMATCH, AMBIGUOUS with overlapping matches, a multi-line CRLF break, a collection error as ERROR) and
the byte-exact restore; the others pin the hazards the engine exists for: a red control aborts, a hung
child is killed with its tree and the target still restored, the copy imports itself (a broken working
tree does not reach a HEAD snapshot; a module reached from the live repo is a LEAK), a stale `.pyc`
cannot stand in for the broken source, a failed restore aborts, and a run is stopped only by the PID
it recorded. every pytest child is real; the toy suite runs with plugin autoload off, so each child
costs about a second (35 passed in 24 s, 2026-10-05)

sabotaged 2026-10-05 with the engine itself (`tools/sabotage.py run <table> --worktree`, this file as the
selection; control and closing control 35 passed): each guard broken alone, each RED, first red test shown

    S1  skip the first control                     test_mixed_verdicts[control-PASSED]
    S2  never remove __pycache__                   test_a_stale_pyc_cannot_stand_in
    S3  TIMEOUT reported as GREEN                  test_a_timeout_kills_the_tree_and_restores
    S4  skip the restore verify                    test_a_failed_restore_aborts
    S5  count matches non-overlapping              test_mixed_verdicts[overlapping-AMBIGUOUS]
    S6  no PYTHONDONTWRITEBYTECODE for the child   test_mixed_verdicts[control-PASSED]
    S7  taskkill without /T (the child, not its tree)  test_a_timeout_kills_the_tree_and_restores
    S8  a control ignores the import guard         test_a_module_from_the_live_repo_is_a_leak
    S9  stop skips the creation-time check         test_stop_refuses_a_reused_pid
    S10 pytest's cwd the live repo, not the copy   test_mixed_verdicts[control-PASSED]
    S11 skip the closing control                   test_mixed_verdicts[control-closing-PASSED]
    S12 stop goes on without a pid file            test_stop_refuses_a_name_with_no_pid_file

and a placebo (a comment in main()) came out GREEN. the `clear_caches` after each restore is not in the
table: the next row's clear and the one before the closing control cover it by design
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('sabotage', ROOT / 'tools' / 'sabotage.py')
sabotage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sabotage)

CORE = '\r\n'.join([
    'import os',
    'import subprocess',
    'import sys',
    'import time',
    '',
    "PAD = 'aaa'",
    '',
    '',
    'def double(x):',
    '    return 2 * x',
    '',
    '',
    'def label(x):',
    '    # a comment',
    '    if x < 0:',
    "        return 'neg'",
    "    return 'pos'",
    '',
    '',
    'def unused():',
    "    return 'never called'",
    '',
    '',
    'def hang():',
    "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])",
    "    with open(os.environ['TOY_GRANDCHILD'], 'w') as f:",
    '        f.write(str(child.pid))',
    '    time.sleep(120)',
    '',
]).encode()

TOY_TEST = b'''import importlib.util
import os
import py_compile
import sys

import toypkg.core as core

if os.environ.get('TOY_PLANT_PYC'):
    # a pyc python trusts whatever the source now says (unchecked hash): the deterministic stand-in for
    # M16c's stale .pyc, which a same-size write within the same second produces by accident
    py_compile.compile(core.__file__, cfile=importlib.util.cache_from_source(core.__file__),
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)


def test_double():
    assert core.double(3) == 6


def test_label():
    assert core.label(-1) == 'neg' and core.label(1) == 'pos'


def test_no_bytecode_is_written():
    assert sys.dont_write_bytecode
'''

FILES = {
    '.gitignore': b'.scratch/\n',
    'pyproject.toml': b'[tool.pytest.ini_options]\npythonpath = ["."]\ntestpaths = ["tests"]\n',
    'toypkg/__init__.py': b'',
    'toypkg/core.py': CORE,
    'tests/test_toy.py': TOY_TEST,
}
SELECT = ['tests/test_toy.py']


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                   check=True, capture_output=True)


def make_repo(path, changes=None):
    """a committed toy repo, `changes` ({path: bytes}) over FILES"""
    files = {**FILES, **(changes or {})}
    _git(path, 'init', '-q', '-b', 'master')
    _git(path, 'config', 'core.autocrlf', 'false')
    for rel, data in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(data)
    _git(path, 'add', '-A')
    _git(path, 'commit', '-q', '-m', 'toy')
    return path


def write_table(path, rows, **top):
    table = path / 'table.json'
    table.write_text(json.dumps({'select': SELECT, 'timeout': 60, **top, 'break': rows}), encoding='utf-8')
    return table


def verdicts(repo, name):
    """{row: (verdict, note)} from the run's TSV"""
    out = {}
    for line in (repo / '.scratch/sabotage' / name / 'verdicts.tsv').read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            cells = line.split('\t')
            out[cells[1]] = (cells[2], cells[9])
    return out


def tree_file(repo, name, rel):
    return (repo / '.scratch/sabotage' / name / 'tree' / rel).read_bytes()


def wait_gone(pid, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if sabotage.process_token(pid) is None:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(autouse=True)
def _fast_children(monkeypatch):
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')  # the toy needs no plugin: ~1 s a child
    for var in ('PYTEST_ADDOPTS', 'TOY_PLANT_PYC', 'PYTHONPATH'):
        monkeypatch.delenv(var, raising=False)


# ---- one run of a mixed table

MIXED = '''\
select = ["tests/test_toy.py"]
timeout = 60

[[break]]
id = "caught"
target = "toypkg/core.py"
old = "return 2 * x"
new = "return 2 + x"

[[break]]
id = "survives"
target = "toypkg/core.py"
old = "'never called'"
new = "'not called!!'"

[[break]]
id = "placebo"
target = "toypkg/core.py"
old = "# a comment"
new = "# a remark!"
expect = "green"

[[break]]
id = "multiline-crlf"
target = "toypkg/core.py"
old = """if x < 0:
        return 'neg'"""
new = """if x > 0:
        return 'neg'"""

[[break]]
id = "nomatch"
target = "toypkg/core.py"
old = "return 3 * x"
new = "return 4 * x"

[[break]]
id = "no-such-file"
target = "toypkg/nope.py"
old = "x"
new = "y"

[[break]]
id = "ambiguous"
target = "toypkg/core.py"
old = "return"
new = "yield"

[[break]]
id = "overlapping"
target = "toypkg/core.py"
old = "aa"
new = "bb"

[[break]]
id = "collection-error"
target = "toypkg/core.py"
old = "def double(x):"
new = "def double(x)"
'''


@pytest.fixture(scope='module')
def mixed(tmp_path_factory):
    repo = make_repo(tmp_path_factory.mktemp('mixed'))
    table = repo / 'mixed.toml'
    table.write_bytes(MIXED.encode())
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
        for var in ('PYTEST_ADDOPTS', 'TOY_PLANT_PYC', 'PYTHONPATH'):
            mp.delenv(var, raising=False)
        rc = sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'mixed', '--keep-tree'])
    return repo, rc, verdicts(repo, 'mixed')


@pytest.mark.parametrize('row, verdict', [
    ('control', 'PASSED'), ('caught', 'RED'), ('survives', 'GREEN'), ('placebo', 'GREEN'),
    ('multiline-crlf', 'RED'), ('nomatch', 'NOMATCH'), ('no-such-file', 'NOMATCH'), ('ambiguous', 'AMBIGUOUS'),
    ('overlapping', 'AMBIGUOUS'), ('collection-error', 'ERROR'), ('control-closing', 'PASSED'),
])
def test_mixed_verdicts(mixed, row, verdict):
    assert mixed[2][row][0] == verdict


def test_a_survivor_exits_1(mixed):
    """GREEN, NOMATCH and AMBIGUOUS each fail the run; the placebo's GREEN does not"""
    assert mixed[1] == 1
    assert 'placebo, as expected' in mixed[2]['placebo'][1]


def test_the_copy_is_restored_byte_for_byte(mixed):
    repo = mixed[0]
    assert tree_file(repo, 'mixed', 'toypkg/core.py') == CORE  # CRLF kept
    assert not list((repo / '.scratch/sabotage/mixed/tree').rglob('__pycache__'))
    assert (repo / 'toypkg/core.py').read_bytes() == CORE  # the live tree was never touched


def test_the_run_leaves_no_pid_file(mixed):
    assert not (mixed[0] / '.scratch/sabotage/mixed/pid').exists()


def test_all_caught_exits_0_and_removes_the_copy(tmp_path):
    repo = make_repo(tmp_path)
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return 2 - x'}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'ok']) == 0
    assert verdicts(repo, 'ok')['caught'][0] == 'RED'
    assert not (repo / '.scratch/sabotage/ok/tree').exists()


@pytest.mark.parametrize('row, message', [
    ({'id': 'a', 'target': 'toypkg/core.py', 'old': 'x', 'new': 'x'}, 'not a break'),
    ({'id': 'a', 'target': '../core.py', 'old': 'x', 'new': 'y'}, 'inside the repo'),
    ({'id': 'control', 'target': 'toypkg/core.py', 'old': 'x', 'new': 'y'}, 'reserved'),
    ({'id': 'a', 'target': 'toypkg/core.py', 'old': 'x', 'new': 'y', 'select': []}, 'select'),
    ({'id': 'a', 'target': 'toypkg/core.py', 'old': 'x', 'new': 'y', 'expect': 'maybe'}, 'expect'),
    ({'id': 'a', 'target': 'toypkg/core.py', 'old': '', 'new': 'y'}, 'empty'),
    ({'id': 'a', 'target': 'toypkg/core.py', 'new': 'y'}, '`old`'),
])
def test_a_bad_row_refuses_the_table(tmp_path, row, message):
    with pytest.raises(sabotage.TableError, match=message):
        sabotage.load_table(write_table(tmp_path, [row]))


def test_duplicate_ids_refuse_the_table(tmp_path):
    row = {'id': 'a', 'target': 'toypkg/core.py', 'old': 'x', 'new': 'y'}
    table = write_table(tmp_path, [row, row])
    with pytest.raises(sabotage.TableError, match='duplicate'):
        sabotage.load_table(table)
    assert sabotage.main(['--repo', str(tmp_path), 'run', str(table), '--name', 'dup']) == 2


def test_occurrences_counts_overlaps():
    assert sabotage.occurrences(b'aaa', b'aa') == 2 and b'aaa'.count(b'aa') == 1


# ---- the hazards

def test_a_red_control_aborts(tmp_path, capsys):
    repo = make_repo(tmp_path, {'toypkg/core.py': CORE.replace(b'return 2 * x', b'return 2 * x + 1')})
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': "return 'pos'",
                                    'new': "return 'neg'"}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'red']) == 2
    assert verdicts(repo, 'red') == {'control': ('FAILED', '')}  # no break ran
    assert 'ABORTED' in capsys.readouterr().err


def test_a_timeout_kills_the_tree_and_restores(tmp_path, monkeypatch):
    """a hung child is killed with its own child, the row is TIMEOUT (caught, exit 0), and the target
    comes back byte for byte"""
    repo = make_repo(tmp_path)
    marker = tmp_path / 'grandchild.pid'
    table = write_table(tmp_path, [{'id': 'hangs', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return hang()', 'timeout': 6}])
    monkeypatch.setenv('TOY_GRANDCHILD', str(marker))
    rc = sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'hang', '--keep-tree'])
    grandchild = int(marker.read_text())
    try:
        assert wait_gone(grandchild), 'the timed-out child\'s own child survived'
    finally:
        sabotage.kill_tree(grandchild)
    assert rc == 0
    got = verdicts(repo, 'hang')
    assert got['hangs'][0] == 'TIMEOUT' and got['control-closing'][0] == 'PASSED'
    assert tree_file(repo, 'hang', 'toypkg/core.py') == CORE
    assert not (repo / '.scratch/sabotage/hang/child').exists()


def test_a_head_snapshot_does_not_see_the_working_tree(tmp_path):
    """the copy imports the copy: the live checkout's package is broken, HEAD's is not, and the control
    on a HEAD snapshot passes; the same table on --worktree sees the broken package and aborts"""
    repo = make_repo(tmp_path)
    (repo / 'toypkg/core.py').write_bytes(CORE.replace(b'return 2 * x', b'return 2 * x + 1'))
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': "return 'pos'",
                                    'new': "return 'neg'"}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'head']) == 0
    assert verdicts(repo, 'head')['caught'][0] == 'RED'
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'wt', '--worktree']) == 2
    assert verdicts(repo, 'wt')['control'][0] == 'FAILED'


def test_a_module_from_the_live_repo_is_a_leak(tmp_path, monkeypatch):
    """the vacuous pass of 2026-10-04: the copy's test imports a module the snapshot lacks, found in the
    live checkout through PYTHONPATH (as an editable install would). the import succeeds, the toy test
    passes, and the guard still aborts the run"""
    repo = make_repo(tmp_path, {'tests/test_extra.py': b'import toyextra\n\n\ndef test_extra():\n    assert toyextra.OK\n'})
    (repo / 'toyextra.py').write_bytes(b'OK = True\n')  # untracked: not in the HEAD snapshot
    monkeypatch.setenv('PYTHONPATH', str(repo))
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return 2 - x'}], select=['tests'])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'leak']) == 2
    verdict, note = verdicts(repo, 'leak')['control']
    assert verdict == 'FAILED' and 'outside the copy' in note and 'toyextra' in note


def test_a_stale_pyc_cannot_stand_in(tmp_path, monkeypatch):
    """each toy run writes a pyc python trusts without reading the source: of the intact code during the
    control, of the broken code during the break. the break is same-size. only clearing __pycache__
    before the break makes it run (RED), and only clearing after the restore lets the closing control
    run the intact code (PASSED)"""
    repo = make_repo(tmp_path)
    monkeypatch.setenv('TOY_PLANT_PYC', '1')
    table = write_table(tmp_path, [{'id': 'same-size', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return 2 + x'}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'pyc']) == 0
    got = verdicts(repo, 'pyc')
    assert got['same-size'][0] == 'RED' and got['control-closing'][0] == 'PASSED'


def test_a_failed_restore_aborts(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setattr(sabotage, 'restore', lambda pristine, target: None)
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return 2 - x'}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'norestore']) == 2
    assert verdicts(repo, 'norestore')['caught'][0] == 'RESTORE-FAILED'
    assert (repo / '.scratch/sabotage/norestore/tree').exists()  # kept as evidence


# ---- by PID only

def test_stop_refuses_a_name_with_no_pid_file(tmp_path, capsys):
    assert sabotage.main(['--repo', str(tmp_path), 'stop', 'nothing-here']) == 2
    assert 'no pid file' in capsys.readouterr().err


def test_stop_refuses_a_reused_pid(tmp_path):
    """the recorded PID now names another process (its creation time differs): left alive"""
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        run = tmp_path / '.scratch/sabotage/reused'
        run.mkdir(parents=True)
        (run / 'pid').write_text(f'{other.pid} 12345\n', encoding='utf-8')
        assert sabotage.main(['--repo', str(tmp_path), 'stop', 'reused']) == 2
        time.sleep(0.5)
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_stop_clears_a_dead_runs_pid_file(tmp_path):
    gone = subprocess.Popen([sys.executable, '-c', 'pass'])
    token = sabotage.process_token(gone.pid)
    gone.wait()
    run = tmp_path / '.scratch/sabotage/dead'
    run.mkdir(parents=True)
    (run / 'pid').write_text(f'{gone.pid} {token}\n', encoding='utf-8')
    assert sabotage.main(['--repo', str(tmp_path), 'stop', 'dead']) == 0
    assert not (run / 'pid').exists()


def test_a_live_name_refuses_to_start(tmp_path):
    repo = make_repo(tmp_path)
    run = repo / '.scratch/sabotage/busy'
    run.mkdir(parents=True)
    sabotage.write_pid(run / 'pid', os.getpid())
    held = (run / 'pid').read_bytes()
    table = write_table(tmp_path, [{'id': 'caught', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return 2 - x'}])
    assert sabotage.main(['--repo', str(repo), 'run', str(table), '--name', 'busy']) == 2
    assert (run / 'pid').read_bytes() == held


def test_stop_kills_the_run_by_its_pid_and_nothing_else(tmp_path):
    """a run mid-break is stopped: the engine, its pytest child and that child's child die, the private
    copy (holding the applied break) is removed. a decoy whose command line names the run and the tool
    is untouched"""
    repo = make_repo(tmp_path)
    marker = tmp_path / 'grandchild.pid'
    table = write_table(tmp_path, [{'id': 'hangs', 'target': 'toypkg/core.py', 'old': 'return 2 * x',
                                    'new': 'return hang()', 'timeout': 120}])
    env = {**os.environ, 'TOY_GRANDCHILD': str(marker)}
    decoy = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', 'tools/sabotage.py', 'run',
                              '--name', 'victim'])
    engine = subprocess.Popen([sys.executable, str(ROOT / 'tools/sabotage.py'), '--repo', str(repo), 'run', str(table),
                               '--name', 'victim'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run = repo / '.scratch/sabotage/victim'
    grandchild = None
    try:
        end = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < end and engine.poll() is None:
            time.sleep(0.1)
        assert marker.exists() and (run / 'applied').exists(), 'the run never reached its break'
        time.sleep(0.2)
        grandchild = int(marker.read_text())
        child = sabotage.read_pid(run / 'child')[0]
        assert sabotage.main(['--repo', str(repo), 'stop', 'victim']) == 0
        assert engine.wait(timeout=15) != 0
        assert wait_gone(child) and wait_gone(grandchild)
        assert not (run / 'tree').exists() and not (run / 'pid').exists()
        assert decoy.poll() is None
    finally:
        for p in (engine, decoy):
            if p.poll() is None:
                p.kill()
                p.wait()
        if grandchild:
            sabotage.kill_tree(grandchild)
