"""
the run ledger (tools/gate.py): a row names the code it ran on, and a verdict on other code does
not count here. the ids are checked on throwaway git repos (they survive a commit, move with any
source byte, ignore prose, refuse to guess), the recorder with a fake pytest, and the ledger's
assumptions about this repo against the files they copy: pyproject's doctest glob, fuzz.yml's
paths-ignore and multiplier, the ignore entry, and no source naming a prose path
"""
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('gate', ROOT / 'tools' / 'gate.py')
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                   check=True, capture_output=True)


def _write(repo, rel, data):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, 'init', '-q', '-b', 'master')
    _git(tmp_path, 'config', 'core.autocrlf', 'false')
    for rel, data in {'.gitignore': b'.gate-runs/\nignored/\n', 'pkg/a.py': b'x = 1\n', 'README.md': b'# r\n',
                      'tests/README.md': b'# t\n', 'notes.md': b'notes\n', 'references/r.py': b'r = 1\n',
                      'pyproject.toml': b'[tool.pytest.ini_options]\ntestpaths = ["tests", "README.md"]\n'}.items():
        _write(tmp_path, rel, data)
    _git(tmp_path, 'add', '-A')
    _git(tmp_path, 'commit', '-q', '-m', 'base')
    _git(tmp_path, 'branch', 'base')
    return tmp_path


@pytest.mark.parametrize('path, kind', [
    ('intervals/ops.py', 'src'), ('pyproject.toml', 'src'), ('.github/workflows/fuzz.yml', 'src'),
    ('tests/itf1788/libieeep1788_elem.itl', 'src'), ('README.md', 'readme'), ('tests/itf1788/README.md', 'readme'),
    ('archive/v1/README.md', 'readme'), ('HANDOFF.md', 'prose'), ('.claude/skills/testing/SKILL.md', 'prose'),
    ('references/modulo-derivations/claude-fable/modulo_v3_prototype.py', 'prose'), ('tests\\x.md', 'prose'),
])
def test_classify(path, kind):
    assert gate.classify(path) == kind


def test_ids_survive_a_commit(repo):
    _write(repo, 'pkg/b.py', b'y = 2\n')
    before = gate.tree_ids(repo)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'b')
    assert gate.tree_ids(repo) == before


def test_ids_survive_a_commit_of_a_deletion(repo):
    """zanzibar's 2026-09-05b hole: a committed deletion moved the id though no byte changed"""
    (repo / 'pkg/a.py').unlink()
    before = gate.tree_ids(repo)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'rm')
    assert gate.tree_ids(repo) == before


@pytest.mark.parametrize('rel', ['pkg/a.py', 'pyproject.toml', 'pkg/new.py'])
def test_a_source_byte_moves_both(repo, rel):
    before = gate.tree_ids(repo)
    _write(repo, rel, (repo / rel).read_bytes() + b'#\n' if (repo / rel).exists() else b'z = 3\n')
    after = gate.tree_ids(repo)
    assert after['code'] != before['code'] and after['src'] != before['src']


def test_deleting_a_source_moves_both(repo):
    before = gate.tree_ids(repo)
    (repo / 'pkg/a.py').unlink()
    after = gate.tree_ids(repo)
    assert after['code'] != before['code'] and after['src'] != before['src']


@pytest.mark.parametrize('rel', ['notes.md', 'references/r.py', 'docs/new.md', 'ignored/x.py', '.gate-runs/l.tsv'])
def test_prose_and_ignored_files_move_neither(repo, rel):
    before = gate.tree_ids(repo)
    _write(repo, rel, b'changed\n')
    assert gate.tree_ids(repo) == before


@pytest.mark.parametrize('rel', ['README.md', 'tests/README.md'])
def test_a_readme_moves_code_only(repo, rel):
    before = gate.tree_ids(repo)
    _write(repo, rel, b'# changed\n')
    after = gate.tree_ids(repo)
    assert after['code'] != before['code'] and after['src'] == before['src']


def test_ids_refuse_to_guess(tmp_path):
    with pytest.raises(gate.TreeIdError):
        gate.tree_ids(tmp_path)  # not a git repo


@pytest.mark.parametrize('git_dir', [None, 'no-such-dir'])
def test_the_cli_prints_an_id_or_nothing(tmp_path, git_dir):
    """the script finds its own repo from any cwd; when git cannot read it, no id and a nonzero exit"""
    env = {k: v for k, v in os.environ.items() if k != 'GIT_DIR'}
    if git_dir:
        env['GIT_DIR'] = str(tmp_path / git_dir)
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'gate.py'), 'tree-id'], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    if git_dir:
        assert (r.returncode, r.stdout) == (2, ''), r
    else:
        assert r.returncode == 0 and re.fullmatch(r'c:[0-9a-f]{12} s:[0-9a-f]{12}\n', r.stdout), r


def _fake(rc, out='3 passed in 0.10s', touch=None):
    """a stand-in for pytest: prints `out`, optionally edits a file, exits rc"""
    code = f'print({out!r})\n'
    if touch:
        code += f'open({touch!r}, "a").write("#\\n")\n'
    return [sys.executable, '-c', code + f'raise SystemExit({rc})']


def _rows(repo):
    return gate.read_ledger(gate.runs_dir(repo) / gate.LEDGER)


@pytest.mark.parametrize('rc, out, status', [
    (0, '3 passed in 0.10s', 'PASSED'),
    (0, '1 failed, 2 passed in 0.10s', 'INCONSISTENT'),
    (0, 'no summary at all', 'INCONSISTENT'),
    (1, '1 failed, 2 passed in 0.10s', 'FAILED'),
    (5, 'no tests ran in 0.01s', 'FAILED'),
])
def test_a_run_is_recorded_with_its_verdict(repo, rc, out, status):
    ids = gate.tree_ids(repo)
    assert gate.run_phase('gate:itf', repo, command=_fake(rc, out)) == (0 if status == 'PASSED' else 1)
    [row] = _rows(repo)
    assert (row['phase'], row['status'], row['code'], row['src']) == ('gate:itf', status, ids['code'], ids['src'])
    assert (gate.runs_dir(repo) / row['log']).read_text(encoding='utf-8').splitlines()[-1] == out


def test_the_command_line_is_not_the_verdict(repo):
    """the log's header echoes the command; a summary in the command's text is not the run's"""
    command = [sys.executable, '-c', 'print("nothing")  # 3 passed in 0.1s']
    assert gate.run_phase('gate:itf', repo, command=command) == 1
    assert _rows(repo)[-1]['status'] == 'INCONSISTENT'


def test_a_run_whose_code_moved_counts_for_nothing(repo):
    assert gate.run_phase('gate:itf', repo, command=_fake(0, touch=str(repo / 'pkg/a.py'))) == 1
    assert _rows(repo)[-1]['status'] == 'MOVED'
    assert gate.run_phase('gate:itf', repo, command=_fake(0, touch=str(repo / 'notes.md'))) == 0  # prose: not a move


def test_a_phase_gets_its_environment(repo, monkeypatch):
    monkeypatch.setenv('INTERVALS_BACKEND', 'gmpy2')
    monkeypatch.setenv('HYPOTHESIS_PROFILE', 'fuzz')
    check = ('import os, sys\n'
             'want = {"INTERVALS_BACKEND": None, "HYPOTHESIS_PROFILE": %r, "FUZZ_MULTIPLIER": %r}\n'
             'got = {k: os.environ.get(k) for k in want}\n'
             'print("3 passed in 0.1s" if got == want else got)\n')
    assert gate.run_phase('gate:rest', repo, command=[sys.executable, '-c', check % (None, None)]) == 0
    assert gate.run_phase('fuzz-x7:rest', repo, command=[sys.executable, '-c', check % ('fuzz', '7')]) == 0


def test_phase_names_are_closed():
    assert gate.phase_spec('gate:itf')[0] == ['tests/itf1788']
    assert gate.phase_spec('fuzz-x10:rest') == (['--ignore=tests/itf1788'], {'HYPOTHESIS_PROFILE': 'fuzz', 'FUZZ_MULTIPLIER': '10'})
    for bad in ('gate', 'gate:all', 'fuzz:itf', 'fuzz-x0:itf', 'fuzz-x10', 'lint'):
        with pytest.raises(ValueError):
            gate.phase_spec(bad)


def test_the_docs_phase_runs_the_collected_readmes(repo):
    assert gate.collected_readmes(repo) == ['README.md', 'tests/README.md']


def _row(phase, status, ids):
    return {'started': '2026-10-01T00:00:00+0800', 'dur_s': '1', 'phase': phase, 'status': status,
            'code': ids['code'], 'src': ids['src'], 'facts': 'rc=0', 'log': 'x.log'}


def test_a_verdict_is_keyed_by_phase_and_code():
    here, there = {'code': 'c:1', 'src': 's:1'}, {'code': 'c:2', 'src': 's:2'}
    rows = [_row('gate:itf', 'PASSED', here), _row('gate:itf', 'PASSED', there), _row('gate:rest', 'PASSED', here)]
    assert gate.gate_covered(rows, here) == {'itf': ['gate:itf'], 'rest': ['gate:rest']}, 'erased by a run elsewhere'
    rows.append(_row('gate:rest', 'FAILED', here))
    assert gate.gate_covered(rows, here) == {'itf': ['gate:itf'], 'rest': []}, 'a red rerun on this code is red'
    rows.append(_row('fuzz-x1:rest', 'PASSED', here))
    assert gate.gate_covered(rows, here)['rest'] == ['fuzz-x1:rest'], 'a fuzz run runs every test'
    assert gate.gate_covered(rows, {'code': gate.UNKNOWN, 'src': gate.UNKNOWN}) == {'itf': [], 'rest': []}


def test_the_push_plan():
    ids = {'code': 'c:1', 'src': 's:1'}
    fuzzed = [_row(f'fuzz-x{gate.PUSH_MULTIPLIER}:{part}', 'PASSED', ids) for part in gate.PARTS]
    plan = lambda rows, changed, dirty=(): gate.plan(rows, ids, changed, list(dirty))[0]
    assert plan([], ['intervals/ops.py']) == 'fuzz'
    assert plan([], None) == 'fuzz', 'no base: everything counts as changed'
    assert plan(fuzzed, ['intervals/ops.py']) == 'nothing'
    assert plan(fuzzed[:1], ['intervals/ops.py']) == 'fuzz', 'both parts'
    weak = [_row(f'fuzz-x{gate.PUSH_MULTIPLIER - 1}:{part}', 'PASSED', ids) for part in gate.PARTS]
    assert plan(weak, ['intervals/ops.py']) == 'fuzz', 'below the CI multiplier'
    # a README edited after the fuzz run: same src, new code
    readme_later = [_row(r['phase'], 'PASSED', {'code': 'c:0', 'src': 's:1'}) for r in fuzzed]
    assert plan(readme_later, ['intervals/ops.py', 'README.md']) == 'docs'
    assert plan(readme_later + [_row('docs', 'PASSED', ids)], ['intervals/ops.py', 'README.md']) == 'nothing'
    assert plan([], ['README.md', 'HANDOFF.md']) == 'docs'
    assert plan([_row('gate:itf', 'PASSED', ids), _row('gate:rest', 'PASSED', ids)], ['README.md']) == 'nothing'
    assert plan([], ['HANDOFF.md', 'references/x.md']) == 'nothing'
    assert plan([], []) == 'nothing'
    assert plan(fuzzed, ['intervals/ops.py'], dirty=['tools/x.py']) == 'dirty'


def test_dirty_paths_ignore_prose(repo):
    _write(repo, 'notes.md', b'edited\n')
    _write(repo, 'references/new.txt', b'new\n')
    assert gate.dirty_paths(repo) == []
    _write(repo, 'pkg/untracked.py', b'u = 1\n')
    assert gate.dirty_paths(repo) == ['pkg/untracked.py']


def test_the_plan_end_to_end(repo):
    """the CLI pieces on a real repo: changed_since, the recorder's rows, plan"""
    _write(repo, 'pkg/a.py', b'x = 2\n')
    _git(repo, 'commit', '-qam', 'src')
    ids = gate.tree_ids(repo)
    word = lambda: gate.plan(_rows(repo), gate.tree_ids(repo), gate.changed_since('base', repo), gate.dirty_paths(repo))[0]
    assert gate.changed_since('base', repo) == ['pkg/a.py']
    assert word() == 'fuzz'
    for part in gate.PARTS:
        gate.run_phase(f'fuzz-x{gate.PUSH_MULTIPLIER}:{part}', repo, command=_fake(0))
    assert word() == 'nothing'
    _write(repo, 'README.md', b'# edited\n')
    _git(repo, 'commit', '-qam', 'readme')
    assert gate.tree_ids(repo)['src'] == ids['src']
    assert word() == 'docs'
    gate.run_phase('docs', repo, command=_fake(0))
    assert word() == 'nothing'
    assert gate.changed_since('no-such-ref', repo) is None


# ---- the ledger's assumptions about this repo, checked against the files they copy

def test_the_code_scope_keeps_what_pytest_collects_as_doctests():
    """the code scope keeps README.md only; a new doctest glob would collect a file the ids ignore"""
    opts = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert re.findall(r'--doctest-glob=(\S+)', opts['addopts']) == ['README.md']
    assert all(gate.classify(p) != 'prose' for p in opts['testpaths'] + opts['pythonpath'])


def test_the_src_scope_is_what_fuzz_yml_skips():
    """a push the fuzz job skips is one the ledger lets through without a fuzz run, and back"""
    text = (ROOT / '.github' / 'workflows' / 'fuzz.yml').read_text(encoding='utf-8')
    assert re.search(r'paths-ignore: \["\*\*\.md", "references/\*\*"\]', text)
    assert gate.SRC_IGNORED == ('**.md', 'references/**')
    assert re.search(r"FUZZ_MULTIPLIER: \$\{\{ inputs\.multiplier \|\| '(\d+)' \}\}", text).group(1) == str(gate.PUSH_MULTIPLIER)


def test_the_ledger_is_ignored():
    r = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', '.gate-runs/ledger.tsv'])
    assert r.returncode == 0, '.gate-runs/ must be gitignored: an un-ignored ledger is inside its own hash'


def _strings(tree):
    docstrings = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            yield n


def test_no_source_names_a_prose_path():
    """the survey behind the scopes (2026-10-01), re-run: no code-scope python names a markdown file or
    references/ in a string, so none reads one. a hit means a test may read a file the ids ignore,
    and its verdict could go stale unseen: widen `gate.classify` or name the file another way"""
    files = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '*.py'],
                           capture_output=True, text=True, check=True).stdout.split()
    # tools/coremath.py reads references/coremath-runs.tsv for its `status` only; no test reads it, so no
    # recorded verdict rests on it
    exempt = {'tools/gate.py', 'tests/test_gate_ledger.py', 'tools/coremath.py'}
    hits = []
    for rel in files:
        if gate.classify(rel) == 'prose' or rel in exempt:
            continue
        source = (ROOT / rel).read_text(encoding='utf-8')
        for node in _strings(ast.parse(source)):
            if re.search(r'\.md\b', node.value) or node.value.startswith('references'):
                hits.append(f'{rel}:{node.lineno}: {node.value[:60]!r}')
    assert not hits, hits
