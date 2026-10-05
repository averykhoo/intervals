import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import multiinterval.errors as errors

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str) -> subprocess.CompletedProcess:
    """run `code` in a fresh interpreter, so the import-time filters are installed from scratch"""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONWARNINGS'}
    return subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_hierarchy():
    for cls in (errors.EmptySetPropagationWarning, errors.DomainClippedWarning,
                errors.IndeterminateResultWarning, errors.HullWarning):
        assert issubclass(cls, errors.IntervalWarning)
    assert issubclass(errors.IntervalWarning, UserWarning)


def test_suite_turns_library_warnings_into_errors():
    # pins pyproject's filterwarnings entry: without it, the suite would silently ignore these
    with pytest.raises(errors.EmptySetPropagationWarning):
        warnings.warn('x', errors.EmptySetPropagationWarning)


def test_default_filters():
    result = _run(
        'import warnings, multiinterval.errors as e\n'
        'warnings.warn("empty", e.EmptySetPropagationWarning)\n'
        'warnings.warn("clipped", e.DomainClippedWarning)\n'
        'warnings.warn("indeterminate", e.IndeterminateResultWarning)\n'
        'warnings.warn("hull", e.HullWarning)\n'
    )
    assert result.returncode == 0, result.stderr
    assert 'EmptySetPropagationWarning' not in result.stderr
    assert 'DomainClippedWarning' not in result.stderr
    assert 'IndeterminateResultWarning' in result.stderr
    assert 'HullWarning' in result.stderr  # a hull loses precision, so it is shown by default


def test_filter_installed_before_import_wins():
    # the default 'ignore' is appended, so a user's earlier catch-all still applies
    result = _run(
        'import warnings\n'
        'warnings.simplefilter("error")\n'
        'import multiinterval.errors as e\n'
        'warnings.warn("empty", e.EmptySetPropagationWarning)\n'
    )
    assert result.returncode != 0
    assert 'EmptySetPropagationWarning' in result.stderr


def test_tripwire_after_import():
    result = _run(
        'import warnings, multiinterval.errors as e\n'
        'warnings.simplefilter("error", e.EmptySetPropagationWarning)\n'
        'warnings.warn("empty", e.EmptySetPropagationWarning)\n'
    )
    assert result.returncode != 0

