"""
the fuzz profile (tests/conftest.py) keeps an example database under GitHub Actions, so a failure
one fuzz run finds is saved to .hypothesis, carried by fuzz.yml's cache, and replayed first by the
next run. hypothesis loads its `ci` profile (database None) at import when it detects CI, so this
runs in a fresh interpreter with the CI variables set, as the fuzz job does
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CI_VARIABLES = ('CI', 'GITHUB_ACTIONS', 'TF_BUILD', 'BUILDKITE', 'CIRCLECI', 'CIRRUS_CI', 'CODEBUILD_BUILD_ID',
                'GITLAB_CI', 'HEROKU_TEST_RUN_ID', 'TEAMCITY_VERSION', 'bamboo.buildKey')


def _database(profile, ci):
    env = {k: v for k, v in os.environ.items() if k not in CI_VARIABLES and k != 'HYPOTHESIS_PROFILE'}
    if profile is not None:
        env['HYPOTHESIS_PROFILE'] = profile
    if ci:
        env['GITHUB_ACTIONS'] = 'true'
    code = 'import tests.conftest\nfrom hypothesis import settings\nprint(settings.default.database)'
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize('ci', [False, True])
def test_the_fuzz_profile_keeps_a_database(ci):
    assert _database('fuzz', ci).startswith('DirectoryBasedExampleDatabase'), ci


def test_the_ci_profile_alone_has_none():
    """the control: without the fuzz profile, CI's own has no database (the gate's, derandomized)"""
    assert _database(None, True) == 'None'
