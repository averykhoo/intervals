# the fuzz profile, for .github/workflows/fuzz.yml: HYPOTHESIS_PROFILE=fuzz runs every hypothesis
# test randomized, with no deadline and FUZZ_MULTIPLIER (default 100) times its usual examples.
# with HYPOTHESIS_PROFILE unset this file does nothing, so the gate keeps hypothesis's own choice:
# the `default` profile locally and the derandomized `ci` profile under GitHub Actions.
import os

from hypothesis import HealthCheck, settings

PROFILE = os.environ.get('HYPOTHESIS_PROFILE')
FUZZ = PROFILE == 'fuzz'

settings.register_profile(
    'fuzz',
    derandomize=False,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
if PROFILE:
    # any other name goes to hypothesis, which refuses one it does not know
    settings.load_profile(PROFILE)


def pytest_collection_modifyitems(items):
    # a test's own @settings(max_examples=N) overrides any profile's max_examples, and most tests
    # pin one, so the profile cannot raise the count. instead each hypothesis test's settings are
    # rewrapped here with max_examples times the multiplier: 100 -> 10000 for a test that pins
    # nothing, 60 -> 6000 for one that pins 60. a parametrized test is one function behind many
    # items, so each function is rewrapped once, not once per item
    if not FUZZ:
        return
    multiplier = int(os.environ.get('FUZZ_MULTIPLIER', '100'))
    seen = set()
    for item in items:
        test = getattr(getattr(item, 'obj', None), '__func__', getattr(item, 'obj', None))
        if not getattr(test, 'is_hypothesis_test', False) or id(test) in seen:
            continue
        seen.add(id(test))
        own = test._hypothesis_internal_use_settings
        test._hypothesis_internal_use_settings = settings(
            own, max_examples=own.max_examples * multiplier, derandomize=False, deadline=None)
