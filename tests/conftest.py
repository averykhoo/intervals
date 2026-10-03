# the fuzz profile, for .github/workflows/fuzz.yml: HYPOTHESIS_PROFILE=fuzz runs every hypothesis
# test randomized, with no deadline and FUZZ_MULTIPLIER (default 10) times its usual examples.
# with HYPOTHESIS_PROFILE unset this file does nothing, so the gate keeps hypothesis's own choice:
# the `default` profile locally and the derandomized `ci` profile under GitHub Actions.
import os

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase
from hypothesis.vendor import pretty


def _pretty_fraction(obj, printer, cycle):
    """
    hypothesis writes an example's arguments eagerly (an explicit example's too, pass or fail) with its
    own printer, which writes an int past python's 4300-digit limit in hex but a Fraction by `repr`,
    which raises for such a part: `@example(None, Fraction(-1, 10 ** 4300))` failed before running
    (tests/test_fmt.py, m14b-open, 2026-10-03). the parts go through hypothesis's int printer instead
    """
    printer.text('Fraction(')
    printer.pretty(obj.numerator)
    printer.text(', ')
    printer.pretty(obj.denominator)
    printer.text(')')


pretty.for_type_by_name('fractions', 'Fraction', _pretty_fraction)

PROFILE = os.environ.get('HYPOTHESIS_PROFILE')
FUZZ = PROFILE == 'fuzz'

# the database is named, not inherited: under GitHub Actions hypothesis loads its `ci` profile at
# import, whose database is None, and a profile registered without one inherits that, so the fuzz
# job saved nothing and its carried .hypothesis replayed nothing (found 2026-09-29).
# tests/test_fuzz_profile.py pins it
settings.register_profile(
    'fuzz',
    database=DirectoryBasedExampleDatabase(os.path.join('.hypothesis', 'examples')),
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
    # rewrapped here with max_examples times the multiplier, at x10 100 -> 1000 for a test that pins
    # nothing, 60 -> 600 for one that pins 60. a parametrized test is one function behind many
    # items, so each function is rewrapped once, not once per item
    if not FUZZ:
        return
    multiplier = int(os.environ.get('FUZZ_MULTIPLIER', '10'))
    seen = set()
    for item in items:
        test = getattr(getattr(item, 'obj', None), '__func__', getattr(item, 'obj', None))
        if not getattr(test, 'is_hypothesis_test', False) or id(test) in seen:
            continue
        seen.add(id(test))
        own = test._hypothesis_internal_use_settings
        test._hypothesis_internal_use_settings = settings(
            own, max_examples=own.max_examples * multiplier, derandomize=False, deadline=None)
