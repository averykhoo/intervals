"""
which code picks a rounded double: the pure path (`python`, the default) or gmpy2/mpfr (`gmpy2`)

the backend is an implementation detail with one public knob, the environment variable
`INTERVALS_BACKEND`, read once when `intervals` is imported:

* unset, `''` or `python`: the pure path; gmpy2 is never imported
* `gmpy2`: `intervals._gmpy2`, and an `ImportError` at import if gmpy2 is missing or below the floor
  (gmpy2 2.3 with MPFR 4.2): a job that asks for gmpy2 never falls back and passes on the pure path
* `auto`: gmpy2 if it imports and is at the floor and in the series verified (2.3 <= version < 3),
  else the pure path, silently
* anything else: `ValueError` at import

both give the same doubles and the same flags: the pure path is correctly rounded in every direction
already, and gmpy2 answers only "which double", after every exactness, flag and attainment decision
the pure path makes (`intervals._gmpy2`). the only difference is speed, and the failure mode of a
library bug (a missed exact case raises in every direction under gmpy2). `_use` switches the backend
for the tests, which compare the two in one process; it is a module global, not thread safe, and never
used by the library. the backend is untested on free-threaded builds.
"""
import os
import re
from contextlib import contextmanager
from typing import Optional
from typing import Tuple

VARIABLE = 'INTERVALS_BACKEND'
FLOOR = (2, 3, 0)  # gmpy2's; its context API, `ieee()` and `mpfr(x, precision, context)` are leaned on
CEILING = 3  # `auto` takes only gmpy2 2.x from 2.3: the series verified (2.3.1, 2026-09-28)
MPFR_FLOOR = (4, 2)

NAME = 'python'
fast = None  # the module `intervals._gmpy2` when NAME == 'gmpy2', else None


def name() -> str:
    """the backend in use: `'python'` or `'gmpy2'`"""
    return NAME


_RELEASE = re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?(.*)')
_FINAL = re.compile(r'(\.?post\d*)?(\+[0-9a-z.]*)?')
_PRE = re.compile(r'[-._]?(a|b|c|rc|alpha|beta|pre|preview|dev)\d*([-._]?dev\d*)?(\+[0-9a-z.]*)?')


def _release(version: str) -> Optional[Tuple[Tuple[int, int, int], bool]]:
    """`((major, minor, micro), final)` of a version string, or None if it is not one"""
    match = _RELEASE.fullmatch(version.strip().lower())
    if match is None:
        return None
    release = tuple(int(x) for x in match.group(1, 2)) + (int(match.group(3) or 0),)
    rest = match.group(4)
    if _FINAL.fullmatch(rest):
        return release, True
    if _PRE.fullmatch(rest):
        return release, False
    return None


def _supported(version: str, mpfr_version: str, ceiling: bool = True) -> bool:
    """
    whether gmpy2 `version` (`gmpy2.version()`) with `mpfr_version` (`gmpy2.mpfr_version()`, `'MPFR
    4.2.2'`) is at the floor, and with `ceiling` below gmpy2 3. a pre-release counts as just before its
    release (`2.3.0rc1` is below the floor, `2.3.1rc1` is not); a string that is not a version is not
    supported
    """
    parsed = _release(version)
    match = re.match(r'MPFR (\d+)\.(\d+)(?:\.\d+)?', mpfr_version.strip())
    if parsed is None or match is None:
        return False
    release, final = parsed
    if (release, final) < (FLOOR, True):
        return False
    if ceiling and release[0] >= CEILING:
        return False
    return tuple(int(x) for x in match.group(1, 2)) >= MPFR_FLOOR


def _load(forced: bool):
    """`intervals._gmpy2`, or None (auto only) where gmpy2 is missing or not supported"""
    try:
        import gmpy2
    except ImportError as e:
        if forced:
            raise ImportError(f'{VARIABLE}=gmpy2, but gmpy2 does not import: {e}') from e
        return None
    version, mpfr_version = str(gmpy2.version()), str(gmpy2.mpfr_version())
    if not _supported(version, mpfr_version, ceiling=not forced):
        if forced:
            raise ImportError(f'{VARIABLE}=gmpy2 needs gmpy2 >= {".".join(map(str, FLOOR[:2]))} with MPFR >= '
                              f'{".".join(map(str, MPFR_FLOOR))}; found gmpy2 {version} with {mpfr_version}')
        return None
    from intervals import _gmpy2
    return _gmpy2


def _select(value: Optional[str]):
    """`(NAME, fast)` for a value of `INTERVALS_BACKEND` (None: unset)"""
    if value in (None, '', 'python'):
        return 'python', None
    if value in ('gmpy2', 'auto'):
        module = _load(forced=value == 'gmpy2')
        return ('python', None) if module is None else ('gmpy2', module)
    raise ValueError(f"{VARIABLE}={value!r}: expected 'python', 'gmpy2' or 'auto' (or unset)")


@contextmanager
def _use(backend: str):
    """
    for the tests only: run the body under `backend` (`'python'` or `'gmpy2'`, forced), then restore.
    every dispatch reads `fast` at call time, so this reaches the descriptors built at import too
    """
    global NAME, fast
    saved = NAME, fast
    NAME, fast = _select(backend)
    try:
        yield
    finally:
        NAME, fast = saved


NAME, fast = _select(os.environ.get(VARIABLE))
