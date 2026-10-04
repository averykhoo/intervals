"""
the speed tables of M16e's record (`v2-implementation-plan.md` §2 M16e, "speed"): the pure path, then
gmpy2, back to back in one process under `backend._use`, best of 5 per call, as markdown rows
`| call | pure | gmpy2 | ratio |`.
ratios, not absolute times: a loaded machine moves both columns. with `--bound`, the cost of the
class-15 calls (`tests/test_backend.py::_bound_cases`) at a bound of 2**12, 2**16 and 2**20 bits
instead: the pure path at a tiny x grows about quadratically in the bits, which is why the real-bound
test keeps only the cheap calls. not collected by pytest (`tools/` is not in `testpaths`). from the
repo root, with gmpy2 installed:

    C:/Users/user/anaconda3/envs/intervals/python.exe tools/backend_speed.py [--bound]
"""
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

from intervals import MultiInterval as M  # noqa: E402
from intervals import OutwardMultiInterval as O  # noqa: E402
from intervals import _gmpy2  # noqa: E402
from intervals import backend  # noqa: E402
from intervals import elementary  # noqa: E402
from intervals import newton  # noqa: E402
from intervals import ops  # noqa: E402
from intervals.rounding import DOWN  # noqa: E402
from intervals.rounding import UP  # noqa: E402


def best(f, n):
    out = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            f()
        out.append((time.perf_counter() - t) / n)
    return min(out)


def row(label, f, n, unit=1e6):
    times = []
    for b in ('python', 'gmpy2'):
        with backend._use(b):
            f()
            times.append(best(f, n) * unit)
    print(f'| {label} | {times[0]:.3g} | {times[1]:.3g} | {times[0] / times[1]:.2g} |', flush=True)


def speed():
    print('| call | pure µs | gmpy2 µs | ratio |\n|---|---|---|---|')
    row("`rounded('exp', 0.7, DOWN)`", lambda: elementary.rounded('exp', Fraction(0.7), DOWN), 200)
    row("`rounded('exp', 1/3, DOWN)` (declined: pure in both)", lambda: elementary.rounded('exp', Fraction(1, 3), DOWN), 200)
    row("`rounded('log', 0.7, DOWN)`", lambda: elementary.rounded('log', Fraction(0.7), DOWN), 200)
    row("`rounded('sin', 0.7, DOWN)`", lambda: elementary.rounded('sin', Fraction(0.7), DOWN), 200)
    row("`rounded('sin', 1e22, DOWN)`", lambda: elementary.rounded('sin', Fraction(1e22), DOWN), 200)
    row("`rounded('atan', 1/3, DOWN)` (atan2 of the ints)", lambda: elementary.rounded('atan', Fraction(1, 3), DOWN), 200)
    row("`rounded('atan', 2**-30, DOWN)`", lambda: elementary.rounded('atan', Fraction(1, 2 ** 30), DOWN), 200)
    row("`rounded_pow(2, 1/2, UP)`", lambda: elementary.rounded_pow(Fraction(2), Fraction(1, 2), UP), 200)
    row("`rounded_angle(1/3, 1, UP)`", lambda: elementary.rounded_angle(Fraction(1, 3), 1, UP), 200)
    row("hook `add(0.1, 0.2)` down", lambda: ops.OUTWARD['add'].rounded[0](0.1, 0.2), 2000)
    row("hook `div(1.0, 3.0)` down", lambda: ops.OUTWARD['div'].rounded[0](1.0, 3.0), 2000)
    row("hook `add(0.1, 1/3)` down (the mpq route)", lambda: ops.OUTWARD['add'].rounded[0](0.1, Fraction(1, 3)), 2000)
    print('\n| op | pure ms | gmpy2 ms | ratio |\n|---|---|---|---|')
    a = O.from_pieces([(0.1, 0.7), (1.3, 2.9), (4.1, 5.5)])
    b = O.from_pieces([(0.3, 0.9), (2.2, 3.7)])
    row('`OutwardMultiInterval`, 3 float pieces, `.exp()`', a.exp, 30, 1e3)
    row('same, `.log()`', a.log, 30, 1e3)
    row('same, `.sin()` (`floor_over_pi` stays pure)', a.sin, 30, 1e3)
    row('same, `.atan()`', a.atan, 30, 1e3)
    row('`MultiInterval`, 2 float pieces, `.exp()`', M.from_pieces([(0.1, 0.7), (1.3, 2.9)]).exp, 30, 1e3)
    row('A + B (3 x 2 float pieces, outward)', lambda: a + b, 30, 1e3)
    row('A * B', lambda: a * b, 30, 1e3)
    row('A / B', lambda: a / b, 30, 1e3)
    row('`newton(t**2 - 2, Outward(-10.0, 10.0))`', lambda: newton(lambda t: t ** 2 - 2, O(-10.0, 10.0)), 3, 1e3)
    row('`newton(sin(t) - t/3, Outward(-10.0, 10.0))`', lambda: newton(lambda t: t.sin() - t / 3, O(-10.0, 10.0)), 3, 1e3)


def bound():
    from tests import test_backend as t
    saved = _gmpy2.BOUND
    try:
        for b, cheap in ((1 << 12, False), (1 << 16, False), (1 << 20, True)):
            _gmpy2.BOUND = b
            past, inside = t._bound_cases(b, cheap)
            start = time.perf_counter()
            for kind, args in past + inside:
                s = time.perf_counter()
                t.CHECKS[kind](*args)
                if time.perf_counter() - s > 0.05:
                    print(f'2**{b.bit_length() - 1} bits, {kind} {repr(args[0])[:30]}: {time.perf_counter() - s:.2f} s', flush=True)
            print(f'2**{b.bit_length() - 1} bits{" (cheap)" if cheap else ""}: {time.perf_counter() - start:.2f} s total',
                  flush=True)
    finally:
        _gmpy2.BOUND = saved


if __name__ == '__main__':
    bound() if '--bound' in sys.argv[1:] else speed()
