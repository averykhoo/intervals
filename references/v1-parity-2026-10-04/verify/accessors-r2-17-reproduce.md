# verify accessors r2-17: accessors on a WIDE np.longdouble end (reproduce lens), 2026-10-04

claim (r2-accessors/report.md row 29): UNCLEAR, not run, because windows numpy's longdouble is float64 and WSL has no numpy.

## result: REFUTED. the run was made, on a real 80-bit long double, and v1 and v2 agree on every accessor

how it was run: a throwaway linux x86-64 container (`python:3.13-slim`, pulled to WSL docker, `--rm`, repo mounted
read-only, `pip install numpy` inside; container gone afterwards, `wsl docker ps -a --filter name=v1parity` empty).
`np.finfo(np.longdouble)`: "Machine parameters for float128 ... precision = 18", `nmant 63`. python 3.13.16, numpy 2.5.3.

files (all in `.scratch/v1-parity/verify/accessors_r2_17/`):
* `probe_wide_longdouble.py` (own probe; `--sabotage` swaps the exact oracle for `Fraction(float(end))`)
* `run_linux.sh` (container entry: pip install numpy, run probe plain and `--sabotage`)
* `linux_wide.out` (the wide run), `windows_narrow.out` (the same probe on windows: float64 long double, 0 diffs)
* command (Git Bash): `MSYS_NO_PATHCONV=1 wsl bash -c 'docker run --rm --name v1parity-ld-r2-17 -v
  /mnt/c/Users/user/PycharmProjects/intervals:/repo:ro -e PYTHONDONTWRITEBYTECODE=1 python:3.13-slim bash
  /repo/.scratch/v1-parity/verify/accessors_r2_17/run_linux.sh'`

inputs: long doubles that are NOT doubles (checked `is_double=False`): 1+2^-60, -(1+2^-60), 1-2^-63, ld('0.1'), ld(1)/3,
2^64-1, 2^62+1/2, ld('1e4000'), ld('-1e4000'), ld('1e-4000'); 16 hand sets (touching pieces 2^-60 apart, a 2^-60 gap,
rays to +-inf, wide points) + 400 seeded random sets (seed 20261004, 1-3 pieces + optional point, ends from a grid of
wide long doubles and +-inf). compared EXACTLY (each end as `as_integer_ratio()` Fraction), plus v2 brute-force
membership at the ends, just below (np.nextafter), and midpoints.

wide run (linux_wide.out), every v1 accessor vs v2's spelling:
```
construct (exact pieces) same=414 diff=0     infimum/supremum same=414 diff=0   infimum_is_closed/supremum_is_closed same=414 diff=0
is_empty/is_contiguous/is_degenerate/is_finite/is_integral/is_positive/is_negative/is_non_negative/is_non_positive  same=414 diff=0 each
degenerate_points/finite/positive/negative/closed_hull/contiguous_intervals  same=414 diff=0 each
cardinality.rays+points same=414 diff=0      membership at end 20/0, just below 10/0, mid 6/0
cardinality.length vs exact (v2, no float end) same=155 diff=0
TOTAL diffs 0
v2 sup type/value: Fraction 1152921504606846977/1152921504606846976 == exact True ; float(end) = 1.0
v1 supremum type/value: longdouble np.longdouble('1.0000000000000000009')
v2 raw ld: 1+2^-60 in [1, 1+2^-60): ('ok', False) ; 1+2^-61 in it: ('ok', True)
```
sabotage (same container, `--sabotage`): `TOTAL diffs 2700` (e.g. `construct same=15 diff=399`, `infimum same=170 diff=241`),
so the probe can fail on a wide build. (on windows the sabotage is a no-op, as expected: every long double is a double.)

## by-products (v1 wrong, v2 right: V1_BUG_FIXED, not gaps)

* v1 refuses finite long doubles past the doubles: `v1.MultiInterval(ld('-1e4000'), ld('1e4000'))` raises
  `ValueError('-inf cannot be contained in Interval')`, `v1.MultiInterval(ld('1e4000'))` raises "the degenerate interval at
  infinity cannot exist", because `math.isinf(x)` converts to float (`float(ld('1e4000')) = inf`; `np.isinf -> False`).
  v2: builds both, `inf` equals the exact value (an int), `is_finite True`, `10**3999 in it` True, `[1e4000].is_degenerate` True.
* v1 cardinality length on wide ends: `same=25 diff=282` against exact. `[1, 1+2^-60)`: v1 `0` (its `float(start) <
  float(end)` guard), exact `2^-60`; `[ld(0.1), 1]`: v1 `ld 0.9` = 8301034833169298227/2^63, exact 132816557330708771635/2^67.
  v2 is exact on every set with no float end (155/155).

## side note (not a regression vs v1)

`Size.length` with a mixed float/Fraction set rounds through python's `Fraction - float -> float`: `[1.0, 1+2^-60)` built from
long doubles (1.0 is a double so it stays a float; the other end a Fraction) gives `v2 size.length = 0.0`, exact 2^-60.
v1 also says 0 there, so no regression; `Size.length` is typed `Union[int, Fraction, float]` (kernel.py::Size) and the same
happens with any `[1.0, Fraction(...)]`. not in this claim's slice; flag for the arithmetic/size slice if anyone cares.

## verdict

refuted: TRUE. corrected verdict: EQUIVALENT_RENAMED (values: `A.inf`/`A.sup`/`A.inf_closed`/`A.sup_closed`/`A.pieces`/
`A.size`, the rest same names; 0 diffs in 414 sets on an 80-bit long double) with the return TYPE difference (v1 hands back
the `np.longdouble`, v2 an exact `Fraction`/int) already DIFFERS_DOCUMENTED (v2-plan.md:1117-1124 "an `np.longdouble` wider than
a double (x86-64 linux ...) ... [is] exact"), and two V1_BUG_FIXED by-products above.
