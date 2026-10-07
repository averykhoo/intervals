# other 1788 libraries on pypi (survey, 2026-10-07)

the owner asked: are there other ieee 1788 libraries on pypi, could we use them as a reference, and is
there anything to learn from them. `test-vector-sources.md` covers the non-python libraries (IntervalArithmetic.jl,
octave `interval`, inari, libieeep1788, mpfi, jinterval, kv) as sources of test data; this file is the python side.

method: about 50 candidate names queried on the pypi json api (`https://pypi.org/pypi/<name>/json`), web search
for "1788" on pypi, then the two that claim 1788 read first-hand: decoint's sdist, pyintval's repo
(`tests/itf1788/README.md`, `known_deviations.txt`, `src/pyintval/_core.pyi`, `tests/itf1788/pyintval_arith.yaml`,
`CHANGELOG.md`). pyintval was missed by the name sweep and named by the owner. neither was installed or run for
the survey; pyintval was installed afterwards for the cross-check (last section).

## the two that claim 1788

**pyintval 0.3.0** (https://github.com/marciogameiro/pyintval, MIT, Marcio Gameiro; uploads 2026-08-10 to
2026-08-12; repo created 2026-08-09, 0 stars; classifier Beta)
- a header-only C++20 kernel with pybind11 bindings; wheels for windows, macos, linux, cpython 3.10 to 3.14. the
  kernel is also used from C++ by CMGDB.
- 1788-2015 set-based over binary64. `+ - * / sqrt fma` correctly rounded by error-free transformations (no
  rounding-mode switching). elementary functions on vendored CORE-MATH kernels, **widened one ulp per end on
  purpose**. has `erf`, `erfc`, `expm1`, `log1p`, `cbrt`, `hypot`, `pown`, `pow`, `atan2`.
- decorated intervals, NaI, `cancel_minus`/`cancel_plus`, `sqr_rev`, `abs_rev`, `mul_rev` (binary and ternary).
  never signals: its itf1788 plugin leaves the exception predicates empty.
- **not implemented** (its own README, "Excluded (7 files)", and the plugin's mapped ops): the 13-state `overlap`,
  `mulRevToPair`, the other reverse ops (`pownRev`, `sinRev`, `cosRev`, `tanRev`, `coshRev`, `powRev1/2`,
  `atan2Rev1/2`), `sum`/`dot`, `rootn`. so it is not the full set-based flavour of 1788-2015; against 1788.1-2017,
  whose annex drops the reverse ops, `mulRevToPair` and `overlap`, it looks complete but for the signals.
- itf1788 at `b6ee1e2`, its own report: 7,236/7,236 enclose the vector, 5,748/7,236 (79%) are the tightest. the
  denominator is the ops it maps: itf1788 skips a test whose op the plugin leaves unset. CI gates on enclosure only.

**decoint 1.0.1** (https://github.com/arjavsharma91/decoint-IEEE-1788.1-2017, MIT, Arjav Sharma; uploaded
2026-09-06; repo created 2026-06-04, 6 stars)
- pure python, about 2,700 lines, endpoints rounded through gmpy2 (mpfr) with directed rounding contexts.
- claims full 1788.1-2017 conformance and ~5,000 itf1788-derived tests. **it has no `cancelMinus`/`cancelPlus`**
  (no `cancel` anywhere in the sdist, none in its `documentation/CONFORMANCE.md` checklist), which 1788.1 requires
  (`ieee-standard-for-interval-arithmetic-simplified.pdf`: "An implementation shall provide a T-version of each of
  the operations cancelMinus and cancelPlus"). no reverse ops, `mulRevToPair` or 13-state `overlap` either (1788.1
  does not need them). has NaI and decorations.

## the rest: interval arithmetic, not 1788

| package | latest upload | what |
|---|---|---|
| `intvalpy` 2.0.3 | 2026-03-08 | "developed taking into account" 1788-2015; Kaucher arithmetic and interval linear systems; no decorations |
| `pyinterval` 1.2.0 | 2017-03-05 | unions of intervals, outward rounding through CRlibm; dormant |
| `python-flint` 0.9.0 | 2026-07-03 | arb balls; already our oracle for the functions (D14) |
| `mpmath` 1.4.1 | 2026-08-21 | `mpmath.iv`, arbitrary precision; no decorations |
| `pyibex` 1.9.2, `codac` 2.1.1 | 2020-11-21, 2026-08-27 | ibex/codac bindings, constraint programming |
| `intervalarithmetic` 0.2.0, `interval-py` 1.0.5, `KaucherPy` 0.1.0 | 2017, 2025-09-01, 2019 | small; no 1788 claim |

ranges or sets with no arithmetic: `portion`, `python-intervals`, `intervals`, `pyintervals`, `intervalpy`,
`interval`, `pyinter`. unregistered: `ieee1788`, `libieeep1788`, `ieee-1788`, `pyieee1788`, `interval1788`.

## what we could use or learn

1. **pyintval as a differential check of the 1788 layer, on arbitrary doubles. built: `tools/pyintval_check.py`,
   local only (owner, 2026-10-07); results in the last section.** it is independently written (C++, binary64,
   CORE-MATH) and decides decorations itself. what it adds: `tests/test_propagation.py`'s decoration oracle is a
   brute-force one on a quarter-integer grid (ends in [-3, 3]), so its operands are never arbitrary or large
   doubles. what it does not add: tightness, which pyintval gives up by design (the tool's own MPFR referee
   decides it instead).
2. **decoint: nothing to take.** mpfr-rounded functions are what our gmpy2 and arb oracles already give, and its
   coverage is smaller.
3. **the literal edge cases pyintval fixed (its F4) we already answer the same** (probe 2026-10-07, at `88b5540`,
   `ieee1788.text_to_decorated_interval`): `[1.0E+400]_com` is `[max, inf]_dac`; `0.0??` is entire, `_dac`;
   `2.5??u` is `[2.5, inf]`; `0.0??_com` raises (com needs bounded); `[2/3, 1]` and `[-4/2, 10/5]` parse as
   rationals. a bare `2/3` raises here and parses in pyintval: 1788 has no bare-number literal, so theirs is an
   extension.
4. **functions they have that we do not**: `erf`, `erfc` (pyintval). not in 1788's required set; a feature
   question only.
5. **NaI**: both libraries have it and never raise; pyintval's reason is that bulk computation should not abort
   mid-sweep. that is the use D16 deferred ("a per-element 'invalid' in batch work ... come with numpy or data
   import, if ever"). nothing here changes D16; it is evidence for the numpy case if that comes up.

## the cross-check: `tools/pyintval_check.py` (built 2026-10-07)

the owner chose to try pyintval as a local benchmark: `pip install pyintval==0.3.0` into the `intervals` env
(the cp313 windows wheel), in no extra and not in CI. how to run it: the testing skill. the tool feeds the
same random binary64 operands to the `ieee1788` layer and to pyintval, 74 ops (the 54 with a set result also
decorated), and classifies each pair. pyintval is loose by design, so a difference is first put to the
tool's referee, which uses neither library: MPFR values at the box's points (ends, infinite ends as limits,
0, domain edges, sin's and cos's extrema with pi to 2300 bits), cancelMinus/cancelPlus exactly by 1788.1
4.5.3, an empty mulRev exactly, and 1788's decoration where it is decidable.

**result, 2026-10-07 at `b77c2e8` (the library unchanged since `88b5540`): seeds 1, 2, 3 at n = 2000, 768,000
comparisons. no `UNSOUND`, no `NOT 1788's`, no `theirs tighter`, `crossed`, `differ` or raise.** ours was
tighter in 137,254, every one shown to be the tightest answer by the referee (134,095 within two ulps,
pyintval's widening; 3,159 by more); 4,783 decorations differed, every one 1788's on our side (4,763 ours
stronger, 20 theirs stronger). one lead was left, decided by hand below. time per seed: ours 83-92 s,
pyintval 1.3-1.6 s.

**nothing found in the library.** what differs is pyintval's, in these families (each with the evidence that
our answer is 1788's):

| family | example (operands: ours; theirs) | evidence |
|---|---|---|
| elementary functions one ulp wide | `exp [1]`: `[2.718281828459045, 2.7182818284590455]`; `[2.7182818284590446, ...]` | by design (its README); MPFR at the points |
| pow, pown up to ~20 ulps and more wide | `pow [-9.75, 3.14e-17] [-2.5e11, -0.375]`: lo `1543614.8772613327`; `...3266` | MPFR at 300 bits: ours is the tightest double, nine cases checked by hand, then the referee |
| tan, sin, cos at large arguments or near a pole: `entire` / `[-1, 1]` | `tan [4.712388980384691, 4.712406956722049]_com`: `[-1.419e15, -55628.68]_com`; `[entire]_trv` | lo is 7.04e-16 above 3pi/2 (2300-bit pi), so no pole; ours the tightest |
| atan2 on a box touching y = 0 with x < 0, or holding the origin: `[-pi, pi]`, `def` | `atan2 [0, 19]_com [-2.16e12]_com`: `[3.14159265358101, pi]_dac`; `[-pi, pi]_def` | `libieeep1788_elem.itl:3830` `atan2 [0.0, 0.0]_com [-2.0, -0.1]_dac = [pi]_dac` |
| step functions `dac` where 1788 keeps `com` | `floor [-15.5, -15.49999999278225]_com`: `[-16]_com`; `[-16]_dac` | `libieeep1788_elem.itl:4139` `sign [1.0,2.0]_com = [1.0,1.0]_com`, `:4203` `floor [-1.2,-1.1]_com = [-2.0,-2.0]_com` |
| pow on a base below 0: pyintval computes `x^n` for an integral `y` and claims `def` to `com` | `pow [-1.375]_com [13]_def`: `[empty]_trv`; `[-62.796447571474346]_def` | 1788's pow is defined for x > 0, and x = 0 with y > 0: `libieeep1788_elem.itl:2852` `pow [-1.0,0.0] [0.0,1.0] = [0.0,0.0]` |
| pow on a base from 0 with y > 0: `trv` | `pow [0, 0.5]_dac [1.33e-12]_com`: `_dac`; `_trv` | `libieeep1788_elem.itl:3033` `pow [0.0,0.5]_com [0.1,0.1]_com = [...]_com` |
| ternary mulRev where no x solves `b * x = c`: `[0, 0]` or `[-max, -max]` | `mulRev [-0.375, -2.2e-308] [8.98846567431158e307, 8.98853425086244e307] [-max, -0.00275]`: `[empty]`; `[-max, -max]` | `abs(x) = c / abs(b) >= 2.4e308`, past the largest double (exact in the referee) |
| cancelPlus near overflow: `entire` | `cancelPlus [-8.988465674311582e307, -1] [8.98846567431158e307, 8.988465674311707e307]`: `[1.2573793950068735e294, 8.98846567431158e307]`; `[entire]` | widths allow the cancellation (1788.1 4.5.3, exact in the referee) |

**the family decided by hand** (the referee does not compute a non-empty mulRev): `mulRev [-inf, 4.71238898038469]
[-8.75, -2.4940488365719526e-13] [-0.7853981633974483, 0]` is `[-0.7853981633974483, -5.292536008706892e-14]`
here and `[-0.785..., 0]` in pyintval. `x = 0` needs `c = 0`, not in C; `x < 0` needs `b = c / x > 0`, so `b <=
4.71` gives `|x| >= 2.494e-13 / 4.712 = 5.2925e-14`. ours. a run that shows only this family's leads is clean.

**sabotage** (2026-10-07, the tool imported and one op of ours replaced, seed 0 or 1, n = 300 or 2000):

| break | the tool |
|---|---|
| exp's lower end one ulp up (unsound, still inside pyintval's widening) | red: 271 `UNSOUND` bare, 266 decorated; pyintval alone saw only `ours tighter, <=2 ulps` |
| sin's top end below 1 when an interior maximum is reached | red: 973 `UNSOUND` |
| add's upper end one ulp wide | red: 224 `theirs tighter` |
| sqrt decorated `com` always | red: 148 `dec ours stronger` |
| floor `dac` where com | red: 212 `dec NOT 1788's` (pyintval agrees with the break: `dec equal` all 2000) |
| atan2 `com` on every bounded result | red: 1623 decorations differ, 8 `dec NOT 1788's` |
| pow keeps `def` on a base below 0 | red: 403 `dec NOT 1788's` |
| cancelMinus `entire` where 1788 answers | red: 481 `set NOT 1788's exact answer` |
| mulRevTen empty where an x solves | red: 363 `UNSOUND` |

**the referee's own bugs, found and fixed while building it** (so the next change to it knows where it broke):
a limit at an open domain edge was taken even when the box does not reach into the domain there (`pow
[-1, 0] [-4.75, inf]`: `pow(+0, y < 0) = +inf` counted although no x > 0 is in the box), 24 false `UNSOUND`;
fixed by `OPEN_EDGES`. tan's pole test first used 300 bits of pi, too few past about 2^240; now 2300.
