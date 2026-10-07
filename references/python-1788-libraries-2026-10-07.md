# other 1788 libraries on pypi (survey, 2026-10-07)

the owner asked: are there other ieee 1788 libraries on pypi, could we use them as a reference, and is
there anything to learn from them. `test-vector-sources.md` covers the non-python libraries (IntervalArithmetic.jl,
octave `interval`, inari, libieeep1788, mpfi, jinterval, kv) as sources of test data; this file is the python side.

method: about 50 candidate names queried on the pypi json api (`https://pypi.org/pypi/<name>/json`), web search
for "1788" on pypi, then the two that claim 1788 read first-hand: decoint's sdist, pyintval's repo
(`tests/itf1788/README.md`, `known_deviations.txt`, `src/pyintval/_core.pyi`, `tests/itf1788/pyintval_arith.yaml`,
`CHANGELOG.md`). pyintval was missed by the name sweep and named by the owner. neither was installed or run.

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

1. **pyintval as a differential check of the 1788 layer, on arbitrary doubles. worth a try, test-only.** it is
   independently written (C++, binary64, CORE-MATH) and decides decorations itself. on random binary64 operands,
   for the ops both have, the check is: our `ieee1788` result inside theirs; equal for `+ - * / sqrt fma`; and the
   decorations compared, a stronger decoration from them being a lead (they may be weaker: they allow it). what it
   adds to what we have: `tests/test_propagation.py`'s decoration oracle is a brute-force one on a quarter-integer
   grid (ends in [-3, 3]), so its operands are never arbitrary or large doubles: `tan` near a pole at a large
   double, say, is left to the itf1788 vectors (not checked further here). what it does not add: tightness (arb and gmpy2 already
   decide that, and pyintval is one ulp loose by design). it would be a new test dependency: the owner's call.
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
