# verify accessors-r2-17 (docs lens): accessors on a WIDE np.longdouble end

claim: UNCLEAR (not run; windows longdouble is float64). date 2026-10-04.

## records that cover THIS behaviour
* v2-plan.md:1116-1125 "a foreign real is its exact value ... any other real is the float it equals where it is a
  double ..., else its exact `as_integer_ratio()` as a `Fraction` or an int ... so an `np.longdouble` wider than a
  double (x86-64 linux, where CI runs: 64-bit significand) and a wide `mpfr` are exact"
* v2-plan.md:1789-1791 (M16d revision) "a foreign real is exact (Q15(b)) ... fixes the outward class for `np.longdouble` on linux"
* v2-plan.md:1661-1662 (2026-10-03 revision) "kept as built, now owner-confirmed: ... D23 (Q15, numpy) (a), (b), (d), (e)" -> owner-confirmed, not Q21
* README.md:196-197 "an `np.longdouble` wider than a double is exact, as any foreign real"
* intervals/cuts.py:17-22 module docstring (same rule, names "an `np.longdouble` wider than a double")
* references/owner-questions-2026-10-03/numpy.md:141-165 Q15(b) option 1 "(built)", "anyone on linux with `np.longdouble` data"
* pinned: tests/test_numpy_compat.py::test_longdouble_is_exact ("discriminates where the long double is wider than a double
  (every linux CI job)"); HANDOFF.md:152-153 records it discriminates on CI's linux only
* HANDOFF Q21 (M8 items a-i) does not touch this.

## run (stand-in, since no wide longdouble here)
probe: .scratch/v1-parity/verify/accessors-r2-17/probe_wide_ld_standin.py (a numbers.Real stand-in with exact
compare/+/-, float() rounding, exact as_integer_ratio, as x86-64's long double behaves for these ops)
`timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/accessors-r2-17/probe_wide_ld_standin.py`
-> `same=302 diff=0` (inf/sup/is_degenerate values exactly equal to the end on v1 and v2; 2 hand + 300 seeded with up to
2**-63 fractions); `--sabotage` -> `same=0 diff=302`. types: v1 hands back the stored object, v2 an int/Fraction/float.

## verdict
refuted: the behaviour is documented and owner-confirmed (Q15(b)); values agree exactly (stand-in run), types differ as
documented. corrected verdict: DIFFERS_DOCUMENTED (type of the returned end; values EQUIVALENT_RENAMED). a real linux
numpy run is still not done here, but CI's linux test_longdouble_is_exact pins v2's side.
