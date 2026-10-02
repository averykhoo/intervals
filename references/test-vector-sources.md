# test-vector sources beyond ITF1788 (survey, 2026-09-29)

a research agent's survey, kept as it reported (the owner asked: is the 1788 set complete, and what community
test data could be pulled in). every count and licence below was fetched on 2026-09-29; UNVERIFIED marks what
was not. the session checked the claims about this tree first-hand: both errata lines are in the vendored files
(`libieeep1788_num.itl:168`, `mpfi.itl:603`) and `tests/itf1788/itl.py::parse_file` collapses whitespace
inside quoted strings. one correction: §2b calls that collapse harmless "(text constructors not run)"; they do
run (M13g), so the 40 whitespace vectors are live and weaker than upstream (`HANDOFF.md` row vectors-ext).

---


Findings appended as verified. UNVERIFIED marks anything not confirmed against a fetched source.

## 1. ITF1788 head, branches, PRs, forks (GitHub API, 2026-09-29)

- oheim/ITF1788 master head = b6ee1e24d209 (2018-09-22). Last push 2018-09-27 (PR #15). Not archived. VERIFIED.
- Branches: master, julia. `julia` is behind master by 25, ahead 0 -> adds nothing.
- Open PR #15 "Update atan2 -> atan": touches only itf1788/plugins/julia/.../arith.yaml (1 line). No .itl.
- Open issue #16 (2021-05-28, lucaferranti): `midRad [nai] [nai] = NaN NaN;` in libieeep1788_num.itl (line ~168) looks like a typo (midRad takes one arg). Unanswered. -> check how our parser/adapter treats it.
- Open issue #17 (2022-05-08, Nathalie Revol) asking about MPFI revamp tests / 1788 compliance. No vectors.
- Open issue #14: request to update Julia tests; no vectors.
- Parent chain: oheim <- kiesner/ITF1788 (last push 2015-04-23) <- nehmeier/ITF1788 (master e0e0d7e 2015-02-16; develop also 2015-02-16). nehmeier master is 133 behind / 0 ahead of oheim master; develop 134 behind / 0 ahead.
- Forks of oheim (7): unageek (branches build, inari: 21/25 ahead, all in itf1788/plugins/rust + discovery.py/__main__.py - an inari (Rust) output plugin, NO itl changes), krish8484 (12 ahead, Julia output/ValidatedNumerics files, 0 itl/ files), dpsanders, miguelraz, lucaferranti, yashrajgupta, ignacio-vc (all behind/identical, 0 ahead in itl).
- Fork of nehmeier: smithjlrhit identical to nehmeier master.
- CONCLUSION: no fork or branch on GitHub adds .itl vectors beyond the 19 files. Our snapshot is the full ITF1788 set.

## 2a. IntervalArithmetic.jl (JuliaIntervals, MIT; last push 2026-09-10) — VERIFIED 2026-09-29

- test/itl/ vendors 14 of our 19 .itl files (no abs_rev, pow_rev, libieeep1788_mul_rev, _rev, _reduction). Blob hashes: 12 identical to ours, 2 DIFFER — both are ERRATA FIXES, introduced in commit a6991258 "Beta 0.22 (#593)" 2023-12-13:
  - libieeep1788_num.itl:168  `midRad [nai] [nai] = NaN NaN;` -> `midRad [nai] = NaN NaN;` (the typo of oheim issue #16)
  - mpfi.itl:603  `wid [0.0, 0.0] = -0;` -> `wid [0.0, 0.0] = 0;` (wid of a point is +0 under RoundUp of 0-0; mpfi.itl says -0)
  -> actionable for us: check whether our runner compares the sign of a zero numeric result for wid, and whether midRad [nai] [nai] is silently skipped (we have no NaI, D16).
- test/ITF1788_tests/*.jl are generated from the .itl (test/generate_ITF1788.jl) -> no new vectors.
- Hand-written Julia tests in test/intervals/: e.g. arithmetic/power.jl 447 `@test` lines, trigonometric.jl 381, basic.jl 257, hyperbolic.jl 158, rounding.jl 130 (counted 2026-09-29). Julia @test expressions (interval(...) literals, often with BigFloat or rational bounds, many are property checks not vectors). Would need per-file hand translation; no machine-readable vector format. Value: moderate at best.

## 2b. GNU Octave `interval` package 3.2.2 (2026-02-16, latest; maintainer Oliver Heimlich) — VERIFIED 2026-09-29

Source: https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/interval-3.2.2.tar.gz
(sha256 c0131b7a...d176 matches gnu-octave/packages interval.yaml). Repo: https://sourceforge.net/p/octave/interval/ . Package licence GPL-3.0-or-later.

- inst/test/itl.mat (Octave binary, gzip): the ITF1788 data. Parsed with a throwaway Octave-binary reader and compared record-by-record against our 19 .itl files (floats, decorations, signals):
  9457 records vs our 9542 (mulRevToPair's 347 stored as two ops, re-paired). Content IDENTICAL except:
  * absent from itl.mat: acot 30, acoth 30, midRad 25 (Octave lacks those ops / NaI midRad); 
  * 40 textToInterval records differ only by whitespace INSIDE the quoted string, e.g. `b-textToInterval "[ Empty  ]"` — the .itl has two spaces, OUR tests/itf1788/itl.py collapses whitespace inside quoted strings (parse_file: `' '.join(statement.split())`), so our Text value is "[ Empty ]". Harmless today (text constructors not run), a fidelity bug once they are.
  -> itl.mat adds ZERO vectors beyond our 19 files; it does not carry the two IA.jl errata either (by construction it matches ours).
- inst/test/crlibm.mat + crlibm.tst: CRlibm's test/*.testdata converted to Octave binary. 52,504 vectors, 14 functions x 4 roundings (RD/RU/RN/RZ): acos asin atan cos cosh exp expm1 log log10 log1p log2 sin sinh tan. Counts: e.g. sin_rn 10533, cos_rn 10790, exp_rn 4771, atan_rn 5623, tan_rn 5723; RD/RU per function 8..649 (log1p_rd 8, exp_rd 13, asin_rd 595). Licence header: GPL-2.0-or-later (Octave conversion); original authors de Dinechin, Lauter, Lefevre (crlibm upstream is LGPL, see §3).
  -> directly relevant: RD/RU expected results = exactly our interval endpoints; prefer crlibm upstream .testdata over the Octave .mat.
- ~311 inst/*.m files carry inline `%!` tests: 1305 `%!assert` lines, 972 `%!test` blocks; ~1090 of the %! lines reference itl/testdata (i.e. are the ITF1788 data again). Remainder = hand-written Octave unit tests (GPL-3+), Octave syntax; translation by hand. Low value.

## 3a. CRlibm tests/*.testdata — VERIFIED 2026-09-29

Mirror: https://github.com/taschini/crlibm (last push 2020-10-01), tests/*.testdata. 21 files, 63,617 vectors (counted lines starting N/P/M/Z or RN/RU/RD/RZ):
acos 272, acospi 61, asin 2625, asinpi 58, atan 6852, atanpi 126, cos 11941, cosh 2105, cospi 244, exp 6140, expm1 732, log 1226, log10 148, log1p 586, log2 431, pow 10003 (all RN), sin 11017, sinh 1543, sinpi 341, tan 6886, tanpi 280.
Format: one line per vector, `<mode> <hi32 hex> <lo32 hex> <hi32 hex> <lo32 hex>  # comment`, mode N/P/M/Z (= RN/RU/RD/RZ); pow has two inputs. Trivial to parse (~20 lines). Contents: special cases, subnormals, exact cases, bad cases from Gvozdev, and "one in five of the very worst cases computed by Lefevre and Muller".
e.g. exp: RN 4771, RU(P) 638, RD(M) 13, RZ 718; sin: RN 10533, P 148, M 150, Z 186.
Licence: each testdata header says "distributed under the GNU Public Licence, see file COPYING" (COPYING = GPL v2); the library itself is LGPL-2.1 (COPYING.LIB). Test data only (our wheel ships intervals/ alone) - but GPL files in a repo: owner decision.
Coverage vs ours (grep of intervals/*.py 2026-09-29): we implement acos asin atan cos cosh exp expm1 log log10 log1p log2 sin sinh tan (14 match); we do NOT have sinpi/cospi/tanpi/asinpi/acospi/atanpi; pow: we have 1788 pow (RN-only data gives RN point -> tests our RU/RD bracket only if we derive: RD<=RN<=RU with RN exact-rounding check).
Use as oracle: RU and RD lines check endpoints directly; RN/RZ lines: RD(x) and RU(x) must bracket and one of them must equal the RN/RZ value (and exactness => both equal).

## 3b. CORE-MATH worst-case files — VERIFIED 2026-09-29

Repo: https://gitlab.inria.fr/core-math/core-math (GitLab project 35719), master 31a1c6f7 committed 2026-09-29 (actively maintained). LICENSE: MIT (repo-wide; .wc files carry no separate licence header in the samples read).
- src/binary64/<f>/<f>.wc = INPUTS ONLY (hex-float per line, `x,y` for bivariate; `#` comments). Expected outputs are computed at test time by MPFR (support/check_worst_uni.c tests FE_TONEAREST/TOWARDZERO/UPWARD/DOWNWARD against MPFR_RNDN/Z/U/D). So using it needs our own oracle (python-flint/arb or gmpy2/MPFR) per input and per rounding mode.
- binary64 functions with .wc (41): acos acosh acospi asin asinh asinpi atan atan2 atan2pi atanh atanpi cbrt cos cosh cospi erf erfc exp exp10 exp10m1 exp2 exp2m1 expm1 hypot lgamma log log10 log10p1 log1p log2 log2p1 pow rsqrt sin sincos sinh sinpi tan tanh tanpi tgamma. No rootn (we have rootn; cbrt covers n=3 only).
- Sizes (bytes, X-Gitlab-Size): exp 25.4M (1,129,426 non-comment lines), pow 32.1M (~1.0M lines), sin 44.0M, cos 24.9M, tan 24.6M, asinh 26.6M, acosh 25.5M, log1p 7.8M, log 3.1M, cbrt 2.2M (106,254 inputs), hypot 1.4M, atan2 1.2M, expm1 2.7M, log2 0.7M, log10 1.5M, exp2 1.7M, exp10 1.4M, sinh 6.1M, cosh 0.8M, tanh 0.7M, atanh 0.7M, asin 5.2M, acos 5.8M, atan 1.2M. All 41 files ~430 MB total -> order of 19M inputs (estimate from ~22.5 bytes/line; UNVERIFIED exact total).
- Coverage vs ours: exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh cbrt hypot pow = 24 of our functions. Not ours: *pi, erf, erfc, gamma, *m1/*p1 base-2/10, rsqrt, sincos.
- Effort: parse trivial; cost is the oracle evaluation (per input, 2 directed modes) - with MPFR via gmpy2 fast; with arb / Decimal slow. Too big to vendor whole; SAMPLE (e.g. every k-th line, or first N per file) and vendor the sample with the MIT notice, or fetch at test time with a pinned commit + hash.

## 3c. Lefevre (vinc17.net) hard-to-round cases — VERIFIED 2026-09-29

Page: https://www.vinc17.net/research/testlibm/index.en ("Worst cases used for the tests above (775 KB). Last update on 2020-11-27").
- testlibm-data.xz: 53,355 lines, all mode N. Format `N <fn> <x in binary sci notation> <RN(f(x)) in binary> <rounding bit>`; testmpfr.c (hrtests.tar.xz) shows the rounding bit + RN value determine RZ/RD/RU exactly (it checks MPFR_RNDZ/D/U from them). So EVERY line yields directed-rounding expected results.
  Functions (codes from hrtests testmpfr.c): ach=acosh 1877, acs=acos 1569, ash=asinh 2262, asn=asin 1655, ath=atanh 1552, atn=atan 1735, cbr=cbrt 138, ch=cosh 2026, cos 1576, cub=x^3 150, e01 538 (UNVERIFIED meaning, not in testmpfr table read), e10=exp10 1668, em1=expm1 7578, ex2=exp2 1145, exp 2271, isq=x^-2 2439, isr=x^-1/2 2611, l10=log10 1883, l1p=log1p 7550, lg2=log2 929, log 2819, sh=sinh 2215, sin 1611, tan 1706, th=tanh 1852.
- hrcases/ directory (9 files, 194,340 lines, mixes N and Z lines; Z = RZ value + bit, same derivation): hrcases-cub 504, -ex2lg2 4517, -ex10lg10 8703, -explog 45503, -hyper 20498, -isq 10878, -powint 37399 (p=x^n, r=x^(1/n) i.e. pown/rootn for small n, plus q/s variants UNVERIFIED meaning), -trig 28551, -trig2pi 37787 (att/stp/ttp/ast/ctp/act = pi-scaled trig; not ours).
- Licence: NO licence statement on the page or in the data files (UNVERIFIED terms -> ask V. Lefevre or treat as all-rights-reserved). hrtests programs are GPL (Copyright 2003-2015 Vincent Lefevre).
- Coverage vs ours: exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh cbrt + pown (x^n) + rootn (x^(1/n)) = 22+ functions. Parse effort: small (binary mantissa strings -> Fraction); oracle is self-contained (no MPFR needed). Size small (~1 MB xz total).
- CRlibm's worst-case lines ("one in five of the very worst cases computed by Lefevre and Muller") are a subset of the same research.

## 3d. glibc math/auto-libm-test-out-* — VERIFIED 2026-09-29 (not in the original brief; found while surveying)

Source: https://sourceware.org/git/?p=glibc.git;a=tree;f=math (HEAD). Generated by gen-auto-libm-tests.c with MPFR from auto-libm-test-in (10,874 lines; LGPL-2.1-or-later header, "Copyright (C) 1997-2026 FSF").
Format: `= <fn> <downward|tonearest|towardzero|upward> <binary32|binary64|intel96|m68k96|binary128|ibm128> <args hex-float> : <result hex-float> : <flags e.g. inexact-ok, overflow, errno-erange>`. Expected results for ALL FOUR rounding modes, per format, with exception flags. Parse effort small (one regex).
binary64 vectors per directed mode (downward = upward count), counted 2026-09-29:
acos 119, acosh 161, asin 93, asinh 152, atan 53, atan2 405, atanh 183, cbrt 68, cos 132, cosh 172, exp 182, exp10 152, exp2 157, expm1 122, fma 318, hypot 331, log 57, log10 60, log1p 96, log2 75, pow 1330, pown 494, rootn 434, sin 141, sinh 284, sqrt 176, tan 134, tanh 136, compoundn 332.
=> ~5,400 binary64 vectors x 2 directed modes over 28 of our ops, INCLUDING pown, rootn, hypot, fma, sqrt, atan2 (things CRlibm/Lefevre do not cover or cover thinly).
Also narrow-add/sub/mul/div/sqrt/fma files: binary64-RESULT lines (downward): add 975, sub 975, mul 729, div 1029, sqrt 184, fma 1458 - but these are "narrowing" ops (args may need wider formats, arg_fmt(...) field); the subset with binary64-representable args is UNVERIFIED (needs arg_fmt filtering). Relevant to directed + - * / sqrt fma.
Other files: acospi asinpi atanpi atan2pi cospi sinpi tanpi erf erfc lgamma tgamma j0/j1/jn/y0/y1/yn exp2m1 exp10m1 log2p1 log10p1 powr rsqrt sincos + complex c* (not ours).
Licence LGPL-2.1+ (test data only). Total size for our ~30 files ~19 MB (all formats); binary64-only extraction would be a few hundred KB.

## 3e. LLVM libc and RLIBM — checked 2026-09-29

- LLVM libc libc/test/src/math/*_test.cpp (Apache-2.0 WITH LLVM-exception): a handful of hard-coded hex inputs per function (e.g. exp_test.cpp lines ~43-44 list 0x3FD79289C6E6A5C0...), expected values computed at test time by MPFR via EXPECT_MPFR_MATCH_ALL_ROUNDING. Inputs-only, small; hypotf_hard_to_round.h is binary32. Low value vs CORE-MATH/glibc. SKIP.
- RLIBM (rutgers-apl: rlibm, rlibm-all, rlibm-prog, The-RLIBM-Project): oracle files and exhaustive validation are for 32-bit and smaller formats (per repo descriptions / search results); no published binary64 directed-rounding vector set found. SKIP (UNVERIFIED that no double data exists in rlibm-prog).

## 2c. inari (Rust) — VERIFIED 2026-09-29
https://github.com/unageek/inari (MIT, last push 2025-01-26). tests/itf1788_tests/*.rs are generated (tools/gen_itf1788_tests.sh) from the ITF1788 submodule pinned at unageek/ITF1788 d8c2a644, which is 24 commits ahead of oheim master with 0 changes under itl/. 15 generated files (same set as IA.jl + mul_rev). No new vectors; inline #[cfg(test)] unit tests in src/*.rs not counted (Rust syntax, hand translation). SKIP.

## 2d. libieeep1788 (C++, nehmeier) — VERIFIED 2026-09-29
https://github.com/nehmeier/libieeep1788 (Apache-2.0; last commit 2015-03-30). The eleven libieeep1788_*.itl are conversions of its test/p1788 Boost tests. NOT converted into ITF1788: flavor io (test_mpfr_bin_ieee754_flavor_io.cpp, 586 BOOST_CHECK*), util_func (171), validation_func (311), decoration/test_decoration.cpp (227), io/test_io_manip.cpp (41), integration io (28). (counts = grep -c BOOST_CHECK, include non-vector checks). io = text input/output (textToInterval, intervalToText with format manipulators) - relevant only if/when we implement 1788 text I/O; validation = is_valid/setDec-style checks. C++ Boost syntax -> hand/regex conversion, moderate effort. MAYBE (only for text constructors / output).

## 2e. MPFI current test suite vs mpfi.itl — VERIFIED 2026-09-29 (counts approximate where noted)
Repo https://gitlab.inria.fr/mpfi/mpfi (GitLab 28417), active: "Version 1.5.5" committed 2026-09-27. Repo COPYING.LESSER is now LGPL **v3** (mpfi.itl header says LGPL-2.1+ for the 2015-16 conversion). tests/*.dat carry no licence header (sampled exp.dat, exp10.dat).
- tests/ has 124 .dat files, 6,070 data lines total. Format: whitespace columns `inexact-flag prec lo hi prec lo hi [...]` with hex/decimal endpoints, many at precisions other than 53, some with NaN endpoints.
- mpfi.itl has 57 testcases / 1,382 vectors (our file, counted 2026-09-29). Lines in .dat where every precision field is 53 and no NaN (heuristic, imprecise for mixed scalar ops): 2,583; files with more such lines than mpfi.itl has vectors: extra ~1,260 lines.
- What is genuinely new for us: (a) whole functions never converted though present since 2010: exp10 (~16 b64 lines), exp10m1 (~17), exp2m1 (~13), rec_sqrt (~12, added 2019), plus diam/mag-type utilities, bisect, blow, has_zero, union/intersect extras; (b) mixed interval-scalar ops add_d/sub_d/mul_d/div_d (~6 more each than itl), _si/_ui/_z/_q/_fr variants (~20-46 each; for binary64 mostly redundant with interval-interval ops), d_div/d_sub/ui_div/si_div etc. (scalar OP interval); (c) a few more lines for existing ops (abs 14 vs 12, hypot 22 vs 17, union 19 vs 14, mul 55 vs 50, add 22 vs 19) - UNVERIFIED whether those are genuinely new or precision/NaN edge lines my filter mis-classified. Last .dat edits 2024-02-08 / 2024-06-18.
- Effort: small parser (hex-float/decimal -> Fraction, filter prec==53, drop NaN rows), plus op mapping. Value: modest - mostly exp10, hypot, mixed-scalar ops.

## 2f. Others checked 2026-09-29
- JInterval (https://github.com/jinterval/jinterval, BSD-2-Clause, last push 2018-08-10): 887-entry tree, 15 Java test files, no data/vector files (.itl/.dat/.txt other than surefire reports). Cited with ITF1788 in Revol/Benet/Ferranti/Zhilin, arXiv 2205.11837 (2022), which links no new vector set on its abstract page. SKIP.
- kv (C++, Kashiwagi; http://verifiedby.me/kv/index-e.html): kv-0.4.62 (2026-08-01), MIT. Only test programs (e.g. test/test-rounding.cc); no 1788 vector data mentioned. SKIP.
- C-XSC: latest 2.5.4 (2014-02-28), LGPL; c-xsc.itl was converted from 2.5.4 (Octave itl.mat.license says so) -> nothing newer. SKIP.
- FI_LIB: fi_lib.itl converted from FI_LIB 1.2 (Octave itl.mat.license). No newer FI_LIB release found; filib++ test suite NOT checked (UNVERIFIED).
- IEEE 1788.1-2017: no public conformance vector set found (web search; the standard itself says a 1788.1 program should run unchanged on a 1788-2015 implementation, so ITF1788 covers it). UNVERIFIED absence, but nothing surfaced.

## 4. IEEE 754 binary64 arithmetic suites — VERIFIED 2026-09-29

- IBM FPgen "Test Suite for IEEE 754R Compliance": mirror https://github.com/sergev/ieee754-test-suite (last push 2025-06-03; original IBM download page now 404). Downloaded all 24 binary .fptest files: 93,175 vectors, ALL `b32` (binary32); binary64 lines = 0. Rest is decimal. Header "Copyright of IBM Corp. 2005", no licence grant (GitHub spdx none). Format `b32<op> <rm: > < 0 =0 =^> <trapped> <inputs> -> <output> <flags>`. For binary64: nothing. SKIP (b64 version from IBM UNVERIFIED/unavailable).
- Berkeley TestFloat 3e (http://www.jhauser.us/arithmetic/TestFloat.html, GitHub ucb-bar/berkeley-testfloat-3, BSD-3-Clause-style "License for Berkeley TestFloat Release 3e", 2018-01-20): a GENERATOR (testfloat_gen) not a vector file; needs Berkeley SoftFloat 3 to build. f64 ops: f64_add sub mul mulAdd div rem sqrt roundToInt + compares/conversions; rounding options -rnear_even -rnear_maxMag -rminMag -rmin -rmax (-rodd); output = hex operands + expected result + exception-flag bits; levels 1/2 (level 2 = millions). Build in WSL, generate -rmin/-rmax f64 vectors for add/sub/mul/div/sqrt/mulAdd, vendor a sample. Best source for directed + - * / sqrt fma.
- UCBTest (https://www.netlib.org/fp/ucbtest.tgz, 1 MB): ucb/ucblib/*d.input double-precision vector files, 25 functions. Line format `<fn> <rm n|z|p|m> <rel eq|uo|vn|nb|ge|le> <flags> <hex words of inputs> <hex words of result>`; rel `eq` = exact (correctly rounded) match. Non-comment lines: addd 1431 (1179 eq), subd 1313 (1061 eq), muld 1361 (1088 eq), divd 1558 (1282 eq), sqrtd 404 (230 eq), powd 1858 (1213 eq), expd 368 (237 eq), logd 325 (53 eq), sind 205 (118 eq), plus acos asin atan atan2 cos cosh sinh tan tanh hypot log10 cabs ceil floor fabs fmod. Modes split roughly evenly n/z/p/m (divd: m 379, n 383, p 380, z 380). Licence: "Copyright (C) 1988-1994 Sun Microsystems ... Any person is hereby authorized to download, copy, use, create bug fixes, and distribute" subject to: no fee beyond media, keep notice, comply with US export control. Easy parse. Good small hand-picked directed-rounding set for + - * / sqrt; for transcendental functions only `eq` rows are correctly-rounded claims (vn/nb are tolerance rows).
- Paranoia (netlib): a diagnostic program that probes arithmetic properties; no vectors. SKIP (licence not checked).

## 1b. GitHub code search for .itl files (`testcase extension:itl`, 114 hits, 2026-09-29) — blob-compared with ours

- JuliaIntervals/ITF1788.jl (MIT, push 2024-12-09): all 19 files; 17 identical, libieeep1788_num + mpfi identical to IA.jl's errata versions (midRad, wid +0).
- denehoffman/maryada (Apache-2.0, push 2026-08-17): 7 files; only libieeep1788_num differs = the same midRad errata fix.
- dpsanders/ValidatedNumericsTests.jl (push 2020-02-08): all 19, differ only in licence/comment headers (older GPL-header snapshot); no vector differences in elem (checked). Superseded by ours.
- nehmeier/ITF1788: old names/versions (known).
- **neilkichler/cuinterval** (CUDA, MIT, push 2026-09-23), tests/itl/: 3 NEW files + 3 edited:
  * custom.itl (2025-11-13..2026-09-22): 26 vectors (sqrt 13, log 7, log1p 6) - domain-edge cases (-0, [-inf,0], empty). Parses with our itl.py.
  * intervalarithmeticjl.itl (2024-02-29..2026-06-01): 57 vectors (sinpi 12, cospi 12, tan 4, rootn 29) "adapted from IntervalArithmetic.jl"; one non-ITL literal `[1./3., 1./2.]` breaks our parser.
  * filib.itl = our fi_lib.itl with op `logp1` renamed `log1p` (30 lines), no value changes.
  * mpfi.itl edits: logp1->log1p rename (7), plus 2 VALUE edits; libieeep1788_elem.itl 1 VALUE edit. I ADJUDICATED all three with mpmath (prec 300) — ALL THREE cuinterval edits are WRONG, ITF1788 is right:
    - log10 [0X1.B333333333333P+0,0X1.C81FD88228B2FP+98]: exact log10(hi)=29.7517829123777708440...; ITF upper 0X1.DC074D84E5AABP+4 >= exact (ok); cuinterval's ...AAAP+4 < exact (not an enclosure).
    - log2 [1.0, 0x8ac74d932fae3p-21]: exact 30.1166405535250808429...; ITF upper 0x1e1ddc27c2c70fp-48 ok; cuinterval ...70ep-48 < exact.
    - cot [0x13a28c59d5433bp-44, 0x9d9462ceaa19dp-43]: lo/pi = 100.000000000000000625 > 100, hi/pi = 100.318 -> no multiple of pi inside; cuinterval's `[entire]` claim ("float64(100*pi) < 100*pi so we cross the asymptote") is false; ITF's finite result stands.
  -> worth pulling: custom.itl (26) + the rootn/tan rows of intervalarithmeticjl.itl (33); NOT the edits.

## 3f. licence search for Lefevre's data, and CORE-MATH's provenance (the session, 2026-10-02)

the owner asked: is there a licence for Lefevre's data anywhere, and if not, CORE-MATH. fetched and searched 2026-10-02:

- Lefevre: NO licence anywhere. searched: the testlibm page (index.en) and the site home page; all 10 data files
  decompressed (`testlibm-data` 53,355 lines, `hrcases-*` 194,340 lines: every line is a data row, no header or
  comment); `hrtests.tar.xz` (mktestlibm, mktestmpfr, testlibm.c, testmpfr.c, version-info.h: the programs are
  GPL-3-or-later, "Copyright (c) 2003-2015 Vincent Lefevre", no README, nothing about the data); the four papers the
  page links (arith13, arith15, arith17, ieeetc1998-tcrt; no terms for the data). the page notes that some
  `testlibm-data` rows are not true worst cases (an old filter turned a worst case of f into one of f^-1; fixed in
  2007, the file kept as is so old machines' results stay comparable), and that `hrcases/` is "not all the
  hard-to-round cases I have found, but at least the most important ones", subnormals ignored.
- CORE-MATH: LICENSE at master is MIT ("The CORE-MATH code is distributed under the following license"), and the
  `.wc` files carry no other terms. their comments give each block's source: most are CORE-MATH's own BaCSeL runs
  (e.g. cbrt.wc records the bacsel command lines, >= 44 identical bits after the round bit; exp.wc >= 41 bits), and
  some blocks are LEFEVRE'S, cited by URL: `log.wc` "worst cases from .../hrcases/hrcases-explog.xz", `sin.wc`
  "worst cases from .../testlibm-data.xz (update from 2020-11-27), from 0 to pi, with 46 to 59 identical bits".
  so part of Lefevre's data already ships under CORE-MATH's MIT notice. also in the files: argument-reduction worst
  cases for sin (smallest |x/(2 pi) cmod 1| per binade), special values, non-regression inputs, and some "AI
  generated" coverage inputs (log.wc, atan.wc), which are not worst cases.

## 3g. CORE-MATH worst cases through our evaluator: a probe (the session, 2026-10-02)

CORE-MATH master `284b3b0e198042c38f5c30316f696786b10816b0`; seven binary64 `.wc` files fetched whole (finite
inputs: log 134,951, cbrt 106,248, atan 55,764, exp2 77,176, cosh 36,686, sin 1,975,921, exp 1,129,426). 300
inputs sampled per file (seed 1), each through `elementary.rounded` DOWN and UP on the pure path
(`backend.fast` None), the final Ziv precision recorded by wrapping `elementary._ziv`, each result compared with
MPFR (gmpy2, 53 bits, emin -1073, emax 1024, subnormalize, RoundDown/RoundUp). 4,200 calls, 0 mismatches.
per call: median 0.0-0.1 ms, p99 0.1-1.0 ms, max 1 ms (this laptop). final p (64 is `_START_PRECISION`):
log 64: 316, 128: 284; exp 298/302; sin 244/356; cosh 326/272; exp2 348/230 (22 exact); atan 64: 210, 128: 376,
256: 2, 1024: 2, 2048: 2, 4096: 8; cbrt 590 of 600 never reach the Ziv loop: most of `cbrt.wc` is its
"exact cases in [1,8)" block (exact cubes, answered by `exact`). so about half the worst cases end at 128 bits
and a few atan inputs go to 4096; a uniform sample of a file follows its biggest block, not its hardest one.
cost of the whole set for our functions, estimated from the survey's sizes (about 250 MB, about 11M inputs, x2
directions, about 0.15 ms each): about an hour on one process, before pow's bivariate rows are measured.

## 3h. second opinion: what the worst cases can catch (a review agent, 2026-10-02; the session re-checked the starred items)

- sabotage, exp, 3000 BaCSeL worst cases and 3000 random doubles, DOWN/NEAREST/UP against MPFR (*): with no
  `_widen` at all, with `_guard` = 0, and with `_exp_fix`'s guard `q = p`, both sets stay green; with `_ziv`
  returning at p = 64 without doubling, 2138 of 9000 worst-case calls go red and 0 of 9000 random ones. so the
  worst cases pin that the p > 64 path exists and works; a wider-but-rigorous enclosure is absorbed by the
  doubling and no binary64 input sees it (they sit 2^-97..2^-113 from a boundary, the second step decides at
  2^-128 or finer).
- within one BaCSeL block the inputs end about half at 64 and half at 128 bits whatever the block's `-m`; whole-file
  runs of atan, cosh, exp2, log and cbrt (410k inputs, ~1.23M calls, three directions): 0 mismatches. the inputs
  that behave differently are in the small blocks (special values, thresholds, non-regression, argument-reduction
  extremes, subnormal outputs); atan's ±2^e block is the one reaching 512-4096 bits.
- bivariate files map onto our scalar calls with 0 mismatches on the rows run: atan2.wc via
  `elementary.rounded_angle`, hypot.wc via `rounded('sqrt', x*x + y*y)`, a pow.wc sample via `rounded_pow`.
- (*) `elementary.rounded('log', Fraction(-1), DOWN)` never returns (timed out at 20 s; the scalar's contract is
  "x inside the domain", the set layer clips first), and log.wc's special-values block has negative inputs.
- the ledger: `tools/gate.py::read_ledger` skips any row whose width is not `len(COLUMNS)` (*), so a third
  content-id scope as a new column would orphan every recorded row without a migration.
- its recommendation: a vendored gate sample by block (small blocks whole, ~200 per large block, three
  directions, a floor on rows ending at p >= 128), a `tools/coremath.py` that fetches at a pinned commit with a
  sha256 manifest and runs `--all` once as a dated census here, and automation through the ledger only if the
  census ever finds what the sample did not.

## 3i. built: tools/coremath.py, the gate sample, the first full check (the session, 2026-10-02)

- `tools/coremath.py` (`fetch`, `sample [--check]`, `check`, `status`, `pin`) at CORE-MATH `284b3b0e1980` (24 files,
  236 MB in `.scratch/coremath-cache/`, kept; sha256 in `tests/coremath/MANIFEST.tsv`); 16 of the 24 are
  WORST_SYMMETRIC upstream, so their inputs are also checked at -x. one parser trap: a unary line may carry a
  trailing bit count (`0x1.4f1d73be27a31p+1 44` in sin.wc), so a unary line's operand is its first token only.
- the gate sample `tests/coremath/*.tsv`: 29,054 rows (2.7 MB), every one right on the pure path, 12 s for the 24
  functions; calls past 64 bits per function from 56 (hypot) to 7,224 (atan). sabotage table in the commit
  `3044197` (4 breaks, each red).
- the first full check, `check --jobs 4` at `3044197`: 17,077,691 inputs (sign mirrors included), 51,233,073 calls
  (DOWN, NEAREST, UP), 0 mismatches against MPFR, none over 1 s, 2547 s (`references/coremath-runs.tsv`).
- `elementary` now refuses a point outside a function's domain (`008fa4a`): log, log2, log10, log1p, atanh and
  acoth used to loop forever there, asin and acosh raised a misleading error, and acos(-2) answered 0.0.

## Summary / ranking (2026-09-29)
Answer: yes, we have the full ITF1788 vector set (no fork, branch, PR or downstream adds .itl vectors to those 19 files; Octave's itl.mat is the same data). Known upstream errata: 2 (midRad [nai] [nai]; mpfi wid [0,0] = -0), fixed in IA.jl/ITF1788.jl/maryada, not upstream.
Pull-in order: (1) glibc auto-libm-test-out binary64 directed rows; (2) Lefevre testlibm-data + hrcases (licence question first); (3) CRlibm testdata RU/RD (GPL-2 header); (4) CORE-MATH .wc sampled + own MPFR/arb oracle; (5) TestFloat-generated f64 -rmin/-rmax + UCBTest eq rows for + - * / sqrt fma; (6) cuinterval custom.itl + IA.jl-derived rootn/tan rows; (7) MPFI exp10/hypot/mixed-scalar .dat rows. Skip FPgen (b32 only), LLVM libc, RLIBM, JInterval, kv, Paranoia, Octave %! tests.
