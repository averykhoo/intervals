# fuzz and prepush parallelization study (2026-10-06)

a read-only research agent's study of whether the x10 fuzz (`tools/prepush.sh`, about 75-125 min serial on this
laptop) can run in parallel. it ran on a private copy of `1330182` (src `s:6fb5edf5749a`); no tracked file was
changed and pytest-xdist was installed to a side directory, not the env. moved here from `.scratch/fuzz-speed/`
on 2026-10-08, so the measurements are tracked; the logs, the tree copy and the side install were not kept.

**unfinished**: `report.md` is the agent's incremental log; it stops after its Q3 section and has no conclusion.
the bottom line below is the session's reading of it (2026-10-08), not the agent's.

## bottom line

* **xdist works on this suite, with no change to what a fuzz verdict means**: xdist splits tests between
  processes, never one test's examples, so a fuzz-x10 phase still runs every test at 10x. R3 (`-n 8 --dist
  worksteal`, light load): 34941 passed in 1257 s (21 min), the same count as serial, against 4450-7495 s serial
  on the ledger (3.5x-6x). R4 (x1, `-n 8`): 172 s; R5 (gmpy2, `-n 4`): 230 s.
* **the laptop bounds it**: an i7-1365U, 2 P-cores (4 threads) + 8 E-cores. R1 (`-n 4`, under the prepush and
  other repos' jobs) took 5439 s and failed one wall-clock test,
  `tests/test_fmt.py::test_white_space_runs_parse_in_linear_time[many-separators]` (10.22 s against `< 10`; 3.48 s
  alone, 5.00 s in R3). the timing tests are the parallel-safety risk; nothing else failed or reordered.
* **ten x1 runs are not a better search than one x10 run** (Q3, the three slowest tests): half the wall time but
  2x the compute (every run repeats the `@example`s and the simplest case), and fewer distinct inputs (843 against
  887), smaller ones (the size-capped early phase is 10 % of an x1 run, 5 % of an x10 one).
* **the hypothesis database is safe under concurrent writers** (source read and a two-process stress test: 0
  errors); a green run writes nothing to it anyway.
* what `tools/gate.py` would need: `phase_spec` adds `-n K --dist worksteal` for the fuzz and gmpy2 phases (K from
  an env var), the row records `workers=K`, `tests/test_gate_ledger.py`'s pins of the exact arguments change, and
  pytest-xdist joins `[test]` and the env (report §"what tools/gate.py would need").
* not explained: R6 (`-n 4`, x10, started 14:39) ended green the next day after 82919 s (23 h); most likely the
  laptop slept. it is in `data/runs.tsv` and is not evidence of anything.

## files

* `report.md`: the agent's log, verbatim (hypothesis 6.167.1 source facts, parallel-safety survey, per-test
  durations, the runs R1-R6, the scaling subset, Q3).
* `data/`: `runs.tsv` (each run's command, load and result), the per-test duration aggregates (`x1_`, `R1_`,
  `R3_durations.txt`, `x1_summary.tsv`), `q3_times.tsv` and `q3_obs.txt` (Q3's table), `growth.txt`, `alone.txt`.
* `scripts/`: what produced them (`xrun.sh` a run, `x1_loop.sh` the per-file x1 loop, `scale.sh` the subset scaling,
  `q3.sh` + `obs.py` Q3, `durations.py` the aggregates, `dbstress.py` the database test, `explicit_only.py` the
  pytest plugin for the fixed-cost run). they name `.scratch/fuzz-speed/` paths and the side install; they are a
  record, not runnable as they stand.
