#!/usr/bin/env bash
# scaling study on a fixed subset at x10: serial, -n 2, -n 4, -n 8, one after another
D=/c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed
S="tests/test_relations.py tests/test_minmax_fma.py tests/test_orders.py"
for n in 0 2 4 8; do
  bash $D/xrun.sh S_x10_n$n $n HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 -- $S
done
echo SCALE_DONE >> $D/runs.tsv
