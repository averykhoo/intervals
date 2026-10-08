#!/usr/bin/env bash
# x1 (default hypothesis profile, as the local gate) per-file durations, one worker, in the private copy
cd /c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed/tree
PY=C:/Users/user/anaconda3/envs/intervals/python.exe
unset HYPOTHESIS_PROFILE FUZZ_MULTIPLIER MULTIINTERVAL_BACKEND
units="tests/itf1788 $(ls tests/test_*.py) multiinterval README.md tests/oracles.py tests/strategies.py"
for u in $units; do
  name=$(echo "$u" | tr '/' '_')
  s=$(date +%s%N)
  $PY -m pytest -q -p no:cacheprovider --durations=0 "$u" > ../x1/$name.log 2>&1; rc=$?
  e=$(date +%s%N)
  printf '%s\t%s\t%.1f\t%s\n' "$u" "$rc" "$(awk -v a=$s -v b=$e "BEGIN{print (b-a)/1e9}")" "$(grep -E ' in [0-9.]+s' ../x1/$name.log | tail -1)" >> ../x1/summary.tsv
done
echo done >> ../x1/summary.tsv
