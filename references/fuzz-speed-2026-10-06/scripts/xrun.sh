#!/usr/bin/env bash
# one whole-suite run in the private copy. usage: xrun.sh <label> <workers|0> <env...> -- [pytest args]
#   workers 0 = no xdist. env words are KEY=VALUE (e.g. HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10)
set -u
label=$1; workers=$2; shift 2
envs=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
[ "${1:-}" = -- ] && shift
D=/c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed
cd $D/tree
PY=C:/Users/user/anaconda3/envs/intervals/python.exe
xd=()
[ "$workers" -gt 0 ] && xd=(-n "$workers" --dist worksteal)
load() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ForEach-Object { \$_.ProcessId.ToString() + ':' + (\$_.CommandLine -replace '.*python.exe\"? ','').Substring(0, [Math]::Min(60, (\$_.CommandLine -replace '.*python.exe\"? ','').Length)) }) -join ' | '; 'cpu%=' + (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average" 2>/dev/null | tr '\r\n' '  '; }
echo "$(date '+%F %T') START $label workers=$workers ${envs[*]} $* :: load: $(load)" >> $D/runs.tsv
s=$(date +%s)
env -u MULTIINTERVAL_BACKEND -u HYPOTHESIS_PROFILE -u FUZZ_MULTIPLIER "${envs[@]}" PYTHONPATH="$(cygpath -w $D/site)" \
  $PY -m pytest -q -p no:cacheprovider --durations=0 "${xd[@]}" "$@" > $D/$label.log 2>&1
rc=$?
e=$(date +%s)
echo "$(date '+%F %T') END $label rc=$rc wall=$((e - s))s :: $(grep -E ' in [0-9.]+s' $D/$label.log | tail -1) :: load: $(load)" >> $D/runs.tsv
