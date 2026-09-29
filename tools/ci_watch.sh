#!/usr/bin/env bash
# the babysitter's watch after a push: waits until every workflow run of a commit on GitHub has
# finished, prints one line per run, and for a red run saves what a local fix needs to
# .scratch/ci/<run id>/: failed.log (`gh run view --log-failed`), and the run's artifact if it has
# one (the fuzz job uploads fuzz.log and .hypothesis/). it prints the failing tests, hypothesis's
# falsifying examples and @reproduce_failure lines. exits 0 if all green, 1 if any run was not.
#
#   tools/ci_watch.sh                # origin/master's commit
#   tools/ci_watch.sh <sha> [n]      # wait for at least n runs (default 2 on master: ci, fuzz)
#
# the token comes from the git credential store per call (gh is not logged in on this machine).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
sha=$(git rev-parse "${1:-origin/master}")
want=${2:-2}
token() { printf 'protocol=https\nhost=github.com\n\n' | git credential fill | sed -n 's/^password=//p'; }
runs() {
    GH_TOKEN=$(token) gh run list --commit "$sha" --limit 20 \
        --json databaseId,workflowName,status,conclusion \
        --jq '.[] | "\(.databaseId) \(.workflowName) \(.status) \(.conclusion)"'
}
echo "ci_watch: ${sha:0:7}, waiting for $want run(s) to finish ($(date '+%Y-%m-%d %H:%M'))"
while true; do
    list=$(runs 2>/dev/null) || list=""
    total=$(printf '%s\n' "$list" | grep -c . || true)
    open=$(printf '%s\n' "$list" | awk 'NF && $3 != "completed"' | grep -c . || true)
    if [ "$total" -ge "$want" ] && [ "$open" -eq 0 ]; then
        break
    fi
    sleep 60
done
status=0
while read -r id name state conclusion; do
    echo "ci_watch: $name run $id: $conclusion"
    [ "$conclusion" = success ] && continue
    status=1
    dir=.scratch/ci/$id
    mkdir -p "$dir"
    GH_TOKEN=$(token) gh run view "$id" --log-failed > "$dir/failed.log" 2>&1
    GH_TOKEN=$(token) gh run download "$id" -D "$dir" > /dev/null 2>&1 || true
    # gh prefixes each log line with job, step and timestamp; the pytest text follows the timestamp
    sed -E 's/^[^\t]*\t[^\t]*\t[0-9TZ:.-]+ //' "$dir/failed.log" \
        | grep -E '^FAILED|passed in|failed in|Falsifying example|Failing test case|@reproduce_failure|^E +[a-z_]+ ?=' \
        | head -n 40
    echo "ci_watch: saved $dir"
done <<< "$list"
exit $status
