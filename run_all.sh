#!/bin/sh
# Full library run: refresh likes, analyze tracks then sets, build the reports. Resumable: rerun it to pick up new likes.
# Usage: ./run_all.sh <soundcloud-username> [HH:MM]   optional start time (today); logs to data/run.log
# Run it under caffeinate so the Mac doesn't sleep, e.g.:
#   nohup caffeinate -i -s ./run_all.sh <username> 18:00 > /dev/null 2>&1 &
cd "$(dirname "$0")" || exit 1
[ -n "$1" ] || { echo "usage: $0 <soundcloud-username> [HH:MM]"; exit 1; }
PY=${PY:-./.env/bin/python}
LOG=data/run.log
mkdir -p data
echo $$ > data/run.pid

if [ -n "$2" ]; then
  start=$(date -j -f "%H:%M:%S" "$2:00" +%s)
  now=$(date +%s)
  [ "$start" -gt "$now" ] && echo "$(date) waiting until $2" >> "$LOG" && sleep $((start - now))
fi

echo "$(date) start" >> "$LOG"
$PY -m sctag.download_models >> "$LOG" 2>&1
$PY -m sctag.fetch_likes "$1" >> "$LOG" 2>&1 || { echo "$(date) could not fetch likes" >> "$LOG"; rm -f data/run.pid; exit 1; }
for kind in tracks sets; do
  for pass in 1 2; do  # second pass retries failures (DRM tracks will keep failing, that's expected)
    echo "$(date) $kind pass $pass" >> "$LOG"
    $PY -m sctag.analyze "$kind" --workers 6 2>&1 | grep --line-buffered -v "INFO\|Warning\|warn" >> "$LOG"
  done
done
$PY -m sctag.report >> "$LOG" 2>&1
echo "$(date) done: $(ls data/analysis/tracks/*.json | wc -l) tracks, $(ls data/analysis/sets/*.json | wc -l) sets, $(ls data/analysis/*/*.error 2>/dev/null | wc -l) failed" >> "$LOG"
rm -f data/run.pid
