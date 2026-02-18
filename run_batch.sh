#!/usr/bin/env bash
set -u

# Simple sequential runner for clusterize_batch.py
YEARS=(2015 2016 2018 2019 2020)
START_MM_DD="06-01"
END_MM_DD="10-30"
JOBS=1

PYTHON=${PYTHON:-python} 

# For using a uv venv set PYTHON to `uv run`, e.g.:
# export PYTHON="uv run"
# If you have a uv venv in another directory, set the UV_PROJECT_ENVIRONMENT variable in the .env file, e.g.:
# UV_PROJECT_ENVIRONMENT=/home/francesco/.venvs/ppcx-domains
# and then use `uv run --env-file .env` as PYTHON command, e.g.:
# export PYTHON="uv run --env-file .env"

SCRIPT=${SCRIPT:-clusterize_batch.py}
LOG_DIR=${LOG_DIR:-logs}
ERROR_LOG="${LOG_DIR}/errors.log"

mkdir -p "$LOG_DIR"

for label in "${YEARS[@]}"; do
  year=$(echo "$label" | grep -oE '^[0-9]{4}')
  if [[ -z "$year" ]]; then
    echo "[$(date -Iseconds)] Skipping invalid label: $label"
    continue
  fi

  start="${year}-${START_MM_DD}"
  end="${year}-${END_MM_DD}"
  logfile="${LOG_DIR}/${label}.log"

  echo "[$(date -Iseconds)] Starting ${label}: ${start} -> ${end} (log: ${logfile})"

  if $PYTHON "$SCRIPT" --date-range --start "$start" --end "$end" --jobs "$JOBS" >"$logfile" 2>&1; then
    echo "[$(date -Iseconds)] Completed ${label} OK"
  else
    rc=$?
    echo "[$(date -Iseconds)] ERROR (rc=$rc) running ${label} (see ${logfile})" | tee -a "$ERROR_LOG"
    echo "[$(date -Iseconds)] Moving on to next year after failure: ${label}"
  fi
done

echo "[$(date -Iseconds)] All runs finished."