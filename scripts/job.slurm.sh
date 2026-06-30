#!/bin/bash
#SBATCH --job-name=pseudoavalanche
#SBATCH --output=pseudoavalanche_%A_%a.out
#SBATCH --error=pseudoavalanche_%A_%a.err
#SBATCH --array=1-200
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1

SCRIPT_DIR="$(dirname "$0")"
JOB_LIST="$SCRIPT_DIR/job_list.json"
PYTHON="python"
PYTHONPATH="$SCRIPT_DIR/../src:${PYTHONPATH:-}"
export PYTHONPATH

if [ ! -f "$JOB_LIST" ]; then
  echo "Job list not found: $JOB_LIST"
  exit 1
fi

TOTAL_JOBS=$($PYTHON - <<PY
import json
from pathlib import Path
jobs = json.loads(Path(r"$JOB_LIST").read_text())
print(len(jobs))
PY
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
CHUNK_SIZE=$(( (TOTAL_JOBS + 199) / 200 ))
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
if [ $END -gt $TOTAL_JOBS ]; then
  END=$TOTAL_JOBS
fi

for IDX in $(seq $START $END); do
  $PYTHON -m pseudoavalanche.run --job-list "$JOB_LIST" --job-index "$IDX"
done
