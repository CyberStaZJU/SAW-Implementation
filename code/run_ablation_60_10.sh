#!/usr/bin/env bash
# Five-seed CIFAR-100 cache ablations, sequential on the selected device.
set -euo pipefail
ROOT="${SAW_CODE_ROOT:-${SAW_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
PYTHON="${PYTHON:-python3}"
STATE="${SAW_STATE_ROOT:-$HOME/.local/state/SAW-Implementation}"
OUT_DIR="${OUT_DIR:-$STATE/runs/active/cifar100_ablation_60_10}"
LOG_DIR="${LOG_DIR:-$STATE/logs/cifar100_ablation_60_10}"
: "${SAW_DATA_ROOT:?Set SAW_DATA_ROOT to the existing external dataset cache}"
PYTHON="$(command -v "$PYTHON")"
if [[ "$PYTHON" != /* ]]; then
  PYTHON="$(cd "$(dirname "$PYTHON")" && pwd -P)/$(basename "$PYTHON")"
fi
ROOT="$(cd "$ROOT" && pwd -P)"
if [[ -f "$ROOT/main_fed.py" && "$(basename "$ROOT")" == code ]]; then
  ROOT="$(cd "$ROOT/.." && pwd -P)"
fi
CODE="$ROOT/code"
[[ -f "$CODE/main_fed.py" ]] || { echo "[ERROR] Missing source entry: $CODE/main_fed.py" >&2; exit 1; }
export SAW_CODE_ROOT="$ROOT"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export SAW_DATA_ROOT="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$SAW_DATA_ROOT")"
external_path() {
  "$PYTHON" - "$1" "$ROOT" <<'PATH_CHECK'
import sys
from pathlib import Path
path, root = (Path(value).expanduser().resolve() for value in sys.argv[1:])
if path == root or root in path.parents:
    raise SystemExit('Runtime paths must be outside the source repository')
print(path)
PATH_CHECK
}
STATE="$(external_path "$STATE")"
OUT_DIR="$(external_path "$OUT_DIR")"
LOG_DIR="$(external_path "$LOG_DIR")"
mkdir -p "$OUT_DIR" "$LOG_DIR"
for algo in SAW_EqualW SAW_AdaInside; do
  for alpha in 0.01 0.1 0.5; do
    for seed in 1 2 3 4 5; do
      out="$OUT_DIR/$algo/alpha=$alpha/seed=$seed"
      acc="$out/cifar100-$algo-$alpha-pareto-60-10-$seed-test_acc.txt"
      if [ -e "$acc" ]; then
        "$PYTHON" - "$acc" <<'CHECK'
import sys
from utils.result_validation import load_accuracy
load_accuracy(sys.argv[1], expected_rounds=500)
CHECK
        echo "[SKIP] $algo alpha=$alpha seed=$seed complete"
        continue
      fi
      task="$STATE/runs/active/cifar100_ablation_60_10/tasks/$algo/alpha=$alpha/seed=$seed"
      mkdir -p "$task"
      attempt="$(mktemp -d "$task/attempt=XXXXXXXX")"
      log="$LOG_DIR/${algo}_alpha${alpha}_seed${seed}_$(basename "$attempt").log"
      (
        cd "$attempt"
        "$PYTHON" "$CODE/main_fed.py" --dataset cifar100 --algo "$algo" \
          --distribution noniid --alpha "$alpha" --random_cost pareto \
          --num_users 100 --concurrency 60 --buffer_size 10 --seed "$seed" \
          --total_rounds 500 --data_dir "$SAW_DATA_ROOT" \
          --output_dir "$out" --system_metrics_log_dir "$out/metrics" \
          --gpu "${DEVICE_INDEX:-0}" > "$log" 2>&1
      )
    done
  done
done
