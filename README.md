# SAW Implementation

Reference implementation of **SAW** (Server-Side Age-weighted Aggregation for asynchronous federated learning) together with the asynchronous baselines used in the paper: FedAsync, FedBuff, CA2FL, CASA, FADAS, AdaMasFL, plus the SAW cache-weighting ablations (`SAW_EqualW`, `SAW_AdaInside`, Arrival-Frozen-Cache, FreshTopK, Power, Exp).

SAW aggregates the valid update cache on the server using harmonic weights normalized by current total model age; clients run plain local SGD. The implementation supports NVIDIA GPUs and Ascend NPUs through device abstraction, with persistent client caches on CPU by default (`--cache_device device` keeps the device-resident variant).

## Repository layout

```
algorithm/    server aggregation rules (SAW + baselines + cache ablations)
models/       ResNet and CNN models, client local update
builders/     dataset and model construction
data_reader/  FEMNIST and GSpeech readers
utils/        sampling, aggregation, CPU state-dict cache, validation helpers
tools/        dataset preparation (FEMNIST, GSpeech) and grid generator
tests/        unit and runtime-contract tests
```

## Installation

Python >= 3.10 is required. Install into an external environment that already has the matching backend wheel:

```bash
pip install -r requirements.txt
```

- NVIDIA GPU: torch >= 2.8 with CUDA.
- Ascend NPU: install torch-npu matching torch; the qualified stack uses CANN 8.5.

Datasets live outside the repository. Set `SAW_DATA_ROOT` to your dataset cache (default `~/SAW-data`); runtime outputs default to `SAW_STATE_ROOT` (default `~/.local/state/SAW-Implementation`).

## Dataset preparation

```bash
export SAW_DATA_ROOT="$HOME/SAW-data"
bash code/scripts/prepare_femnist_full.sh          # full FEMNIST via LEAF
python code/tools/prepare_femnist_truncated.py   # 1000-client subset
python code/tools/convert_femnist_to_pt.py       # .pt cache for fast loading
python code/tools/prepare_gspeech.py --download  # Google Speech Commands v2
```

CIFAR-10/100 and TinyImageNet are downloaded automatically by torchvision into `SAW_DATA_ROOT`.

## Single training run

```bash
RUN_DIR="$SAW_STATE_ROOT/runs/active/seed=1"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
python "$SAW_CODE_ROOT/code/main_fed.py" --algo SAW --dataset cifar --distribution noniid \
  --alpha 0.1 --num_users 100 --concurrency 40 --buffer_size 5 \
  --seed 1 --total_rounds 500 --local_period 10 --random_cost pareto \
  --data_dir "$SAW_DATA_ROOT" --cache_device cpu --gpu 0 \
  --output_dir "$RUN_DIR" --system_metrics_log_dir "$RUN_DIR/metrics"
```

Notes:

- `--algo` selects the method (see `code/algorithm/`).
- On Ascend, activate the qualified CANN environment first; CANN writes compilation caches relative to the process working directory, so launch from an external task directory.
- `--gpu -1` runs on CPU.
- Every run exclusively creates its accuracy, metrics, resolved-config and identity files; rerunning the same output directory is rejected. Each run records its Git source identity, actual command, working directory and exit status in a paired `*-identity.json`.

## Result collection

```bash
python code/gather.py --source "$SAW_STATE_ROOT/runs/active" --output-dir "$SAW_STATE_ROOT/runs/awaiting_review" --total-rounds 500        # dry-run
python code/gather.py --source "$SAW_STATE_ROOT/runs/active" --output-dir "$SAW_STATE_ROOT/runs/awaiting_review" --total-rounds 500 --copy
```

The collector requires paired accuracy / system metrics / resolved config / identity per attempt, validates configured rounds, and refuses repository-local or overlapping destinations. `organize_completed_results.py` shares the same contract with `--sources ... --target-root ...`.

## Tests

```bash
cd code
python -m pytest tests -q
```

The suite covers runtime contracts (exclusive output reservation, identity lifecycle, collector pairing), algorithm recurrence on small models, dataset preparation paths and result validation. Tests that need a GPU/NPU skip automatically on CPU-only hosts.

## Method summary

- **SAW**: server-side harmonic re-weighting of the update cache by current total age; freshest cached updates dominate.
- **Cache ablations**: equal weights, inside-only weighting, arrival-time frozen weights, top-K freshest, power and exponential kernels.
- **Baselines**: FedAsync (staleness-weighted Polyak averaging), FedBuff, CA2FL, CASA, FADAS, AdaMasFL.

Training uses Pareto-distributed response costs by default (`--random_cost pareto`); staleness functions follow each method's original formulation.

## License

Released under the MIT License; see [LICENSE](LICENSE). Baseline methods follow their original publications; LEAF (FEMNIST) and Google Speech Commands datasets keep their respective licenses and are prepared, not redistributed, by the tools above.
