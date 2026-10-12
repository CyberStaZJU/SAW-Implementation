# SAW Implementation

Reference implementation of **Staleness-Aware Weighting (SAW)** for *Staleness-Aware Aggregation with Repeated Update Reuse in Asynchronous Federated Learning*.

At each server commit, SAW replaces the arriving clients' cache entries and recomputes normalized inverse-current-age weights over all valid entries. The age includes both pre-insertion delay and cache residence. Clients use ordinary local SGD. The implementation supports NVIDIA GPUs and Ascend NPUs, with persistent client caches on CPU by default.

## Layout

```text
code/
  algorithm/    SAW, seven asynchronous baselines, and cache ablations
  models/       local optimization, ResNet, and CNN models
  builders/     dataset and model construction
  data_reader/  FEMNIST and GSpeech readers
  utils/        arguments, sampling, aggregation, and result validation
  tools/        dataset preparation and experiment definitions
  scripts/      FEMNIST preparation
  tests/        algorithm, runtime, preparation, and collection tests
```

Datasets, results, logs, checkpoints, and environments belong outside this repository. The paper and private historical evidence are maintained separately from this public implementation.

## Environment

Use Python >= 3.10 and an external environment containing a matching backend wheel. From the repository root:

```bash
pip install -r code/requirements.txt
```

NVIDIA execution requires a compatible torch/CUDA installation. The qualified Ascend stack uses torch/torch-npu 2.9 and CANN 8.5. Activate the backend environment before launching a run. CANN may create compilation caches relative to the working directory, so run training from an external attempt directory.

## Data preparation

```bash
export SAW_DATA_ROOT="$HOME/SAW-data"
export SAW_STATE_ROOT="$HOME/.local/state/SAW-Implementation"
bash code/scripts/prepare_femnist_full.sh
python code/tools/prepare_femnist_truncated.py --num-users 1000
python code/tools/convert_femnist_to_pt.py
python code/tools/prepare_gspeech.py --download
```

FEMNIST uses writer partitions; GSpeech uses speaker partitions. Vision benchmarks use Dirichlet label partitions. The paper denotes their concentration by `alpha_Dir`; the compatible CLI argument remains `--alpha`. Dataset preparation defaults to `SAW_DATA_ROOT` or `~/SAW-data`; set the variable explicitly when reusing an existing cache.

## Single run

Start from the repository root and use a new output directory for every attempt:

```bash
export SAW_CODE_ROOT="$(pwd)"
export SAW_DATA_ROOT="$HOME/SAW-data"
export SAW_STATE_ROOT="$HOME/.local/state/SAW-Implementation"
export PYTHONDONTWRITEBYTECODE=1
RUN_DIR="$SAW_STATE_ROOT/runs/active/cifar-a01-m100-mc40-b5/SAW/seed=1/attempt=1"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
python "$SAW_CODE_ROOT/code/main_fed.py" \
  --algo SAW --dataset cifar --model resnet --distribution noniid \
  --alpha 0.1 --num_users 100 --concurrency 40 --buffer_size 5 \
  --seed 1 --total_rounds 500 --interval 1 \
  --local_period 10 --local_epochs 0 --local_bs 100 --lr 0.01 \
  --random_cost pareto --data_dir "$SAW_DATA_ROOT" \
  --cache_device cpu --gpu 0 \
  --output_dir "$RUN_DIR/accuracy" --system_metrics_log_dir "$RUN_DIR/metrics"
```

`--gpu -1` selects CPU. `--cache_device device` retains device-resident persistent caches. Accuracy outputs are exclusively reserved, and each run records its actual arguments, source identity, working directory, and terminal status in paired metadata. Reusing an existing output is rejected.

## Paper protocol and interpretation

The manuscript evaluates eight methods: SAW, FedAsync, FedBuff, CA2FL, MA-Light (`FedBuffMALight`), CASA, FADAS, and AdaMasFL. Datasets are CIFAR-10, CIFAR-100, TinyImageNet, FEMNIST, and Google Speech Commands v2. Vision partitions use `alpha_Dir` in `{0.1, 0.5}`. The six `(M, Mc, b)` settings are `(100,30,5)`, `(100,40,5)`, `(100,60,10)`, `(100,80,10)`, `(50,30,5)`, and `(50,40,5)`, with seeds 1 through 5 and 500 server commits.

`--random_cost pareto` samples `10 * numpy.random.pareto(2)`, a Lomax/Pareto-II service duration with scale 10 and shape 2, supported on nonnegative simulated time. Service duration and model-version age are distinct quantities. The local-period default is 10 mini-batch steps. A positive `--local_epochs` changes the local SGD loop length. AdaMasFL's hyperparameter formulas and server normalization still use `--local_period`; reproduce its paper protocol with `--local_period 10 --local_epochs 0`. The current manuscript explicitly specifies ten steps for vision tasks and for AdaMasFL on every task. Historical FEMNIST/GSpeech settings for other methods require their original run configurations; parser defaults alone do not reconstruct those runs.

The paper reports batch sizes of 100 for vision, 35 for FEMNIST, and 20 for GSpeech, with client learning rates of 0.01 for vision/FEMNIST and 0.1 for GSpeech. The CLI defaults to batch 100 and learning rate 0.01 across datasets: explicitly pass `--local_bs 35 --lr 0.01` for FEMNIST and `--local_bs 20 --lr 0.1` for GSpeech. Method-specific overrides apply separately. FADAS vision runs use local/server rates 0.1/0.0001 and disable the optional delay-adaptive switch. AdaMasFL uses its published formulas with ten local steps and 500 commits. Changing buffer size at a fixed commit horizon changes the completed-update count `K*b`.

Cache ablations include buffer-only weighting (`SAW_AdaInside`), arrival-frozen weighting (`ArrivalFrozenCache`), freshest-five selection, power/exponential decay, and SAW. `SAW_UniformCache` supports full-cache uniform weighting; `SAW_EqualW` is a separate implementation. The current manuscript's reported ablation table does not contain a matched full-cache uniform-weight result. The included `plot_ablation_60_10.py` targets a separate CIFAR-100 ablation and is not the plotting entrypoint for the manuscript's CIFAR-10 figure.

Tail-10 means the final ten evaluations within each run. Report the sample standard deviation across seeds separately from confidence intervals or training-process variability. Small-model tests establish their stated contracts; they do not establish convergence, wall-clock acceleration, or complete historical protocol equivalence.

## Collection and tests

From the repository root:

```bash
python code/gather.py --source "$SAW_STATE_ROOT/runs/active" \
  --output-dir "$SAW_STATE_ROOT/runs/awaiting_review" --total-rounds 500
```

This previews collection. Add `--copy` to preserve the paired accuracy, metrics, configuration, and identity in a review directory. The collector refuses overlapping or repository-local destinations and checks the configured final round. Formal verification also requires terminal-state evidence and a result audit.

```bash
cd code
python -m pytest tests -q
```

## License

MIT; see [LICENSE](LICENSE). Baseline implementations follow their respective publications. LEAF/FEMNIST and Google Speech Commands retain their dataset licenses and are prepared rather than redistributed by this repository.
