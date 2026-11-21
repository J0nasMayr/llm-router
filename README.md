# Resource-Aware LLM Routing in the Edge-Cloud Continuum

A router for LLM inference across edge and cloud nodes.
For each query, a contextual bandit picks the model that best trades accuracy against energy, choosing only among the models expected to answer within a latency deadline given the current state of the nodes.
In the background, a placement controller decides which models each node keeps in GPU memory, following the demand of the router.

## About this fork

This repository is a fork of [GreenServ](https://github.com/TZData1/llm-inference-router) ([paper](https://arxiv.org/abs/2601.17551)) at commit `8e055dd`.
It holds the code and LaTeX source (`tex/`) of a bachelor thesis that extends GreenServ's router to several edge and cloud nodes.
Everything below the horizontal line is GreenServ's original README, unchanged; it describes GreenServ's offline experiments, which this fork leaves untouched.

**Inherited from GreenServ, unchanged**

- `src/metrics/` except `normalizer.py`: the accuracy and energy code behind the reward.
- `db/`, `data/`, `config/metrics.yaml`, `config/datasets.yaml`: GreenServ's profiling data, the normalization bounds derived from it, and the dataset settings.
- `experiments/`, `paper_figures/`, `Makefile`: GreenServ's offline experiments X0-X4. The thesis does not reuse them.

Apart from the removals listed below and one adapted call, the only edits to these files, and to GreenServ's `config/general.yaml`, are cosmetic: an underscore prefix on four internal methods, in `src/metrics/accuracy.py` and `data/process.py`; the removal of a redundant root key from `config/datasets.yaml` (and the three `data/` files that read it) and from `experiments/config/feature_extraction.yaml`; and the removal of a first-line comment that repeated the file's path, from `config/datasets.yaml` and `config/general.yaml`.
The adapted call: `experiments/shared/analysis_utils.py` passes the bounds of `config/metrics.yaml` to `MetricNormalizer`, which no longer reads the file itself (see `src/metrics/normalizer.py` below).

**Extended**

- `src/bandit/`, `src/services/router_service.py`: the bandits accept a mask of feasible models; LinUCB also reports its shadow choice, the model it would pick if no model were masked.
- `src/api/router.py`: reads the orchestrator from the app state, and passes the orchestrator's error message to the client unchanged.
- `src/services/feature_service.py`: reports the context dimension the bandit is built with, and the context vector always has that length. The task type is the query's dataset label; no task classifier is built, so a query without a label gets an all-zero task-type block. GreenServ's paper describes a classifier, but its reported experiments also use the dataset label (see "Task Type" in GreenServ's README below).
- `src/feature_extractor/`: the context features (task type, semantic cluster, text complexity). The default complexity bins, used when the config sets none, are those of `config/feature_extraction.yaml` (GreenServ's default `[3]` gave a single one-hot slot that stayed empty for almost every query). `TaskTypeExtractor`, which this fork does not use, inherits from `BaseFeatureExtractor` like the other two extractors.
- `src/services/orchestrator.py`: feasibility filter on node telemetry, dispatch to a node, and recording the demand the placement controller reads.
- `src/services/inference_service.py`: the worker keeps several models in GPU memory, follows a target placement, and publishes telemetry. It measures energy with Zeus only, as GreenServ's paper did, and exits at startup if Zeus cannot measure its GPU.
- `src/queue/redis_client.py`: a task queue per node, node telemetry and target placements.
- `src/services/evaluation_service.py`, `src/metrics/normalizer.py`: separate reward weights `alpha` and `beta`, and the normalization bounds are looked up per dataset. The normalizer receives the bounds (`run_api.py` loads `config/metrics.yaml`) instead of reading the file itself, so a missing, invalid or empty file stops the API at startup; before, it only logged an error and every normalized value became 0.5.
- `config/models.yaml`: reduced to the 16 models GreenServ profiled, each assigned to the edge or cloud tier.
- `config/mab.yaml`: an `algorithm` key selects the bandit, with one block of constructor parameters per bandit (GreenServ's file held a single epsilon-greedy block).
- `config/feature_extraction.yaml`: complexity bins that every score falls into (with the bounds in GreenServ's file, the first slot stayed unused and the hardest texts got none). The file also names the embedding model of the semantic clustering, which GreenServ's file left to the code default (the same model).
- `requirements.txt`: Zeus 0.11, which measures the GPU without root (0.10 also needed access to the CPU's RAPL counters).
- `config/manager.py`: a missing or invalid config file raises instead of loading as empty, and `config/` is found from any working directory.

**New**

- `src/services/placement_controller.py`: the placement controller (Kalman trigger and placement program).
- `src/services/demand_counter.py`: per-model arrival rates, realized and shadow, that the orchestrator records and the placement controller reads as its demand signal.
- `src/services/node_telemetry.py`: readers of the node telemetry (free GPU memory, average latency, model size and load time on a node) and the model-size estimate they fall back to, shared by the worker, the filter and the controller.
- `run_api.py`: wires the services together and starts the API and the controller.
- `config/feasibility.yaml`: the feasibility filter's latency deadline, assumed load speed and minimum free GPU memory.
- `config/placement.yaml`, `config/placement_prior.yaml`, `scripts/placement_prior.py`: controller settings and the prior demand for its first plan.
- `tests/`: unit tests for the latency estimate of the filter, the worker, the placement controller and the HTTP API.
- `tex/`: the thesis.

**Removed**

- `src/models/registry.py` (an unused `ModelRegistry` stub) and the empty `src/api/schemas.py`: GreenServ's leftovers that nothing imported.
- From `src/metrics/energy.py`: the PyNVML measurement, the factory that chose between it and Zeus, and their base class, since the worker measures with Zeus only; and from `requirements.txt` the `pynvml` pin, which nothing imports directly any more (Zeus brings its own NVML binding).

-------------------------------------------------------------------------------------------------------------------------------------------------

# Contextual Multi-Armed Bandits for LLM Routing

Implementation of contextual multi-armed bandit algorithms for dynamic LLM routing, optimizing the accuracy-efficiency trade-off in large language model inference.

## Overview

This project explores how contextual bandits can intelligently route queries to different LLMs based on:
- Task type  
- Semantic features
- Query complexity
- Desired accuracy-energy trade-offs

**Algorithms**: Epsilon-Greedy, LinUCB, Thompson Sampling

## Quick Start

```bash
# Setup everything (database, dependencies, etc.)
make setup

# Run algorithm comparison experiment
make exp01-full

# Generate plots from existing results
make exp01-plot <timestamp>
```

## Repository Structure

```
experiments/         # Experiment implementations
├── X0_warmup/      # Algorithm warm-up analysis
├── X1_algorithm_comparison/  # Main comparison
├── X2_feature_ablation/      # Feature importance
├── X3_lambda_sweep/          # Trade-off analysis
└── X4_adaptability/          # Model pool changes

src/                # Core implementations
├── bandit/         # MAB algorithms
├── feature_extractor/  # Context extraction
└── services/       # Routing and evaluation
```

## Setup

### Prerequisites
- Python 3.10+
- Docker & docker-compose
- NVIDIA GPU with 80GB+ VRAM (A100/H100)
- 200GB storage
- HuggingFace account

### Installation

1. **Quick setup**: `make setup`

2. **Manual setup**:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   docker-compose -f db/docker-compose-db.yaml up -d
   gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U tz -d llm_db
   ```

3. **HuggingFace token**:
   ```bash
   export HF_TOKEN="token_here"
   ```

## Running Experiments

### Full Experiments
```bash
make exp01-full
make exp02-full
make exp03-full
make exp04-full
```

### Paper Figure Mapping

- **Figure 2a, 2b & 3**: `make exp01-full` → Algorithm comparison
- **Figure 4 & 5**: `make exp02-full` → Feature ablation  
- **Figure 6**: `make exp03-full` → Lambda trade-off
- **Figure 7**: `make exp04-full` → Adaptability

**Note**: The final, polished plots as they appear in the submitted paper are located in the `paper_figures/` directory. The experiment scripts will regenerate the underlying data and create similar plots in the respective `experiments/*/plots/` directories. Minor stylistic differences between the generated and final plots are expected.


## Configuration

Main parameters in `experiments/config/experiments.yaml`:
- `n_runs`: 20 (number of experiment runs)
- `samples_per_dataset`: 500 (queries per dataset)
- `lambda_weight`: Accuracy-energy trade-off

Results are saved in `experiments/*/results/`.

## Experimental Notes

- **Raw Data**: Large `detailed_results` files are excluded from the repository due to their size. The summary statistics required to generate the paper's plots are included.
- **Task Type**: The implementation includes a lightweight logistic regression classifier (`models/exemplary_task_classifier.pkl`) to determine the task type (e.g., summarization, Q&A) from a query's instruction text. This is designed for a real-world scenario where task labels are not explicitly known for the whole population of queries. However, to eliminate classification noise from the evaluation of the whole system, the experiments reported in the paper use the ground-truth task label derived directly from the source dataset of each query.
- **Latency Constraint**: The latency constraint described in the paper (Section 4.1) is a formal part of the framework but was not implemented in the experiments to isolate and focus on the accuracy-energy trade-off.
- **Sample Size**: Experiments are run on a 2500-query sample for computational feasibility.