# Resource-Aware LLM Routing in the Edge-Cloud Continuum

A router for LLM inference across edge and cloud nodes.
For each query, a contextual bandit picks the model that best trades accuracy against energy, choosing only among the models expected to answer within a latency deadline given the current state of the nodes.
In the background, a placement controller decides which models each node keeps in GPU memory, following the demand of the router.

## About this fork

This repository is a fork of [GreenServ](https://github.com/TZData1/llm-inference-router) ([paper](https://arxiv.org/abs/2601.17551)) at commit `8e055dd`.
It holds the code and LaTeX source (`tex/`) of a bachelor thesis that extends GreenServ's router to several edge and cloud nodes.
Everything below the horizontal line is GreenServ's original README, unchanged; it describes GreenServ's offline experiments, which this fork leaves untouched.

**Inherited from GreenServ, unchanged**

- `src/feature_extractor/`, `src/metrics/`: the context features (task type, semantic cluster, text complexity) and the accuracy, energy and normalization code behind the reward.
- `db/`, `data/`, `config/metrics.yaml`, `config/datasets.yaml`: GreenServ's profiling data, the normalization bounds derived from it, and the dataset settings.
- `experiments/`, `paper_figures/`, `Makefile`: GreenServ's offline experiments X0-X4. The thesis does not reuse them.

The only edits to these files, and to GreenServ's `config/general.yaml` and `src/api/router.py`, are cosmetic: an underscore prefix on four internal methods, in `src/metrics/accuracy.py` and `data/process.py`; the removal of the redundant `datasets:` root key from `config/datasets.yaml` and the three `data/` files that read it; and the removal of a first-line comment that repeated the file's path, from `config/datasets.yaml`, `config/general.yaml` and `src/api/router.py`.

**Extended**

- `src/bandit/`, `src/services/router_service.py`: the bandits accept a mask of feasible models; LinUCB also reports its shadow choice, the model it would pick if no model were masked.
- `src/services/feature_service.py`: reports the context dimension the bandit is built with, and the context vector always has that length.
- `src/services/orchestrator.py`: feasibility filter on node telemetry, dispatch to a node, and the demand counters the placement controller reads.
- `src/services/inference_service.py`: the worker keeps several models in GPU memory, follows a target placement, and publishes telemetry.
- `src/queue/redis_client.py`: a task queue per node, node telemetry and target placements.
- `src/services/evaluation_service.py`: separate reward weights `alpha` and `beta`, and the normalization bounds are looked up per dataset.
- `config/models.yaml`: reduced to the 16 models GreenServ profiled, each assigned to the edge or cloud tier.
- `config/mab.yaml`: an `algorithm` key selects the bandit, with one block of constructor parameters per bandit (GreenServ's file held a single epsilon-greedy block).
- `config/feature_extraction.yaml`: complexity bins that every score falls into (with the bounds in GreenServ's file, the first slot stayed unused and the hardest texts got none), and the `feasibility` block: latency deadline, assumed load speed and minimum free GPU memory.
- `config/manager.py`: a missing or invalid config file raises instead of loading as empty, and `config/` is found from any working directory.

**New**

- `src/services/allocation_controller.py`: the placement controller (Kalman trigger and placement program).
- `src/services/node_telemetry.py`: readers of the node telemetry (free GPU memory, average latency, model size and load time on a node) and the model-size estimate they fall back to, shared by the worker, the filter and the controller.
- `run_api.py`: wires the services together and starts the API and the controller.
- `config/allocation.yaml`, `config/placement_prior.yaml`, `scripts/placement_prior.py`: controller settings and the prior demand for its first plan.
- `tests/`: unit tests for the latency estimate of the filter, the worker and the placement controller.
- `tex/`: the thesis.

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