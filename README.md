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