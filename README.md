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
make exp01-plot
```

## Repository Structure

```
experiments/         # Experiment implementations
├── 00_warmup/      # Algorithm warm-up analysis
├── 01_algorithm_comparison/  # Main comparison
├── 02_feature_ablation/      # Feature importance
├── 03_lambda_sweep/          # Trade-off analysis
├── 04_hyperparameter_tuning/ # Parameter sensitivity
└── 05_adaptability/          # Model pool changes

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
- ~100GB storage
- HuggingFace account

### Installation

1. **Quick setup**: `make setup`

2. **Manual setup**:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   docker-compose -f db/docker-compose-db.yaml up -d
   gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U postgres -d llm_db
   ```

3. **HuggingFace token**:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## Running Experiments

### Full Experiments
```bash
make exp01-full  # Algorithm comparison (~3h on A100)
make exp02-full  # Feature ablation (~2h)
make exp03-full  # Lambda sweep (~2h)
make exp04-full  # Hyperparameter tuning (~4h)
make exp05-full  # Adaptability (~1h)
```

### Paper Figure Mapping
- **Figure 3**: `make exp01-full` → Algorithm comparison
- **Figure 4**: `make exp02-full` → Feature ablation  
- **Figure 5**: `make exp03-full` → Lambda trade-off
- **Figure 6**: `make exp05-full` → Adaptability


## Configuration

Main parameters in `experiments/config/experiments.yaml`:
- `n_runs`: 20 (number of experiment runs)
- `samples_per_dataset`: 500 (queries per dataset)
- `lambda_weight`: Accuracy-energy trade-off

## Results

- Contextual bandits outperform non-contextual baselines by 15-20%
- LinUCB achieves best exploration-exploitation balance
- Task type and complexity are most informative features

Results are saved in `experiments/*/results/` with timestamp.

## Notes

- We use dataset names as task labels for experimental clarity (see paper Section 4.2)
- Sample size limited to 2500 queries for computational feasibility