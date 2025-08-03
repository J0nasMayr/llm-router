# Simple Makefile for Contextual LLM Router

.PHONY: help setup clean-docker setup-venv install-deps start-db load-db stop-db clean \
        exp00 exp02 exp04 exp01 exp03 exp05 \
        exp00-full exp02-full exp04-full exp01-full exp03-full exp05-full \
        exp00-plot exp02-plot exp04-plot exp01-plot exp03-plot exp05-plot \
        clean-exp00-plots clean-exp02-plots clean-exp04-plots clean-exp01-plots clean-exp03-plots clean-exp05-plots \
        clean-all-plots \
        run-all run-all-full run-all-plot check-status

# Setup targets
setup: clean-docker setup-venv install-deps start-db load-db
	@echo "Setup complete!"

setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv

install-deps:
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

start-db:
	@echo "Starting database containers..."
	docker-compose -f db/docker-compose-db.yaml up -d
	@echo "Waiting for database to be ready..."
	@sleep 25

load-db:
	@echo "Loading database backup after sleep..."
	gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U postgres -d llm_db

stop-db:
	@echo "Stopping database containers..."
	docker-compose -f db/docker-compose-db.yaml down

clean-docker:
	@echo "Cleaning up Docker containers..."
	-docker-compose -f db/docker-compose-db.yaml down

clean: stop-db
	@echo "Cleaning up..."
	rm -rf venv
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Experiment targets (full runs)
exp00-full:
	@echo "Running A2 Warmup experiment..."
	./venv/bin/python -m experiments.00_warmup.run_experiment

exp02-full:
	@echo "Running A3 Feature Ablation experiment..."
	./venv/bin/python -m experiments.02_feature_ablation.run_experiment

exp04-full:
	@echo "Running A4 Hyperparameter Tuning experiment..."
	./venv/bin/python -m experiments.04_hyperparameter_tuning.run_experiment

exp01-full:
	@echo "Running A5 Algorithm Bakeoff experiment..."
	./venv/bin/python -m experiments.01_algorithm_comparison.run_experiment

exp03-full:
	@echo "Running A6 Lambda Sweep experiment..."
	./venv/bin/python -m experiments.03_lambda_sweep.run_experiment

exp05-full:
	@echo "Running A8 Adaptability experiment..."
	./venv/bin/python -m experiments.05_adaptability.run_experiment

# Plot-only targets (regenerate plots from existing data)
# Usage: make exp00-plot [TS=20250714_123456]
exp00-plot:
	@echo "Generating A2 Warmup plots..."
	./venv/bin/python -m experiments.00_warmup.plotting $(TS)

exp02-plot:
	@echo "Generating A3 Feature Ablation plots..."
	./venv/bin/python -m experiments.02_feature_ablation.plotting $(TS)

exp04-plot:
	@echo "Generating A4 Hyperparameter Tuning plots..."
	./venv/bin/python -m experiments.04_hyperparameter_tuning.plotting $(TS)

exp01-plot:
	@echo "Generating A5 Algorithm Bakeoff plots..."
	./venv/bin/python -m experiments.01_algorithm_comparison.plotting $(TS)

exp03-plot:
	@echo "Generating A6 Lambda Sweep plots..."
	./venv/bin/python -m experiments.03_lambda_sweep.plotting $(TS)

exp05-plot:
	@echo "Generating A8 Adaptability plots..."
	./venv/bin/python -m experiments.05_adaptability.plotting $(TS)

# Legacy shortcuts 
exp00: exp00-full
exp02: exp02-full
exp04: exp04-full
exp01: exp01-full
exp03: exp03-full
exp05: exp05-full

# Clean plot directories
clean-exp00-plots:
	@echo "Cleaning A2 plots..."
	@rm -f experiments/00_warmup/plots/*.png experiments/00_warmup/plots/*.jpg

clean-exp02-plots:
	@echo "Cleaning A3 plots..."
	@rm -f experiments/02_feature_ablation/plots/*.png experiments/02_feature_ablation/plots/*.jpg

clean-exp04-plots:
	@echo "Cleaning A4 plots..."
	@rm -f experiments/04_hyperparameter_tuning/plots/*.png experiments/04_hyperparameter_tuning/plots/*.jpg

clean-exp01-plots:
	@echo "Cleaning A5 plots..."
	@rm -f experiments/01_algorithm_comparison/plots/*.png experiments/01_algorithm_comparison/plots/*.jpg

clean-exp03-plots:
	@echo "Cleaning A6 plots..."
	@rm -f experiments/03_lambda_sweep/plots/*.png experiments/03_lambda_sweep/plots/*.jpg

clean-exp05-plots:
	@echo "Cleaning A8 plots..."
	@rm -f experiments/05_adaptability/plots/*.png experiments/05_adaptability/plots/*.jpg

# Clean all plots
clean-all-plots: clean-exp00-plots clean-exp02-plots clean-exp04-plots clean-exp01-plots clean-exp03-plots clean-exp05-plots
	@echo "All plots cleaned!"

# Run all experiments in screen sessions (super slow)
run-all-full:
	@echo "Starting all experiments in screen sessions with logging..."
	@mkdir -p logs
	@screen -L -Logfile logs/exp_00.log -dmS exp_00 ./venv/bin/python -m experiments.00_warmup.run_experiment
	@echo "Started A2 in screen session 'exp_00' (log: logs/exp_00.log)"
	@screen -L -Logfile logs/exp_02.log -dmS exp_02 ./venv/bin/python -m experiments.02_feature_ablation.run_experiment
	@echo "Started A3 in screen session 'exp_02' (log: logs/exp_02.log)"
	@screen -L -Logfile logs/exp_04.log -dmS exp_04 ./venv/bin/python -m experiments.04_hyperparameter_tuning.run_experiment
	@echo "Started A4 in screen session 'exp_04' (log: logs/exp_04.log)"
	@screen -L -Logfile logs/exp_01.log -dmS exp_01 ./venv/bin/python -m experiments.01_algorithm_comparison.run_experiment
	@echo "Started A5 in screen session 'exp_01' (log: logs/exp_01.log)"
	@screen -L -Logfile logs/exp_03.log -dmS exp_03 ./venv/bin/python -m experiments.03_lambda_sweep.run_experiment
	@echo "Started A6 in screen session 'exp_03' (log: logs/exp_03.log)"
	@screen -L -Logfile logs/exp_05.log -dmS exp_05 ./venv/bin/python -m experiments.05_adaptability.run_experiment
	@echo "Started A8 in screen session 'exp_05' (log: logs/exp_05.log)"
	@echo "All experiments started! Logs are in logs/ directory."
	@echo "Use 'screen -ls' to see sessions and 'screen -r exp_XX' to attach."

# Check status of running experiments
check-status:
	@echo "=== Running Screen Sessions ==="
	@screen -ls || echo "No screen sessions running"
	@echo ""
	@echo "Check logs in logs/ directory to see experiment progress."

# Run all plot generations
run-all-plot:
	@echo "Generating plots for all experiments..."
	@echo "=== A2 Warmup ==="
	@./venv/bin/python -m experiments.00_warmup.plotting || echo "A2 plot generation failed"
	@echo ""
	@echo "=== A3 Feature Ablation ==="
	@./venv/bin/python -m experiments.02_feature_ablation.plotting || echo "A3 plot generation failed"
	@echo ""
	@echo "=== A4 Hyperparameter Tuning ==="
	@./venv/bin/python -m experiments.04_hyperparameter_tuning.plotting || echo "A4 plot generation failed"
	@echo ""
	@echo "=== A5 Algorithm Bakeoff ==="
	@./venv/bin/python -m experiments.01_algorithm_comparison.plotting || echo "A5 plot generation failed"
	@echo ""
	@echo "=== A6 Lambda Sweep ==="
	@./venv/bin/python -m experiments.03_lambda_sweep.plotting || echo "A6 plot generation failed"
	@echo ""
	@echo "=== A8 Adaptability ==="
	@./venv/bin/python -m experiments.05_adaptability.plotting || echo "A8 plot generation failed"
	@echo ""
	@echo "All plot generation complete!"

# Legacy run-all (points to run-all-full)
run-all: run-all-full