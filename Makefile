# Simple Makefile for Contextual LLM Router

.PHONY: help setup clean-docker setup-venv install-deps start-db load-db stop-db clean \
        exp00 exp01 exp02 exp03 exp04 \
        exp00-full exp01-full exp02-full exp03-full exp04-full \
        exp00-plot exp01-plot exp02-plot exp03-plot exp04-plot \
        clean-exp00-plots clean-exp01-plots clean-exp02-plots clean-exp03-plots clean-exp04-plots \
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
	gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U tz -d llm_db

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
	@echo "Running Warmup experiment..."
	./venv/bin/python -m experiments.X0_warmup.run_experiment
	
exp01-full:
	@echo "Running Algorithm Comparison experiment..."
	./venv/bin/python -m experiments.X1_algorithm_comparison.run_experiment

exp02-full:
	@echo "Running Feature Ablation experiment..."
	./venv/bin/python -m experiments.X2_feature_ablation.run_experiment

exp03-full:
	@echo "Running Lambda Sweep experiment..."
	./venv/bin/python -m experiments.X3_lambda_sweep.run_experiment

exp04-full:
	@echo "Running Adaptability experiment..."
	./venv/bin/python -m experiments.X4_adaptability.run_experiment

# Plot-only targets (regenerate plots from existing data)
# Usage: make exp00-plot
exp00-plot:
	@echo "Generating Warmup plots..."
	./venv/bin/python -m experiments.X0_warmup.plotting $(TS)

exp01-plot:
	@echo "Generating Algorithm Comparison plots..."
	./venv/bin/python -m experiments.X1_algorithm_comparison.plotting $(TS)

exp02-plot:
	@echo "Generating Feature Ablation plots..."
	./venv/bin/python -m experiments.X2_feature_ablation.plotting $(TS)

exp03-plot:
	@echo "Generating Lambda Sweep plots..."
	./venv/bin/python -m experiments.X3_lambda_sweep.plotting $(TS)

exp04-plot:
	@echo "Generating Adaptability plots..."
	./venv/bin/python -m experiments.X4_adaptability.plotting $(TS)

# Clean plot directories
clean-exp00-plots:
	@echo "Cleaning exp00 plots..."
	@rm -f experiments/X0_warmup/plots/*.png experiments/X0_warmup/plots/*.jpg

clean-exp01-plots:
	@echo "Cleaning exp01 plots..."
	@rm -f experiments/X1_algorithm_comparison/plots/*.png experiments/X1_algorithm_comparison/plots/*.jpg

clean-exp02-plots:
	@echo "Cleaning exp02 plots..."
	@rm -f experiments/X2_feature_ablation/plots/*.png experiments/X2_feature_ablation/plots/*.jpg

clean-exp03-plots:
	@echo "Cleaning exp03 plots..."
	@rm -f experiments/X3_lambda_sweep/plots/*.png experiments/X3_lambda_sweep/plots/*.jpg

clean-exp04-plots:
	@echo "Cleaning exp04 plots..."
	@rm -f experiments/X4_adaptability/plots/*.png experiments/X4_adaptability/plots/*.jpg

# Clean all plots
clean-all-plots: clean-exp00-plots clean-exp01-plots clean-exp02-plots clean-exp03-plots clean-exp04-plots
	@echo "All plots cleaned!"

# Run all experiments in screen sessions (super slow)
run-all-full:
	@echo "Starting all experiments in screen sessions with logging..."
	@mkdir -p logs
	@screen -L -Logfile logs/exp_00.log -dmS exp_00 ./venv/bin/python -m experiments.X0_warmup.run_experiment
	@echo "Started Warmup in screen session 'exp_00' (log: logs/exp_00.log)"
	@screen -L -Logfile logs/exp_01.log -dmS exp_01 ./venv/bin/python -m experiments.X1_algorithm_comparison.run_experiment
	@echo "Started Algorithm Comparison in screen session 'exp_01' (log: logs/exp_01.log)"
	@screen -L -Logfile logs/exp_02.log -dmS exp_02 ./venv/bin/python -m experiments.X2_feature_ablation.run_experiment
	@echo "Started Feature Ablation in screen session 'exp_02' (log: logs/exp_02.log)"
	@screen -L -Logfile logs/exp_03.log -dmS exp_03 ./venv/bin/python -m experiments.X3_lambda_sweep.run_experiment
	@echo "Started Lambda Sweep in screen session 'exp_03' (log: logs/exp_03.log)"
	@screen -L -Logfile logs/exp_04.log -dmS exp_04 ./venv/bin/python -m experiments.X4_adaptability.run_experiment
	@echo "Started Adaptability in screen session 'exp_04' (log: logs/exp_04.log)"
	@echo "All experiments started! Logs are in logs/ directory."
	@echo "Use 'screen -ls' to see sessions and 'screen -r exp_XX' to attach."

# Check status of running experiments
check-status:
	@echo "=== Running Screen Sessions ==="
	@screen -ls || echo "No screen sessions running"
	@echo ""
	@echo "Check logs in logs/ directory to see experiment progress."
