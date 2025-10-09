# DREAM-RNN Training Pipeline Makefile

.PHONY: help setup conda-setup pip-setup gpu-setup test clean

help: ## Show this help message
	@echo "DREAM-RNN Training Pipeline Setup"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Complete setup (recommended)
	@echo "🚀 Setting up DREAM-RNN training environment..."
	python scripts/setup_environment.py

conda-setup: ## Setup using conda environment
	@echo "📦 Creating conda environment..."
	conda env create -f environment.yml
	@echo "🔧 Running GPU setup..."
	conda run -n active-learning-genomics python scripts/setup_gpu.py

pip-setup: ## Setup using pip virtual environment
	@echo "📦 Creating virtual environment..."
	python -m venv .venv
	@echo "🔧 Activating environment and installing requirements..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "🔧 Running GPU setup..."
	.venv/bin/python scripts/setup_gpu.py

gpu-setup: ## Setup GPU support only
	@echo "🔧 Setting up GPU support..."
	python scripts/setup_gpu.py

test: ## Test the installation
	@echo "🧪 Testing installation..."
	python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

data: ## Download and preprocess data
	@echo "📥 Downloading and preprocessing data..."
	dvc repro preprocess_deepstarr

train: ## Train DREAM-RNN ensemble
	@echo "🚀 Training DREAM-RNN ensemble..."
	dvc repro train_dream_rnn_ensemble

predict: ## Generate predictions
	@echo "🔮 Generating predictions..."
	dvc repro predict_ensemble

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	rm -rf models/test_ensemble
	rm -rf results/test_predictions.csv
	rm -rf data/processed
	@echo "✅ Cleanup complete"

full-pipeline: setup data train predict ## Run complete pipeline: setup → data → train → predict
	@echo "🎉 Complete pipeline finished!"

# Default target
.DEFAULT_GOAL := help
