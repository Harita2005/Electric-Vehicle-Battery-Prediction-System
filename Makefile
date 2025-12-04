.PHONY: setup simulate-data train-baseline run-dashboard deploy-infra deploy-models clean

# Local development commands
setup:
	pip install -r requirements.txt
	docker-compose up -d

simulate-data:
	python simulator/generate_data.py --vehicles 100 --years 3

train-baseline:
	python models/train_baseline.py
	python models/train_sequence.py

evaluate:
	jupyter notebook evaluation/model_evaluation.ipynb

run-dashboard:
	cd dashboard && npm start

# AWS deployment
deploy-infra:
	cd deployment && cdk deploy EVBatteryStack

deploy-models:
	python deployment/deploy_models.py

# Utilities
clean:
	docker-compose down
	rm -rf data/processed/*
	rm -rf models/artifacts/*

test:
	pytest tests/ -v

lint:
	black . && flake8 .