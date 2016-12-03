## Install Python Dependencies
requirements:
	pip install -r requirements.txt

## Make Dataset
data:
	python data_quality/data/make_dataset.py

## Clean compiled files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint Code
lint:
	flake8 .

.phony: requirements data clean