## Install Python Dependencies
requirements:
	pip install -r requirements.txt

## Make Dataset
data:
	python data_quality/data/make_dataset.py

## clean processed dataset
clean-data:
	find data/processed -name "*.csv" -exec rm {} \;

## Clean compiled files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint Code
lint:
	flake8 .


## Train model
train:
	python data_quality/train.py

.PHONY: requirements data clean clean-data train