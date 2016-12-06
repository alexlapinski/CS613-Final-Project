## Install Python Dependencies
requirements:
	pip install -r requirements.txt

## Make Dataset
data:
	python -m data_quality.data.make_dataset

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

## Find Parameters
params:
	python -m data_quality.find_params

## Visualize Data
visualize:
	frameworkpython -m data_quality.visualize

.PHONY: requirements data clean clean-data train visualize find_params