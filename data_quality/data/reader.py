import pandas as pd
from ..features import dataset
import os


def read_water_treatment_data(filepath=None):
    """
    Read the processed water-treatment data
    :param filepath: optional filepath (default: data/processed/water-treatment.csv)
    :return: One-Class Dataset
    """
    if filepath is None:
        filepath = os.path.join('data', 'processed', 'water-treatment.csv')

    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return dataset.OneClassDataSet(df)


def read_banknote_data(filepath=None):
    """
    Read the processed banknote data
    :param filepath: optional filepath (default: data/processed/data_banknote_authentication.csv)
    :return: One-Class Dataset
    """
    if filepath is None:
        filepath = os.path.join('data', 'processed', 'data_banknote_authentication.csv')

    df = pd.read_csv(filepath)
    return dataset.OneClassDataSet(df)


def read_hdd_training_data(filepath=None):
    """
    Read the processed hdd training data
    :param filepath: optional filepath (default: data/processed/hdd_training.csv)
    :return: One-Class Dataset
    """

    if filepath is None:
        filepath = os.path.join('data', 'processed', 'hdd_training.csv')

    df = pd.read_csv(filepath)
    return dataset.OneClassDataSet(df)


def read_hdd_test_data(filepath=None):
    """
    Read the processed hdd test data
    :param filepath: optional filepath (default: data/processed/hdd_test.csv)
    :return: One-Class Dataset
    """

    if filepath is None:
        filepath = os.path.join('data', 'processed', 'hdd_test.csv')

    df = pd.read_csv(filepath)
    return dataset.OneClassDataSet(df)
