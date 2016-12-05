import pandas as pd


def read_water_treatment_data(filepath):
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return df
