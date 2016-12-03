import pandas as pd
import re


def process_water_treatment():
    # Open .columns file
    columns_file = open('data/interim/water-treatment.columns', 'r')

    # Read Columns
    columns = []

    column_pattern = re.compile(r'\s*(?P<index>\s*\d+)\s*(?P<name>.+?)\s*\((?P<description>.*)\)\s*')
    for line in columns_file:
        match = column_pattern.match(line)
        if match is not None:
            columns.append(match.group('name'))

    # Read Data
    data = pd.read_csv('data/raw/water-treatment.data', names=columns)
    print data.head(2)

    # TODO: Label Data

    # Write to Processed
    print 'Writing processed water-treatment data'
    data.to_csv('data/processed/water-treatment.csv')


if __name__ == "__main__":
    process_water_treatment()