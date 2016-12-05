import re
import pandas as pd
import os
import classify_water_treatment
import util as datautil
from .. import filesystem as fs
from ..features import util


def __read_columns(filepath):
    """
    Read a list of columns where each line is in the format:
    INDEX NAME (Description), where INDEX is an integer, NAME is a string and Description is a string
    :param filepath: filepath of the columns file
    :return: list of column names
    """
    columns = []
    column_pattern = re.compile(r'\s*(?P<index>\s*\d+)\s*(?P<name>.+?)\s*\((?P<description>.*)\)\s*')
    with open(filepath) as columns_file:
        for line in columns_file:
            match = column_pattern.match(line)
            if match is None:
                print 'ERROR: "{0}" is not in the correct format'.format(line)
            else:
                columns.append(match.group('name'))

    print 'Parsed {0} columns from "{1}"'.format(len(columns), filepath)
    return columns


def process_water_treatment(name='water-treatment'):
    """
    Process the water treatment raw dataset.
    :param name: name of the dataset
    :return: None
    """
    columns_filepath = os.path.join('data', 'interim', '{0}.columns'.format(name))
    data_filepath = os.path.join('data', 'raw', '{0}.data'.format(name))
    output_filepath = os.path.join('data', 'processed', '{0}.csv'.format(name))

    print 'Processing the "{0}" dataset'.format(name)
    columns = __read_columns(columns_filepath)

    def date_parser(date):
        return pd.datetime.strptime(date, 'D-%d/%m/%y')
    data = pd.read_csv(data_filepath, names=columns, na_values='?',
                       parse_dates=True, date_parser=date_parser)
    print 'Read {0} samples from "{1}"'.format(len(data), data_filepath)

    print 'Labeling raw data for {0}'.format(name)
    labeled_data = classify_water_treatment.classify(data)

    print 'Writing processed {0} data to "{1}"'.format(name, output_filepath)
    labeled_data.to_csv(output_filepath)

    libsvm_data_filepath = datautil.dataframe_to_libsvm(labeled_data, name)
    print 'Wrote libsvm all data for {0} to "{1}"'.format(name, libsvm_data_filepath)

    # Split anomalous data and test data to separate dataframe
    normal_data = labeled_data[labeled_data['label'] == 1]
    anomalous_data = labeled_data[labeled_data['label'] == -1]

    training_data, test_data = util.split_training_data(normal_data)
    test_data = pd.concat([test_data, anomalous_data])

    training_data_name = '{0}_training'.format(name)
    libsvm_training_data_filepath = datautil.dataframe_to_libsvm(training_data, training_data_name)
    print 'Wrote libsvm training data for {0} to "{1}"'.format(name, libsvm_training_data_filepath)

    test_data_name = '{0}_test'.format(name)
    libsvm_test_data_filepath = datautil.dataframe_to_libsvm(test_data, test_data_name)
    print 'Wrote libsvm test data for {0} to "{1}"'.format(name, libsvm_test_data_filepath)
    print ""
