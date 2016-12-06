import os
import pandas as pd
import util as datautil
from ..features import util


def get_filenames():
    """
    Since the HDD data is by day, we can have multiple days of data.
    :return: List of file paths to read data
    """

    data_directory = os.path.join('data', 'raw', 'hdd')

    return [os.path.join(data_directory, f) for f in os.listdir(data_directory)
            if os.path.isfile(os.path.join(data_directory, f))
            and len(os.path.splitext(f)) > 1 # Exclude '.files'
            and os.path.splitext(f)[1] == '.csv']


def read_training_data(filepaths):
    training_dataframes = []
    for data_filepath in filepaths:
        data = pd.read_csv(data_filepath)
        print 'Read {0} samples from "{1}"'.format(len(data), data_filepath)

        training_dataframes.append(data)
    training_data = pd.concat(training_dataframes)
    print "Read {0} total training samples".format(training_data.size)
    return training_data


def label_data(dataframe):
    labeled_data = dataframe.copy()
    labeled_data['label'] = 1
    labeled_data[labeled_data['failure'] != 0] = -1
    return labeled_data.drop('failure', 1)


def process_hdd(name='hdd'):
    """
    Process the HDD raw dataset.
    :param name: name of the dataset
    :return: None
    """

    data_filepaths = get_filenames()

    # Select the last file as our test data, all others are training data
    training_data_filepaths = data_filepaths[:-1]
    test_data_filepath = data_filepaths[-1]

    training_output_filepath = os.path.join('data', 'processed', '{0}_training.csv'.format(name))
    test_output_filepath = os.path.join('data', 'processed', '{0}_test.csv'.format(name))

    print 'Processing the "{0}" dataset'.format(name)

    # Process Training Data
    training_data = read_training_data(training_data_filepaths)

    # Process Test Data
    test_data = pd.read_csv(test_data_filepath)
    print 'Read {0} test samples from "{1}"'.format(test_data.size, test_data_filepath)

    print 'Labeling raw data for {0}'.format(name)
    labeled_training_data = label_data(training_data)
    labeled_test_data = label_data(test_data)

    print 'Writing processed {0} training data to "{1}"'.format(name, training_output_filepath)
    labeled_training_data.to_csv(training_output_filepath)

    print 'Writing processed {0} test data to "{1}"'.format(name, test_output_filepath)
    labeled_test_data.to_csv(test_output_filepath)

    # Write out libsvm data
    training_data_name = '{0}_training'.format(name)
    libsvm_training_data_filepath = datautil.dataframe_to_libsvm(labeled_training_data, training_data_name)
    print 'Wrote libsvm training data for {0} to "{1}"'.format(name, libsvm_training_data_filepath)

    test_data_name = '{0}_test'.format(name)
    libsvm_test_data_filepath = datautil.dataframe_to_libsvm(labeled_test_data, test_data_name)
    print 'Wrote libsvm test data for {0} to "{1}"'.format(name, libsvm_test_data_filepath)
    print ""