import os
import pandas as pd
import util as datautil
from ..features import util

__columns = [
    'variance of Wavelet Transformed image',
    'skewness of Wavelet Transformed image',
    'curtosis of Wavelet Transformed image',
    'entropy of image',
    'label'
]


def process_banknote(name='data_banknote_authentication'):
    """
    Process the banknote raw dataset.
    :param name: name of the dataset
    :return: None
    """
    data_filepath = os.path.join('data', 'raw', '{0}.txt'.format(name))
    output_filepath = os.path.join('data', 'processed', '{0}.csv'.format(name))

    print 'Processing the "{0}" dataset'.format(name)

    data = pd.read_csv(data_filepath, names=__columns)
    print 'Read {0} samples from "{1}"'.format(len(data), data_filepath)

    print 'Labeling raw data for {0}'.format(name)
    labeled_data = data.copy()
    labeled_data[data['label'] == 0] = -1

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