import math


def randomize_data(dataframe):
    """
    Randomize the input dataframe
    :param dataframe: source dataframe
    :return: randomized dataframe
    """

    random_seed = 0
    return dataframe.sample(len(dataframe), random_state=random_seed)


def split_features_target(dataframe):
    """
    Split the features and target from source dataframe.
    The target is the last column
    :param dataframe: source dataframe
    :return: (features, target)
    """

    return dataframe[dataframe.columns[:-1]], dataframe[dataframe.columns[-1]]


def split_training_data(dataframe, ratio=2.0/3.0):
    """
    Split the source dataframe into training and test dataframes using the given ration (for training data).
    :param dataframe: source dataframe
    :param ratio: percent of source data to use for training.
    :return: (training_dataframe, test_dataframe)
    """

    assert 1 >= ratio >= 0, 'ratio must be between 0.0 and 1.0, but it was "{0}"'.format(ratio)

    max_training_index = int(math.floor(len(dataframe) * ratio))
    training_data = dataframe.iloc[:max_training_index]
    test_data = dataframe.iloc[max_training_index:]
    return training_data, test_data


def standardize_data(dataframe, mean=None, std=None):
    """
    Standardize the given dataframe (subtract mean and divide by standard deviation).
    No columns or rows will be excluded except for the index and headers.
    The mean and standard deviation used in standardizing the data will be returned as well.
    :param dataframe: The dataframe containing data to standardize
    :param mean: A pre-calculated mean to use instead of calculating it (default: None)
    :param std: A pre-calculated standard deviation to use instead of calculating it (default: None)
    :return: (standardized dataframe, mean, standard deviation)
    """

    if mean is None:
        mean = dataframe.mean(axis=0)

    if std is None:
        std = dataframe.std(axis=0)

    standardized_dataframe = (dataframe - mean) / std

    return standardized_dataframe, mean, std

