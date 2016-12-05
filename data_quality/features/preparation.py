from sklearn.preprocessing import Imputer
from sklearn.model_selection import ShuffleSplit
from dataset import CrossValidationDataSet
import util


def prepare_water_treatment_data(data):
    # Randomize Data
    randomized_data = util.randomize_data(data.normal_data)

    # Split Testing / Training Data
    training_data, test_data = util.split_training_data(randomized_data)

    # Split Features / Targets
    training_features, training_targets = util.split_features_target(training_data)
    test_features, test_targets = util.split_features_target(test_data)
    anomalous_features, anomalous_targets = util.split_features_target(data.anomalous_data)

    # Standardize Data
    std_training_features, mean, std = util.standardize_data(training_features)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)
    std_anomalous_features, _, _ = util.standardize_data(anomalous_features, mean, std)

    # Impute missing values
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(std_training_features)

    return imp, std_training_features, std_test_features, std_anomalous_features


def create_cross_validation_data(data, num_folds=4):
    """
    Create num_folds cross-validation datasets
    :param data:
    :param num_folds:
    :return:
    """

    randomized_data = util.randomize_data(data.normal_data)
    randomized_features, _ = util.split_features_target(randomized_data)

    anomaly_features, _ = util.split_features_target(data.anomalous_data)

    datasets = []

    shuffler = ShuffleSplit(n_splits=num_folds, train_size=1/float(num_folds), test_size=None, random_state=0)

    for train_index, test_index in shuffler.split(randomized_features):

        training_data = randomized_data.iloc[train_index]
        test_data = randomized_data.iloc[test_index]

        imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        std_training_data, mean, std = util.standardize_data(training_data)

        imputer.fit(std_training_data)
        std_training_data = imputer.transform(std_training_data)

        std_test_data, _, _ = util.standardize_data(test_data, mean, std)
        std_test_data = imputer.transform(std_test_data)

        std_anomaly_data, _, _ = util.standardize_data(anomaly_features, mean, std)
        std_anomaly_data = imputer.transform(std_anomaly_data)

        datasets.append(CrossValidationDataSet(imputer, std_training_data, std_test_data, std_anomaly_data))

    return datasets
