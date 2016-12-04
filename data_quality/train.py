import data.reader as reader
import features.dataset as dataset
import features.util as util
from sklearn import svm
from sklearn.preprocessing import Imputer
from metrics import Metrics
import numpy as np
import os


def impute_missing_default_svm(std_training_features, std_test_features, std_anomalous_features, strategy, gamma, nu):
    print "Impute Missing values as {0}".format(strategy)
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=0)
    imp.fit(std_training_features)

    clf = svm.OneClassSVM(gamma=gamma, nu=nu)
    print 'One Class SVM Parameters'
    print clf.get_params()
    clf.fit(imp.transform(std_training_features))

    actual_training_targets = clf.predict(imp.transform(std_training_features))
    training_metrics = compute_training_metrics(actual_training_targets)

    actual_test_targets = clf.predict(imp.transform(std_test_features))
    actual_anomaly_targets = clf.predict(imp.transform(std_anomalous_features))
    test_metrics = compute_test_metrics(actual_test_targets, actual_anomaly_targets)

    return test_metrics


def drop_missing_default_svm(std_training_features, std_test_features, std_anomalous_features, gamma, nu):
    print "Drop Missing values"
    std_test_features = std_test_features.dropna()
    std_training_features = std_training_features.dropna()
    std_anomalous_features = std_anomalous_features.dropna()

    clf = svm.OneClassSVM(gamma=gamma)
    print 'One Class SVM Parameters'
    print clf.get_params()
    clf.fit(std_training_features)

    actual_training_targets = clf.predict(std_training_features)
    training_metrics = compute_training_metrics(actual_training_targets)

    actual_test_targets = clf.predict(std_test_features)
    actual_anomaly_targets = clf.predict(std_anomalous_features)
    test_metrics = compute_test_metrics(actual_test_targets, actual_anomaly_targets)

    return test_metrics


def train_one_class_svm(data):
    randomized_data = util.randomize_data(data.normal_data)
    training_data, test_data = util.split_training_data(randomized_data)

    training_features, training_targets = util.split_features_target(training_data)
    test_features, test_targets = util.split_features_target(test_data)
    anomalous_features, anomalous_targets = util.split_features_target(data.anomalous_data)

    std_training_features, mean, std = util.standardize_data(training_features)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)
    std_anomalous_features, _, _ = util.standardize_data(anomalous_features, mean, std)

    gamma = 0.01
    nu = 0.011
    strategy = 'most_frequent'
    metric = impute_missing_default_svm(std_training_features, std_test_features, std_anomalous_features, strategy, gamma, nu)
    print "Impute"
    print repr(metric)


def compute_training_metrics(actual_train_targets):

    # Num True positives are where the actual_test_targets were actually labeled '1'
    num_true_positives = len(actual_train_targets[actual_train_targets == 1])

    # Training data has only data of class '1', no true negatives
    num_true_negatives = 0

    # False negatives are where normal data was labeled as anomalous '-1'
    num_false_negatives = len(actual_train_targets[actual_train_targets == -1])

    # Training data has only data of class '1', no possible false positives
    num_false_positives = 0

    return Metrics(num_true_positives, num_true_negatives, num_false_positives, num_false_negatives)


def compute_test_metrics(actual_test_targets, actual_anomaly_targets):

    # Num True positives are where the actual_test_targets were actually labeled '1'
    num_true_positives = len(actual_test_targets[actual_test_targets == 1])

    # Num True negatives are where the anomalous targets were actually '-1'
    num_true_negatives = len(actual_anomaly_targets[actual_anomaly_targets == -1])

    # False negatives are where normal data was labeled as anomalous '-1'
    num_false_negatives = len(actual_test_targets[actual_test_targets == -1])

    # False positives are where anomalous data was labeled as '1'
    num_false_positives = len(actual_anomaly_targets[actual_anomaly_targets == 1])

    return Metrics(num_true_positives, num_true_negatives, num_false_positives, num_false_negatives)


if __name__ == "__main__":
    water_treatment_filepath = os.path.join('data', 'processed', 'water-treatment.csv')
    water_treatment_data = dataset.DataSet(reader.read_water_treatment_data(water_treatment_filepath))
    train_one_class_svm(water_treatment_data)