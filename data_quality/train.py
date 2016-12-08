import json
import os

from sklearn import svm
from sklearn.preprocessing import Imputer

import pandas as pd

import data.reader as reader
import features.util as util
from metrics import Metrics
import filesystem as fs


def train_one_class_svm(data, kernel, nu, gamma, degree, coef0, test_dataset=None):

    # Reset our default params
    if gamma is None:
        gamma = 1.0/len(data.normal_data.columns)

    if degree is None:
        degree = 3

    if coef0 is None:
        coef0 = 0.0

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    # Prepare Training Data
    if test_dataset is None:
        training_data, test_data = util.split_training_data(data.normal_data)
    else:
        # Use pre-supplied training data if indicated
        training_data = data.normal_data
        test_data = test_dataset.normal_data

    training_features, training_targets = util.split_features_target(training_data)
    test_features, test_targets = util.split_features_target(test_data)

    if test_dataset is None:
        anomalous_features, anomalous_targets = util.split_features_target(data.anomalous_data)
    else:
        anomalous_features, anomalous_targets = util.split_features_target(test_dataset.anomalous_data)

    std_training_features, mean, std = util.standardize_data(training_features)

    imp.fit(std_training_features)

    # Train Model
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree, coef0=coef0)
    print "Training using {0} samples".format(len(std_training_features))
    clf.fit(imp.transform(std_training_features))

    # Prepare Test Data (standardize & impute missing values)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)
    std_anomalous_features, _, _ = util.standardize_data(anomalous_features, mean, std)

    # Test Model
    print "Testing using {0} normal samples".format(len(std_test_features))
    actual_test_targets = clf.predict(imp.transform(std_test_features))

    print "Testing using {0} anomalous samples".format(len(std_anomalous_features))
    actual_anomaly_targets = clf.predict(imp.transform(std_anomalous_features))

    return actual_test_targets, actual_anomaly_targets


def compute_metrics(actual_normal_targets, actual_anomaly_targets):

    # Num True positives are where the normal was actually labeled '1'
    num_true_positives = len(actual_normal_targets[actual_normal_targets == 1])

    # Num True negatives are where the actual_anomaly_targets were actually assigned '-1'
    num_true_negatives = len(actual_anomaly_targets[actual_anomaly_targets == -1])

    # False negatives are where normal data was labeled as anomalous '-1'
    num_false_negatives = len(actual_normal_targets[actual_normal_targets == -1])

    # Num False Positives are where anomaly targets were assigned '1'
    num_false_positives = len(actual_anomaly_targets[actual_anomaly_targets == 1])

    return Metrics(num_true_positives, num_true_negatives, num_false_positives, num_false_negatives)


def read_params_file(filepath):

    with open(filepath, 'r') as data_file:
        data = json.load(data_file)

    return data


def try_get_value(dictionary, key):
    try:
        return float(dictionary[key])
    except:
        return None


def execute_training(data=None, name=None, test_data=None):
    params_filepath = os.path.join('models', 'parameters', name, 'one_class_svm.json')
    params = read_params_file(params_filepath)

    # Execute SVM (for each kernel)
    print "# Executing One-Class SVM for {0}".format(name)
    combined_metrics = []
    for kernel, kernel_params in params.items():
        print "## Using '{0}' Kernel".format(kernel)
        print "Parameters = {0}\n".format(kernel_params)

        if kernel == 'polynomial':
            kernel = 'poly'

        nu = try_get_value(kernel_params, 'nu')
        gamma = try_get_value(kernel_params, 'gamma')
        degree = try_get_value(kernel_params, 'degree')
        coef0 = try_get_value(kernel_params, 'coef')

        actual_test_targets, actual_anomaly_targets = train_one_class_svm(data,
                                                                          kernel, nu, gamma, degree, coef0,
                                                                          test_data)

        # Analyze Results
        metrics = compute_metrics(actual_test_targets, actual_anomaly_targets)
        print " * Accuracy: {0}".format(metrics.compute_accuracy())
        print " * F1 Measure: {0}".format(metrics.compute_f1())
        print " * Precision: {0}".format(metrics.compute_precision())
        print " * Recall / TPR: {0}".format(metrics.compute_recall())
        print " * FPR: {0}".format(metrics.compute_false_positive_rate())
        print "\n"

        combined_metrics.append(pd.Series({
            'accuracy': metrics.compute_accuracy(),
            'f1-measure': metrics.compute_f1(),
            'precision': metrics.compute_precision(),
            'recall': metrics.compute_recall(),
            'false-positive-rate': metrics.compute_false_positive_rate()
        }))

    # Save Metrics to Dataset / print latex table
    output_directory = os.path.join('models', 'results', name)
    fs.ensure_path_exists(output_directory)
    metrics_filepath = os.path.join(output_directory, 'one_class_svm.csv')
    metrics_df = pd.DataFrame(combined_metrics)
    metrics_df.to_csv(metrics_filepath)
    print "Wrote Combined metrics for {0} to '{1}'\n\n".format(name, metrics_filepath)


if __name__ == "__main__":
    data = reader.read_water_treatment_data()
    execute_training(data, 'water-treatment')

    data = reader.read_banknote_data()
    execute_training(data, 'banknote')

    training_data = reader.read_hdd_training_data(max_normal_samples=10000)
    test_data = reader.read_hdd_test_data(max_normal_samples=10000)
    execute_training(training_data, 'hdd', test_data=test_data)
