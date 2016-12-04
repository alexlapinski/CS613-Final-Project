import data.reader as reader
import features.dataset as dataset
import features.util as util
from sklearn import svm
from sklearn.preprocessing import Imputer
from metrics import Metrics
from sklearn.externals import joblib
import os
import numpy as np
import pandas as pd
import filesystem as fs


def prepare_data(data):
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


def train_one_class_svm(data):

    imp, std_training_features, std_test_features, std_anomalous_features = prepare_data(data)

    # Train Model

    plot_data = []
    nu_values = []
    for nu in np.linspace(0.0001, 1, 10):
        nu_values.append(nu)
        true_positive_rates = []
        false_positive_rates = []

        gamma_values = []
        for gamma in np.linspace(0.0001, 1, 10):
            gamma_values.append(gamma)
            print "Using Gamma={0}, Nu={1}".format(gamma, nu)
            clf = svm.OneClassSVM(gamma=gamma, nu=nu)
            clf.fit(imp.transform(std_training_features))

            # Test Model
            actual_test_targets = clf.predict(imp.transform(std_test_features))
            actual_anomaly_targets = clf.predict(imp.transform(std_anomalous_features))
            test_metrics = compute_test_metrics(actual_test_targets, actual_anomaly_targets)

            true_positive_rates.append(test_metrics.compute_precision())
            false_positive_rates.append(test_metrics.compute_recall())

        data = {
            'true_positive_rates': true_positive_rates,
            'false_positive_rates': false_positive_rates
        }
        index = pd.MultiIndex.from_product([[nu], gamma_values], names=['nu', 'gamma'])
        plot_data.append(pd.DataFrame(data, index=index))

    save_plot_data(plot_data, dataset_name='water_treatment')

    return test_metrics, clf


def save_plot_data(plot_data, dataset_name):
    output_dir = os.path.join('data', 'visualizations', 'roc', dataset_name)
    fs.ensure_path_exists(output_dir)

    plot_df = pd.concat(plot_data)
    plot_df.to_csv(os.path.join(output_dir, 'one_class_svm_rbf.csv'))


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
