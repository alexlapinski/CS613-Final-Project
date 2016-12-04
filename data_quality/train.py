import data.reader as reader
import features.dataset as dataset
import features.util as util
from sklearn import svm
from sklearn.preprocessing import Imputer
from metrics import Metrics
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


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

    # Observed 'best'
    gamma = 0.01

    # Observed 'best'
    nu = 0.011

    best_nu = 0
    best_gamma = 0

    best_recall = 0
    for nu in np.linspace(0.0001, 1, 100):

        precision = []
        recall = []

        for gamma in np.linspace(0.0001, 1, 100):
            clf = svm.OneClassSVM(gamma=gamma, nu=nu)
            clf.fit(imp.transform(std_training_features))

            # Test Model
            actual_test_targets = clf.predict(imp.transform(std_test_features))
            actual_anomaly_targets = clf.predict(imp.transform(std_anomalous_features))
            test_metrics = compute_test_metrics(actual_test_targets, actual_anomaly_targets)

            if test_metrics.compute_recall() > best_recall:
                best_gamma = gamma
                best_nu = nu
                best_recall = test_metrics.compute_recall()

            precision.append(test_metrics.compute_precision())
            recall.append(test_metrics.compute_recall())

        plt.plot(precision, recall, label='nu={0}'.format(nu))

    plt.plot(1.0, best_recall, label="Best Gamma={0}/Nu={1}".format(best_gamma, best_nu), marker='o', ms=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', fontsize='xx-small')
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title('One Class SVM; vary gamma, rbf kernel')
    plt.savefig('reports/figures/water-treatment/one_class_svm_rbf.png')

    return test_metrics, clf


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
