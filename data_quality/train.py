import data.reader as reader
import features.dataset as dataset
import features.util as util
from sklearn import svm
from sklearn.preprocessing import Imputer
import os


def train_one_class_svm(data):
    randomized_data = util.randomize_data(data.normal_data)
    training_data, test_data = util.split_training_data(randomized_data)

    training_features, training_targets = util.split_features_target(training_data)
    test_features, test_targets = util.split_features_target(test_data)
    anomalous_features, anomalous_targets = util.split_features_target(data.anomalous_data)

    std_training_features, mean, std = util.standardize_data(training_features)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)
    std_anomalous_features, _, _ = util.standardize_data(anomalous_features, mean, std)

    # Impute the missing values
    # TODO: Try dropping rows with missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(std_training_features)


    clf = svm.OneClassSVM()
    clf.fit(imp.transform(std_training_features))

    y_pred_train = clf.predict(imp.transform(std_training_features))
    y_pred_test = clf.predict(imp.transform(std_test_features))
    y_pred_outliers = clf.predict(imp.transform(std_anomalous_features))
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    print "n_total_train", len(std_training_features)
    print "n_error_train", n_error_train

    print "n_total_test", len(std_test_features)
    print "n_error_test", n_error_test

    print "n_total_outliers", len(std_anomalous_features)
    print "n_error_outliers", n_error_outliers


if __name__ == "__main__":
    water_treatment_filepath = os.path.join('data', 'processed', 'water-treatment.csv')
    water_treatment_data = dataset.DataSet(reader.read_water_treatment_data(water_treatment_filepath))
    train_one_class_svm(water_treatment_data)