from data_quality.data import reader as reader
from data_quality.features import dataset as dataset
from data_quality.features import preparation as prep
from metrics import Metrics
from sklearn import svm
import numpy as np
import os
import math
import time
import filesystem as fs
import json


def compute_test_metrics(actual_test_targets, actual_anomaly_targets):

    # Num True positives are where the actual_test_targets were actually labeled '1'
    num_true_positives = len(actual_test_targets[actual_test_targets == 1])

    # Num True negatives are where the anomalous targets were actually '-1'
    num_true_negatives = len(actual_anomaly_targets[actual_anomaly_targets == -1])

    # False negatives are where normal data was labeled as anomalous '-1'
    num_false_negatives = len(actual_test_targets[actual_test_targets == -1])

    # False positives are where anomalous data was labeled as '1'
    num_false_positives = len(actual_anomaly_targets[actual_anomaly_targets == 1])

    return Metrics(num_true_positives, num_true_negatives,
                   num_false_positives, num_false_negatives)


def execute_algorithm(data, kernel, nu, gamma='auto', degree=3, coef0=0.0):
    """
    Execute the One-Class SVM (Training and Testing)
    :param data: Entire dataset
    :param kernel: kernel ('linear', 'poly', 'rbf', or 'sigmoid')
    :param nu: nu parameter for one-class svm
    :param gamma:
    :param degree:
    :param coef0:
    :return: metrics from evaluation
    """

    # Train
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree, coef0=coef0,
                          cache_size=200)
    clf.fit(data.std_training_features)

    # Test
    actual_test_targets = clf.predict(data.std_test_features)
    actual_anomaly_targets = clf.predict(data.std_anomaly_features)
    return compute_test_metrics(actual_test_targets, actual_anomaly_targets)


def compute_final_metric(metrics):
    """
    Compute Average Precision, Recall and F1Measure
    :param metrics:
    :return:
    """

    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0

    for metric in metrics:
        avg_precision += metric.compute_precision()
        avg_recall += metric.compute_recall()
        avg_f1 += metric.compute_f1()

    num_metrics = float(len(metrics))
    avg_precision /= num_metrics
    avg_recall /= num_metrics
    avg_f1 /= num_metrics

    return avg_precision, avg_recall, avg_f1


def search_params_rbf(datasets, nu_values, gamma_values):
    print "Searching for RBF Parameters"
    #
    # Use f1 metric (based on work in 'One-Class SVMs for Document Classification' by Larry M. Manevitz et. al.)
    #

    # Test RBF Kernel, varying parameters
    best_f1 = 0
    best_parameters = {'nu': 0, 'gamma': 0}
    for nu in nu_values:
        for gamma in gamma_values:
            metrics = []
            for dataset in datasets:
                metrics.append(execute_algorithm(dataset, 'rbf', nu, gamma))
            precision, recall, f1 = compute_final_metric(metrics)

            if f1 > best_f1:
                best_f1 = f1
                best_parameters = {'nu': nu, 'gamma': gamma}

    print 'Best Parameters w/ f1 = {0}: {1}'.format(best_f1, best_parameters)
    return best_parameters


def search_params_linear(datasets, nu_values):
    print "Searching for Linear Parameters"
    #
    # Use f1 metric (based on work in 'One-Class SVMs for Document Classification' by Larry M. Manevitz et. al.)
    #

    # Test RBF Kernel, varying parameters
    best_f1 = 0
    best_parameters = {'nu': 0}
    for nu in nu_values:
        metrics = []
        for dataset in datasets:
            metrics.append(execute_algorithm(dataset, 'linear', nu))
        precision, recall, f1 = compute_final_metric(metrics)

        if f1 > best_f1:
            best_f1 = f1
            best_parameters = {'nu': nu}

    print 'Best Parameters w/ f1 = {0}: {1}'.format(best_f1, best_parameters)
    return best_parameters


def search_params_poly(datasets, nu_values, gamma_values, degree_values, coef_values):
    print "Searching for Polynomial Parameters"
    #
    # Use f1 metric (based on work in 'One-Class SVMs for Document Classification' by Larry M. Manevitz et. al.)
    #

    best_f1 = 0
    best_parameters = {'nu': 0, 'gamma': 0, 'degree': 0, 'coef': 0}
    for degree in degree_values:
        for nu in nu_values:
            for gamma in gamma_values:
                for coef in coef_values:
                    metrics = []
                    i = 0
                    for dataset in datasets:
                        metrics.append(execute_algorithm(dataset, 'poly', nu, gamma, degree, coef))
                        i += 1
                    precision, recall, f1 = compute_final_metric(metrics)

                    if f1 > best_f1 and f1 != float('inf'):
                        best_f1 = f1
                        best_parameters = {'nu': nu,
                                           'gamma': gamma,
                                           'degree': degree,
                                           'coef': coef}

    print 'Best Parameters w/ f1 = {0}: {1}'.format(best_f1, best_parameters)
    return best_parameters


def search_params_sigmoid(datasets, nu_values, gamma_values, coef_values):
    print "Searching for Sigmoid Parameters"
    #
    # Use f1 metric (based on work in 'One-Class SVMs for Document Classification' by Larry M. Manevitz et. al.)
    #

    best_f1 = 0
    best_parameters = {'nu': 0, 'gamma': 0, 'coef': 0}
    for nu in nu_values:
        for gamma in gamma_values:
            for coef in coef_values:
                metrics = []
                for dataset in datasets:
                    metrics.append(execute_algorithm(dataset, 'sigmoid', nu, gamma, coef))
                precision, recall, f1 = compute_final_metric(metrics)

                if f1 > best_f1:
                    best_f1 = f1
                    best_parameters = {'nu': nu,
                                       'gamma': gamma,
                                       'coef': coef}

    print 'Best Parameters w/ f1 = {0}: {1}'.format(best_f1, best_parameters)
    return best_parameters


def iterate_search_sigmoid(coef_from_exp, coef_to_exp, datasets, gamma_from_exp, gamma_to_exp,
                           nu_from_exp, nu_to_exp, num_iterations, search_size):
    best_params = None

    for i in xrange(num_iterations):
        nu_values = np.logspace(nu_from_exp, nu_to_exp, num=search_size, base=2)
        gamma_values = np.logspace(gamma_from_exp, gamma_to_exp, num=search_size, base=2)
        coef_values = np.logspace(coef_from_exp, coef_to_exp, num=search_size, base=2)
        print "Cross Validate, nu:2^{0}->2^{1};gamma:2^{2}->2^{3};coef:2^{4}->2^{5}" \
            .format(nu_from_exp, nu_to_exp, gamma_from_exp, gamma_to_exp,
                    coef_from_exp, coef_to_exp)
        best_params = search_params_sigmoid(datasets, nu_values, gamma_values,
                                            coef_values)
        nu_exponent = math.log(best_params['nu'], 2)
        nu_from_exp = nu_exponent - 1
        nu_to_exp = nu_exponent + 1

        gamma_exponent = math.log(best_params['gamma'], 2)
        gamma_from_exp = gamma_exponent - 1
        gamma_to_exp = gamma_exponent + 1

        coef_exponent = math.log(best_params['coef'], 2)
        coef_from_exp = coef_exponent - 1
        coef_to_exp = coef_exponent + 1

    return best_params


def iterate_search_polynomial(coef_from_exp, coef_to_exp, datasets, degree_values, gamma_from_exp,
                              gamma_to_exp, nu_from_exp, nu_to_exp, num_iterations, search_size):

    best_params = None

    degree_from = degree_values[0]
    degree_to = degree_values[-1]

    for i in xrange(num_iterations):
        nu_values = np.logspace(nu_from_exp, nu_to_exp, num=search_size, base=2)
        gamma_values = np.logspace(gamma_from_exp, gamma_to_exp, num=search_size, base=2)
        coef_values = np.logspace(coef_from_exp, coef_to_exp, num=search_size, base=2)
        print "Cross Validate, nu:2^{0}->2^{1};gamma:2^{2}->2^{3};degree:{4}->{5};coef:2^{6}->2^{7}" \
            .format(nu_from_exp, nu_to_exp, gamma_from_exp, gamma_to_exp,
                    degree_from, degree_to, coef_from_exp, coef_to_exp)
        best_params = search_params_poly(datasets, nu_values, gamma_values,
                                         degree_values, coef_values)
        nu_exponent = math.log(best_params['nu'], 2)
        nu_from_exp = nu_exponent - 1
        nu_to_exp = nu_exponent + 1

        gamma_exponent = math.log(best_params['gamma'], 2)
        gamma_from_exp = gamma_exponent - 1
        gamma_to_exp = gamma_exponent + 1

        degree = best_params['degree']
        degree_from = degree - 1
        degree_to = degree + 1
        degree_values = np.arange(degree_from, degree_to+1, step=1)

        coef_exponent = math.log(best_params['coef'], 2)
        coef_from_exp = coef_exponent - 1
        coef_to_exp = coef_exponent + 1

    return best_params


def iterate_search_linear(datasets, nu_from_exp, nu_to_exp, num_iterations, search_size):
    best_params = None
    for i in xrange(num_iterations):
        nu_values = np.logspace(nu_from_exp, nu_to_exp, num=search_size, base=2)
        print "Cross Validate, nu:2^{0}->2{1}".format(nu_from_exp, nu_to_exp)
        best_params = search_params_linear(datasets, nu_values)
        nu_exponent = math.log(best_params['nu'], 2)
        nu_from_exp = nu_exponent - 1
        nu_to_exp = nu_exponent + 1

    return best_params


def iterate_search_rbf(datasets, gamma_from_exp, gamma_to_exp, nu_from_exp, nu_to_exp, num_iterations, search_size):
    best_params = None

    for i in xrange(num_iterations):
        nu_values = np.logspace(nu_from_exp, nu_to_exp, num=search_size, base=2)
        gamma_values = np.logspace(gamma_from_exp, gamma_to_exp, num=search_size, base=2)
        print "Cross Validate, nu:2^{0}->2^{1}; gamma:2^{2}->2^{3} search w/ rbf".format(nu_from_exp, nu_to_exp,
                                                                                         gamma_from_exp, gamma_to_exp)
        best_params = search_params_rbf(datasets, nu_values, gamma_values)
        nu_exponent = math.log(best_params['nu'], 2)
        nu_from_exp = nu_exponent - 1
        nu_to_exp = nu_exponent + 1

        gamma_exponent = math.log(best_params['gamma'], 2)
        gamma_from_exp = gamma_exponent - 1
        gamma_to_exp = gamma_exponent + 1

    return best_params


def save_params(name, rbf_params, linear_params, poly_params, sigmoid_params):
    """
    Save the parameters for each kernel used in one-class svm
    :param name: Name of the source dataset
    :param rbf_params: RBF Kernel Parameters (as dictionary)
    :param linear_params: Linear Kernel Parameters (as dictionary)
    :param poly_params: Polynomial Kernel Parameters (as dictionary)
    :param sigmoid_params: Sigmoid Kernel Parameters (as dictionary)
    :return: filepath of saved file.
    """

    out_directory = os.path.join('models', 'parameters', name)

    fs.ensure_path_exists(out_directory)

    out_filepath = os.path.join(out_directory, 'one_class_svm.json')

    params = {
        'rbf': rbf_params,
        'linear': linear_params,
        'polynomial': poly_params,
        'sigmoid': sigmoid_params
    }

    with open(out_filepath, 'w') as out_file:
        out_file.write(json.dumps(params, sort_keys=True, indent=4, separators=(',', ':')))

    return out_filepath


def search_params(data, name, training_set_size=50):
    """
    Use the given dataset and search for optimal parameters
    :param data: One-Class Dataset
    :param name: name of the dataset for saving params
    :param training_set_size: size of each training dataset
    :return: nothing
    """
    num_iterations = 10
    search_size = 3

    # Pick a num_folds that gives us at least 'training_set_size' in a training set
    num_folds = len(data.normal_data) / training_set_size
    print "Using {0} Folds for Cross Validation".format(num_folds)

    # Create our Cross-Validation datasets
    datasets = prep.create_cross_validation_data(data, num_folds)

    # NuValues and GammaValues initial range taken from libsvm 'guide' pdf
    nu_from_exp = -2
    nu_to_exp = 0

    gamma_from_exp = -15
    gamma_to_exp = 3

    # Degree - hand selected (anything larger and python chokes)
    degree_values = [1, 2, 3]

    # Degree small to moderate size
    coef_from_exp = -15
    coef_to_exp = 3

    # Search for RBF Params, tune 'nu' and 'gamma'
    start_time = time.time()
    rbf_params = iterate_search_rbf(datasets, gamma_from_exp, gamma_to_exp,
                                    nu_from_exp, nu_to_exp, num_iterations, search_size)
    print "Time Elapsed: {0}\n".format(time.time() - start_time)

    params_filepath = save_params(name, rbf_params, None, None, None)
    print "Wrote Best Parameters to '{0}'".format(params_filepath)

    # Use linear kernel, tune 'nu' only
    start_time = time.time()
    linear_params = iterate_search_linear(datasets, nu_from_exp, nu_to_exp,
                                          num_iterations, search_size)
    print "Time Elapsed: {0}\n".format(time.time() - start_time)

    params_filepath = save_params(name, rbf_params, linear_params, None, None)
    print "Wrote Best Parameters to '{0}'".format(params_filepath)

    # Use Sigmoid Kernel
    # tune nu, gamma and coef
    start_time = time.time()
    sigmoid_params = iterate_search_sigmoid(coef_from_exp, coef_to_exp, datasets,
                                            gamma_from_exp, gamma_to_exp,
                                            nu_from_exp, nu_to_exp, num_iterations, search_size)
    print "Time Elapsed: {0}\n".format(time.time() - start_time)

    params_filepath = save_params(name, rbf_params, linear_params, None, sigmoid_params)
    print "Wrote Best Parameters to '{0}'".format(params_filepath)

    # Use Polynomial Kernel
    # tune nu, gamma, degree and coef
    start_time = time.time()
    poly_params = iterate_search_polynomial(coef_from_exp, coef_to_exp,
                                            datasets, degree_values,
                                            gamma_from_exp, gamma_to_exp, nu_from_exp,
                                            nu_to_exp, num_iterations, search_size)
    print "Time Elapsed: {0}\n".format(time.time() - start_time)

    params_filepath = save_params(name, rbf_params, linear_params, poly_params, sigmoid_params)
    print "Wrote Best Parameters to '{0}'".format(params_filepath)


if __name__ == "__main__":
    print "Searching for Parameters for Water-Treatment Plant data"
    water_treatment_data = reader.read_water_treatment_data()
    search_params(water_treatment_data, name='water-treatment')

    print "Searching for Parameters for Banknote data"
    banknote_data = reader.read_banknote_data()
    search_params(banknote_data, name='banknote')

    print "Searching for Parameters for HDD data"
    hdd_data = reader.read_hdd_training_data()
    search_params(hdd_data, name='hdd')
