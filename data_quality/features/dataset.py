

class OneClassDataSet(object):

    def __init__(self, dataframe, normal_label=1, anomaly_label=-1, max_normal_samples=None,
                 max_anomalous_samples=None):

        if max_anomalous_samples is None:
            self._anomalies = dataframe.loc[dataframe['label'] == anomaly_label]
        else:
            anomalous_samples = dataframe.loc[dataframe['label'] == anomaly_label]
            max_anomalous_samples = min(len(anomalous_samples), max_anomalous_samples)
            self._anomalies = anomalous_samples.sample(max_anomalous_samples)

        if max_normal_samples is None:
            self._normal_data = dataframe.loc[dataframe['label'] == normal_label]
        else:
            normal_samples = dataframe.loc[dataframe['label'] == normal_label]
            max_normal_samples = min(len(normal_samples), max_normal_samples)
            self._normal_data = normal_samples.sample(max_normal_samples)

    @property
    def anomalous_data(self):
        return self._anomalies

    @property
    def normal_data(self):
        return self._normal_data


class CrossValidationDataSet(object):

    def __init__(self, imputer, std_training_features, std_test_features, std_anomaly_features):
        self._imputer = imputer
        self._std_training_features = std_training_features
        self._std_test_features = std_test_features
        self._std_anomaly_features = std_anomaly_features

    @property
    def imputer(self):
        return self._imputer

    @property
    def std_training_features(self):
        return self._std_training_features

    @property
    def std_test_features(self):
        return self._std_test_features

    @property
    def std_anomaly_features(self):
        return self._std_anomaly_features