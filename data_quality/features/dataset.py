

class OneClassDataSet(object):

    def __init__(self, dataframe, normal_label=1, anomaly_label=-1):
        self._anomalies = dataframe.loc[dataframe['label'] == anomaly_label]
        self._normal_data = dataframe.loc[dataframe['label'] == normal_label]


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