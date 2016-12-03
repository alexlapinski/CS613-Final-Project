

class DataSet(object):

    def __init__(self, dataframe, normal_label=1, anomaly_label=-1):
        self._anomalies = dataframe.loc[dataframe['label'] == anomaly_label]
        self._normal_data = dataframe.loc[dataframe['label'] == normal_label]


    @property
    def anomalous_data(self):
        return self._anomalies

    @property
    def normal_data(self):
        return self._normal_data