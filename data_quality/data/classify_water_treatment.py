import water_treatment_labels as labels
from daterange import DateRange
import pandas as pd


def __assign_label(dataframe, label_indices, label):
    for index in label_indices:
        if type(index) is DateRange:
            dataframe.loc[index.from_date:index.to_date, 'label'] = label
        else:
            parsed_index = pd.to_datetime(index, format='D-%d/%m/%y')
            dataframe.loc[parsed_index, 'label'] = label

    return dataframe


def __classify_normal(dataframe, label=1):
    """
    Classify the normal data (classes 1, 5, 8, 9, 10, 11, 12) with label '1'
    :param dataframe: input dataframe
    :return:
    """

    classified_dataframe = dataframe

    normal_data_classes = [labels.class_1, labels.class_5, labels.class_8, labels.class_9,
                           labels.class_10, labels.class_11, labels.class_12]

    for data_class in normal_data_classes:
        classified_dataframe = __assign_label(classified_dataframe, data_class, label)

    return classified_dataframe


def __classify_anomalies(dataframe, label=-1):
    """
    Classify the anomalous data (classes 2, 3, 4, 6, 7, 13) with label '-1'
    :param dataframe: input dataframe
    :return:
    """
    classified_dataframe = dataframe

    normal_data_classes = [labels.class_2, labels.class_3, labels.class_4, labels.class_6,
                           labels.class_7, labels.class_13]

    for data_class in normal_data_classes:
        classified_dataframe = __assign_label(classified_dataframe, data_class, label)

    return classified_dataframe


def classify(dataframe):
    """
    Classify the water treatment data
    :param dataframe: unclassified data
    :return: classified dataframe
    """

    # TODO: Read a 'classifier file'
    classified_dataframe = dataframe
    classified_dataframe['label'] = 0

    classified_dataframe = __classify_normal(classified_dataframe)
    classified_dataframe = __classify_anomalies(classified_dataframe)

    return classified_dataframe
