from ..data import reader as reader
from ..features import dataset as dataset
from ..features import util as util
from ..features import preparation as prep

import os


def cross_validate_one_class_svm(dataset):
    # TODO: Do Cross Validation
    datasets = prep.create_cross_validation_data(dataset)


if __name__ == "__main__":
    water_treatment_filepath = os.path.join('data', 'processed', 'water-treatment.csv')
    water_treatment_data = dataset.OneClassDataSet(reader.read_water_treatment_data(water_treatment_filepath))
    cross_validate_one_class_svm(water_treatment_data)
