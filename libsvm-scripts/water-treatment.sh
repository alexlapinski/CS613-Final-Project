#!/usr/bin/env bash

LIBSVM_PATH=`pwd`/../../libsvm-svdd/
LIBSVM_TOOLS_PATH=`pwd`/../../libsvm-svdd/tools
START_PATH=`pwd`

TRAINING_DATA_PATH=`pwd`/../data/processed/libsvm/water-treatment_training
TEST_DATA_PATH=`pwd`/../data/processed/libsvm/water-treatment_test

SCALING_PARAMETERS="scale_parameters"
SCALED_TRAINING_DATA="training_data.scale"
SCALED_TEST_DATA="test_data.scale"

HYPERSPHERE_MODEL_PATH=`pwd`/../models/water-treatment/libsvm

source ./svdd.sh