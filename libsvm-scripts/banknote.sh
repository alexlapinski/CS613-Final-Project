#!/usr/bin/env bash

TRAINING_DATA_PATH=`pwd`/../data/processed/libsvm/data_banknote_authentication_training
TEST_DATA_PATH=`pwd`/../data/processed/libsvm/data_banknote_authentication_test

SCALING_PARAMETERS="scale_parameters"
SCALED_TRAINING_DATA="training_data.scale"
SCALED_TEST_DATA="test_data.scale"

HYPERSPHERE_MODEL_PATH=`pwd`/../models/data_banknote_authentication/libsvm

source ./svdd.sh