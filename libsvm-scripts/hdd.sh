#!/usr/bin/env bash

TRAINING_DATA_PATH=`pwd`/../data/processed/libsvm/hdd_training
TEST_DATA_PATH=`pwd`/../data/processed/libsvm/hdd_test

SCALING_PARAMETERS="scale_parameters"
SCALED_TRAINING_DATA="training_data.scale"
SCALED_TEST_DATA="test_data.scale"

HYPERSPHERE_MODEL_PATH=`pwd`/../models/hdd/libsvm

source ./svdd.sh