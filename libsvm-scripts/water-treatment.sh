#!/bin/bash

LIBSVM_PATH=`pwd`/../../libsvm-svdd/
LIBSVM_TOOLS_PATH=`pwd`/../../libsvm-svdd/tools
START_PATH=`pwd`

TRAINING_DATA_PATH=`pwd`/../data/processed/libsvm/water-treatment_training
TEST_DATA_PATH=`pwd`/../data/processed/libsvm/water-treatment_test

SCALING_PARAMETERS="scale_parameters"
SCALED_TRAINING_DATA="training_data.scale"
SCALED_TEST_DATA="test_data.scale"

MODEL_FILE=`pwd`/../models/water-treatment/libsvm/rbf.model

RBF_NU_VALUE=0.0625
RBF_GAMMA_VALUE=0.00011094013184055919

# Change Directory to where libsvm tools exist
cd $LIBSVM_TOOLS_PATH

# Validate data
echo "Checking Training Data"
python checkdata.py $TRAINING_DATA_PATH
echo ""

echo "Checking Test Data"
python checkdata.py $TEST_DATA_PATH
echo ""

cd $LIBSVM_PATH
echo "Scaling Data"
./svm-scale -l 0 -u 1 -s $SCALING_PARAMETERS $TRAINING_DATA_PATH > $SCALED_TRAINING_DATA
./svm-scale -l 0 -u 1 -r $SCALING_PARAMETERS $TEST_DATA_PATH > $SCALED_TEST_DATA
rm -f $SCALING_PARAMETERS
echo ""

echo "Training One-Class SVM"
echo "Using RBF Kernel"
./svm-train -s 2 -t 2 -g $RBF_GAMMA_VALUE -n $RBF_NU_VALUE -v 5 $SCALED_TRAINING_DATA $MODEL_FILE

# Remove Scaled Data
#rm -f $SCALED_TRAINING_DATA
#rm -r $SCALED_TEST_DATA

# Change directory back
cd $START_PATH