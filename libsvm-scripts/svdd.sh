#!/usr/bin/env bash

LIBSVM_PATH=`pwd`/../libsvm/
LIBSVM_TOOLS_PATH=`pwd`/../libsvm/tools
START_PATH=`pwd`

# Change Directory to where libsvm tools exist
cd ${LIBSVM_TOOLS_PATH}

# Validate data
echo "Checking Training Data"
python checkdata.py ${TRAINING_DATA_PATH}
echo ""

echo "Checking Test Data"
python checkdata.py ${TEST_DATA_PATH}
echo ""

cd ${LIBSVM_PATH}
echo "Scaling Data"
./svm-scale -l 0 -u 1 -s ${SCALING_PARAMETERS} ${TRAINING_DATA_PATH} > ${SCALED_TRAINING_DATA}
./svm-scale -l 0 -u 1 -r ${SCALING_PARAMETERS} ${TEST_DATA_PATH} > ${SCALED_TEST_DATA}
rm -f ${SCALING_PARAMETERS}
echo ""

if [ ! -d ${HYPERSPHERE_MODEL_PATH} ]; then
  mkdir -p ${HYPERSPHERE_MODEL_PATH};
  echo "Created ${HYPERSPHERE_MODEL_PATH}"
fi

echo "Training SVDD"
./svm-train ${SCALED_TEST_DATA} ${HYPERSPHERE_MODEL_PATH}/hypersphere.model out
echo ""

echo "Testing SVDD"
./svm-predict ${SCALED_TEST_DATA} ${HYPERSPHERE_MODEL_PATH}/hypersphere.model out
echo ""

# Remove Scaled Data
rm -f ${SCALED_TRAINING_DATA}
rm -r ${SCALED_TEST_DATA}
rm -f out

# Change directory back
cd ${START_PATH}