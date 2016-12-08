CS613 - Final Project - Anomaly Detection for Data Quality
===========================================================

How to Run
----------

One Class SVM (All Python)
* Install Dependencies
** make requirements
* Prepare Data
** make data
* Find best parameters
** make params
* Test / Train SVM
** make train

SVDD (patched libsvm)
* Compile libsvm
** cd libsvm
** make
* Prepare Data
** cd ../data
* Run Bash Scripts to test
** cd ../libsvm-scripts
** sh banknote.sh
** sh water-treatment.sh
** sh hdd.sh
