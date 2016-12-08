python -m data_quality.train
# Executing One-Class SVM for water-treatment
## Using 'polynomial' Kernel
Parameters = {u'nu': 0.7286055433745294, u'gamma': 7.62939453125e-06, u'degree': 2, u'coef': 0.2700149347230765}

Training using 341 samples
Testing using 171 normal samples
Testing using 11 anomalous samples

 * Accuracy: 0.648351648352
 * F1 Measure: 0.786666666667
 * Precision: 0.914728682171
 * Recall / TPR: 0.690058479532
 * FPR: 1.0


## Using 'rbf' Kernel
Parameters = {u'nu': 0.0625, u'gamma': 4.8443635619146695e-05}

Training using 341 samples
Testing using 171 normal samples
Testing using 11 anomalous samples

 * Accuracy: 0.945054945055
 * F1 Measure: 0.969879518072
 * Precision: 1.0
 * Recall / TPR: 0.941520467836
 * FPR: 0.0


## Using 'linear' Kernel
Parameters = {u'nu': 0.3401975000435943}

Training using 341 samples
Testing using 171 normal samples
Testing using 11 anomalous samples

 * Accuracy: 0.593406593407
 * F1 Measure: 0.73381294964
 * Precision: 0.953271028037
 * Recall / TPR: 0.59649122807
 * FPR: 0.454545454545


## Using 'sigmoid' Kernel
Parameters = {u'nu': 0.08504937501089857, u'gamma': 3.1748021039363987, u'coef': 7.62939453125e-06}

Training using 341 samples
Testing using 171 normal samples
Testing using 11 anomalous samples

 * Accuracy: 0.923076923077
 * F1 Measure: 0.958823529412
 * Precision: 0.96449704142
 * Recall / TPR: 0.953216374269
 * FPR: 0.545454545455


Wrote Combined metrics for water-treatment to 'models/results/water-treatment/one_class_svm.csv'


# Executing One-Class SVM for banknote
## Using 'polynomial' Kernel
Parameters = {u'nu': 0.5899575796083221, u'gamma': 4.152801514204027e-05, u'degree': 1, u'coef': 0.2700149347230765}

Training using 406 samples
Testing using 204 normal samples
Testing using 762 anomalous samples

 * Accuracy: 0.0
 * F1 Measure: inf
 * Precision: 0.0
 * Recall / TPR: 0.0
 * FPR: 1.0


## Using 'rbf' Kernel
Parameters = {u'nu': 0.0625, u'gamma': 0.05786716951795567}

Training using 406 samples
Testing using 204 normal samples
Testing using 762 anomalous samples

 * Accuracy: 0.76397515528
 * F1 Measure: 0.296296296296
 * Precision: 0.4
 * Recall / TPR: 0.235294117647
 * FPR: 0.0944881889764


## Using 'linear' Kernel
Parameters = {u'nu': 0.15749013123685915}

Training using 406 samples
Testing using 204 normal samples
Testing using 762 anomalous samples

 * Accuracy: 1.0
 * F1 Measure: 1.0
 * Precision: 1.0
 * Recall / TPR: 1.0
 * FPR: 0.0


## Using 'sigmoid' Kernel
Parameters = {u'nu': 0.18371681153444983, u'gamma': 0.01687593342019229, u'coef': 7.62939453125e-06}

Training using 406 samples
Testing using 204 normal samples
Testing using 762 anomalous samples

 * Accuracy: 0.17701863354
 * F1 Measure: 0.300791556728
 * Precision: 0.183279742765
 * Recall / TPR: 0.838235294118
 * FPR: 1.0


Wrote Combined metrics for banknote to 'models/results/banknote/one_class_svm.csv'


# Executing One-Class SVM for hdd
## Using 'polynomial' Kernel
Parameters = {u'nu': 0.0625, u'gamma': 0.0036166980948722292, u'degree': 3, u'coef': 9.332232316608934}

Training using 10000 samples
Testing using 10000 normal samples
Testing using 25 anomalous samples

 * Accuracy: 0.936259351621
 * F1 Measure: 0.967080521354
 * Precision: 0.997343534162
 * Recall / TPR: 0.9386
 * FPR: 1.0


## Using 'rbf' Kernel
Parameters = {u'nu': 0.0625, u'gamma': 3.559964110034533e-05}

Training using 10000 samples
Testing using 10000 normal samples
Testing using 25 anomalous samples

 * Accuracy: 0.93885286783
 * F1 Measure: 0.968380873781
 * Precision: 1.0
 * Recall / TPR: 0.9387
 * FPR: 0.0


## Using 'linear' Kernel
Parameters = {u'nu': 0.0625}

Training using 10000 samples
Testing using 10000 normal samples
Testing using 25 anomalous samples

 * Accuracy: 0.939351620948
 * F1 Measure: 0.968646864686
 * Precision: 1.0
 * Recall / TPR: 0.9392
 * FPR: 0.0


## Using 'sigmoid' Kernel
Parameters = {u'nu': 0.0625, u'gamma': 3.7034988491491614, u'coef': 7.62939453125e-06}

Training using 10000 samples
Testing using 10000 normal samples
Testing using 25 anomalous samples

 * Accuracy: 0.938653366584
 * F1 Measure: 0.968274438999
 * Precision: 1.0
 * Recall / TPR: 0.9385
 * FPR: 0.0


Wrote Combined metrics for hdd to 'models/results/hdd/one_class_svm.csv'


