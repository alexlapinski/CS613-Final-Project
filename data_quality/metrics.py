class Metrics(object):

    def __init__(self, num_true_positives, num_true_negatives, num_false_positives, num_false_negatives):
        self._num_true_positives = num_true_positives
        self._num_true_negatives = num_true_negatives
        self._num_false_positives = num_false_positives
        self._num_false_negatives = num_false_negatives

    def compute_precision(self):
        return self._num_true_positives / float(self._num_true_positives + self._num_false_positives)

    def compute_recall(self):
        return self._num_true_positives / float(self._num_true_positives + self._num_false_negatives)

    def compute_accuracy(self):
        numerator = (self._num_true_positives + self._num_true_negatives)
        denominator = (self._num_true_positives + self._num_true_negatives +
                       self._num_false_positives + self._num_false_negatives)

        return numerator / float(denominator)

    def __repr__(self):
        return "Metrics(precision={0}," \
               "        recall={1}," \
               "        accuracy={2})".format(self.compute_precision(),
                                              self.compute_recall(),
                                              self.compute_accuracy())