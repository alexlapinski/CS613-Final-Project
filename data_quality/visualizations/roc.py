import matplotlib.pyplot as plt
import os


def plot_combined(plot_data, title, dataset_name, filename):
    out_directory = os.path.join('reports', 'figures', dataset_name)
    for index, series in plot_data.iterrows():
        tpr = series['true_positive_rates']
        fpr = series['false_positive_rates']
        print type(tpr), type(fpr)
        print len(tpr), len(fpr)
        print type(tpr[0]), type(fpr[0])
        plt.plot(tpr, fpr, label=index)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper-right', fontsize='xx-small')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.title(title)
    plt.savefig(os.path.join(out_directory, filename))

