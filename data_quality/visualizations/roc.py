import matplotlib.pyplot as plt
import os
from .. import filesystem as fs


def plot_combined(raw_data, level_index, title, dataset_name, filename):
    out_directory = os.path.join('reports', 'figures', dataset_name)
    plt.cla()

    fs.ensure_path_exists(out_directory)

    index_name = raw_data.index.names[level_index]
    index = raw_data.index.levels[level_index]

    for value in index:
        df = raw_data.xs(value, level=level_index)
        tpr = df['true_positive_rates']
        fpr = df['false_positive_rates']
        plt.plot(tpr, fpr, label='{0}={1}'.format(index_name,value))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', fontsize='xx-small')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.title(title)
    plt.savefig(os.path.join(out_directory, filename))

