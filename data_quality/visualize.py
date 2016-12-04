import visualizations.roc as visroc
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_plot_data(directory='roc'):
    folder_path = os.path.join('data', 'visualizations', directory)
    plot_data = {}
    for folder_name in os.listdir(folder_path):
        plot_data[folder_name] = {}
        for file_name in os.listdir(os.path.join(folder_path, folder_name)):
            filepath = os.path.join(folder_path, folder_name, file_name)
            file_data = pd.read_csv(filepath, index_col=0, dtype=[list, list])
            plot_name = os.path.splitext(file_name)[0]
            plot_data[folder_name][plot_name] = file_data

    return plot_data


if __name__ == "__main__":
    plt.style.use('ggplot')

    datasets = get_plot_data('roc')

    for dataset_name, dataset_values in datasets.items():
        for plot_name, plot_data in dataset_values.items():
            filename = '{0}.png'.format(plot_name)
            visroc.plot_combined(plot_data,
                                 title='One Class SVM; vary gamma; rbf kernel',
                                 dataset_name=dataset_name,
                                 filename=filename)
