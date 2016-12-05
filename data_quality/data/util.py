import os
from .. import filesystem as fs


def dataframe_to_libsvm(dataframe, name):
    """
    Take a labeled dataframe (with the last column as the labels) and write it out to a flat file
    in the libsvm format.
    Format (each row): <label> <index1>:<value1> <index2>:<value2> ...

    The index starts at '1' each row is terminated with '\n'
    The label must be an integer.
    :param dataframe: The processed dataframe
    :param name: The name of the dataset
    :return: path to written file
    """
    out_directory = os.path.join('data', 'processed', 'libsvm')
    fs.ensure_path_exists(out_directory)

    assert dataframe.columns[-1] == 'label', "The last column in the dataframe must be 'label'"

    out_filepath = os.path.join(out_directory, name)
    with open(out_filepath, 'w') as f:
        for index, row in dataframe.iterrows():
            label = row[row.index[-1]]
            values = row[row.index[1:-1]].values

            values_as_str = [str(i+1)+':'+str(values[i]) if str(values[i]) != 'nan' else ''
                             for i in xrange(len(values))]
            f.write('{label} {values}\n'.format(label=label, values=' '.join(values_as_str)))

    return out_filepath
