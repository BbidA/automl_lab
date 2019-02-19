import pickle
import numpy as np
import os.path

curr_path = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(curr_path, '../temp_dataset')


def dataset_reader(train_file):
    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    return train_features, train_labels


def all_data(include=None, exclude=None):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if 'train_data.pkl' in file:
                data_name = file[:file.find('_train_data.pkl')]
                if include is not None:
                    if data_name in include:
                        data.append(DataSet(data_name))
                elif exclude is not None:
                    if data_name not in exclude:
                        data.append(DataSet(data_name))
                else:
                    data.append(DataSet(data_name))

    return data


def load_data_sets(names):
    return [DataSet(name) for name in names]


def adult_dataset():
    x1, y1 = dataset_reader(os.path.join(curr_path, "../temp_dataset/adult/adult_train_data.pkl"))
    x2, y2 = dataset_reader(os.path.join(curr_path, "../temp_dataset/adult/adult_test_data.pkl"))
    return np.concatenate([x1, x2]), np.concatenate([y1, y2])


def car_dataset():
    x1, y1 = dataset_reader(os.path.join(curr_path, "../temp_dataset/car/car_train_data.pkl"))
    x2, y2 = dataset_reader(os.path.join(curr_path, "../temp_dataset/car/car_test_data.pkl"))
    return np.concatenate([x1, x2]), np.concatenate([y1, y2])


def cmc_dataset():
    x, y = dataset_reader(os.path.join(curr_path, '../temp_dataset/cmc/cmc_train_data.pkl'))
    x2, y2 = dataset_reader(os.path.join(curr_path, '../temp_dataset/cmc/cmc_test_data.pkl'))

    return np.concatenate([x, x2]), np.concatenate([y, y2])


def banknote_dataset():
    x, y = dataset_reader(os.path.join(curr_path, '../temp_dataset/banknote/banknote_train_data.pkl'))
    x2, y2 = dataset_reader(os.path.join(curr_path, '../temp_dataset/banknote/banknote_test_data.pkl'))

    return np.concatenate([x, x2]), np.concatenate([y, y2])


def _find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def _read_data(file_name):
    path = os.path.join(curr_path, '../temp_dataset')
    return dataset_reader(_find(file_name, path))


class DataSet:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'DataSet({})'.format(self.name)

    def train_data(self):
        file = '{}_train_data.pkl'.format(self.name)
        return _read_data(file)

    def test_data(self):
        file = '{}_test_data.pkl'.format(self.name)
        return _read_data(file)
