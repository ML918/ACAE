from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SegDataLoader(Dataset):
    def __init__(self, data, win_size, step, label=None):
        self.data = data
        self.win_size = win_size
        self.step = step
        self.label = label

    def __len__(self):
        return max((self.data.shape[0] - self.win_size) // self.step + 1, 0)

    def __getitem__(self, index):
        index *= self.step
        if self.label is None:
            return np.float32(self.data[index:index + self.win_size])
        else:
            return np.float32(self.data[index:index + self.win_size]), \
                   np.float32(self.label[index:index + self.win_size])



def get_loader_dataset(data_name, data_path, batch_size, win_size, step):
    normal = StandardScaler()
    allow_pickle = False
    if data_name == "SWAT":
        allow_pickle = True

    train = np.load(data_path + data_name + "/" + data_name + "_train.npy", allow_pickle=allow_pickle)
    test = np.load(data_path + data_name + "/" + data_name + "_test.npy", allow_pickle=allow_pickle)
    test_label = np.load(data_path + data_name + "/" + data_name + "_test_label.npy", allow_pickle=allow_pickle)
    normal.fit(train)
    train = normal.transform(train)
    test = normal.transform(test)
    n_features = train.shape[1]

    train_set = SegDataLoader(train, win_size, step)
    test_data = SegDataLoader(test, win_size, win_size, label=test_label)

    train_size = int(len(train_set) * 0.8)
    valid_size = len(train_set) - train_size
    train_data, valid_data = random_split(train_set, [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0, drop_last=False)

    return train_loader, valid_loader, test_loader, n_features