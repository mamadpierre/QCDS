import torch
from torch.utils.data import Dataset
from sklearn import datasets as sklearn_datasets
import numpy as np


def splittingIRIS():
    iris = sklearn_datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train = np.concatenate((X[:20], X[50:70], X[100:120]), axis=0)
    X_val = np.concatenate((X[20:35], X[70:85], X[120:135]), axis=0)
    X_test = np.concatenate((X[35:50], X[85:100], X[135:150]), axis=0)

    y_train = np.concatenate((y[:20], y[50:70], y[100:120]), axis=0)
    y_val = np.concatenate((y[20:35], y[70:85], y[120:135]), axis=0)
    y_test = np.concatenate((y[35:50], y[85:100], y[135:150]), axis=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


class IRISDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.FloatTensor(data)
        self.target = torch.LongTensor(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_val = self.data[index]
        target = self.target[index]
        return data_val, target

def IRISDataLoaders(args):
    X_train, X_val, X_test, y_train, y_val, y_test = splittingIRIS()
    train = IRISDataset(X_train, y_train)
    val = IRISDataset(X_val, y_val)
    test = IRISDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader
