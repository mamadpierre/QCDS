import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn import datasets as sklearn_datasets
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.FloatTensor(data)
        self.target = torch.LongTensor(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_val = self.data[index]
        target = self.target[index]
        return data_val, target


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



def IRISDataLoaders(args):
    X_train, X_val, X_test, y_train, y_val, y_test = splittingIRIS()
    train = CustomDataset(X_train, y_train)
    val = CustomDataset(X_val, y_val)
    test = CustomDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader


def GlassDataLoaders(args):
    data = []
    target = []
    with open('data/glass', 'r') as f:
        for line in f:
            features = []
            separated = line.split()
            if int(separated[0]) == 1:
                target.append(0)
            elif int(separated[0]) == 2:
                target.append(1)
            elif int(separated[0]) == 3:
                target.append(2)
            elif int(separated[0]) == 5:
                target.append(3)
            elif int(separated[0]) == 6:
                target.append(4)
            elif int(separated[0]) == 7:
                target.append(5)

            for i in range(9):
                features.append(float(separated[i + 1].split(":")[1]))
            data.append(features)

        features_train,features_test, labels_train, labels_test = train_test_split(data, target, random_state=1, shuffle=True)

        train_dataset = CustomDataset(features_train, labels_train)
        test_dataset = CustomDataset(features_test, labels_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader


