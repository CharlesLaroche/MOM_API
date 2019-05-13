# Author : Charles Laroche
# Last update : 03/04/2019

from torch.utils import data


class Dataset(data.Dataset):

    # X : np.array/list, the database
    # Y: np.array/list, the target
    # transform : torchvision transformer

    def __init__(self, X,  Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        if self.transform:
            X = self.transform(self.X[index])

        else:
            X = self.X[index]

        y = self.Y[index]
        return X, y, index
