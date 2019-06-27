# Author : Charles Laroche
# Last update : 03/04/2019

from torch.utils import data


class Dataset(data.Dataset):
    """
    X : np.array/list, the database
    Y: np.array/list, the target
    transform : torchvision transformer
    """

    def __init__(self, x,  y, transform=None):
        super(Dataset, self).__init__()
        self.X = x
        self.Y = y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.X[index])

        else:
            x = self.X[index]

        y = self.Y[index]
        return x, y, index
