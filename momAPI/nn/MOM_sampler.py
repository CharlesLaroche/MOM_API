# Author : Charles Laroche
# Last update : 03/04/2019

import torch
import itertools
import numpy as np
import numpy.random as alea
from torch.utils.data import Sampler, RandomSampler

random_seed = 7
alea.seed(random_seed)

class momsampler(Sampler):
    """
    The sampler generate the indexes of the median block with the following steps :

    First the sampler splits the database into K blocks then computes the loss for each block and selects the indexes of
    the block where the median of the losses is reached
    """
    def __init__(self, model, dataset, K,
                 loss=torch.nn.MSELoss(reduction='elementwise_mean'),
                 random_state=True , m = None, cuda = False):
        """
        To compute the median block we need a dataset, a model, K and a loss

        model : A torch network
        dataset : a Pytorch Dataset
        K : the number of blocks
        loss : torch loss
        random state : boolean True if we shuffle the indexes, else False
        """
        self.model = model
        self.dataset = dataset
        self.n_samples, self.input_shape = len(
            self.dataset), np.shape(self.dataset[0][0])
        self.K = K
        self.batch_size = self.n_samples // K
        self.loss = loss
        self.random_state = random_state
        self.cuda = cuda

    def __iter__(self):
        if self.K == 1:
            # This loop avoid the computation of the loss when K = 1
            return iter(range(self.n_samples))
        
        blocks = []
        means_blocks = []
        
        DataLoader = torch.utils.data.DataLoader(self.dataset, shuffle = self.random_state,
                                                 batch_size = self.batch_size, drop_last = True)
        for inputs, labels, indexes in DataLoader:
            with torch.no_grad():
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                blocks.append(indexes)
                mean = self.loss(self.model(inputs.float()) , labels).item()
                means_blocks.append(mean)
       
        indexes = np.argsort(means_blocks)[int(np.ceil(len(means_blocks) / 2))]
        return iter(blocks[indexes])

    def __len__(self): return 1
