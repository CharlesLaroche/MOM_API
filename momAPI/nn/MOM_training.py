# Author : Charles Laroche
# Last update : 27/06/2019

import torch
from torch.autograd import Variable
from .MOM_sampler import MomSampler
import time
from .utilities import *
import torch.utils.data as data

random_seed = 7
np.random.seed(random_seed)


class MomTraining:
    """
    This class compute the MOM training of a torch neural network.
    In fact, at each step, we select the median block and train the network on this block. If the block is too large,
    we split it into batches and trained it on every batches during the epoch.

    """
    def __init__(self, model, optimizer, loss, n_epochs=10, batch_size=32, device='cpu', n_hist=30):
        """
        model : a torch model, this is the network we want to fit

        optimizer : torch.nn optimizer, the MOM training works for all the classical optimizer (SGd, Adagrad , Adam,
         RMSprop...)

        loss : torch.nn loss , the loss to optimize during the training (Maybe I have to ad some modification to make
         it work           with every losses)

        K : the number of blocks, the number of blocks our database will be splited in

        n_epochs : number of epochs, number of gradient descent on the median block

        batch_size : size of the batch, if the median block is too large, we split it into batches of batch_size and
         trained our
        net on each batch

        history : loss (resp val_loss) contains the loss on the training set (resp validation set) at each step of the
        training,
        acc (resp val_acc) contains the accuracy on the training set (resp validation set) at each step of the training,
        hist contains the frequency of occurence in the median block for the n_hist last step of the training

        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hist = n_hist 
        self.history = {'loss': [], 'val_loss': [],
                        'acc': [], 'val_acc': [], 'hist': []}
        self.sampler = None
        self.K = None
        self.device = device

    def fit(self, data_train, acc=False, data_val=None):

        self.K = len(data_train) // self.batch_size
        self.sampler = MomSampler(self.model, data_train, self.K, loss=self.loss,
                                  random_state=True, device=self.device)
        train_loader = data.DataLoader(data_train, num_workers=1,
                                       sampler=self.sampler, batch_size=self.batch_size)

        if data_val:
            val_loader = data.DataLoader(data_val, batch_size=len(data_val))
            val_inputs, val_labels, _ = val_loader.__iter__().next()
            val_inputs, val_labels = val_inputs.float().to(self.device), val_labels.float().to(self.device)

        for epoch in range(self.n_epochs):
            t1 = time.time()

            # get the input
            inputs, labels, indexes = train_loader.__iter__().next()

            running_loss = 0.0
            running_val_loss = 0.0

            if epoch >= self.n_epochs - self.n_hist:
                self.history['hist'] += list(indexes)

            inputs = Variable(inputs.float().to(self.device), requires_grad=True)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()  # zero the parameter gradients
            loss_ = self.loss(self.model(inputs), labels)
            loss_.backward()  # compute the gradient of the loss
            self.optimizer.step()

            running_loss += loss_.item()
                
            if data_val:
                with torch.no_grad():
                    val_loss_ = self.loss(self.model(val_inputs), val_labels)
                    running_val_loss += val_loss_.item()
                    self.history['val_loss'].append(running_val_loss)
            
            self.history['loss'].append(running_loss)
            
            accu = None
            if acc:            
                accu = accuracy(self.model, data_train, device=self.device)
                self.history['acc'].append(accu)
            
            if data_val:
                with torch.no_grad():
                    val_accu = None
                    if acc:
                        val_accu = accuracy(self.model, data_val, device=self.device)
                    self.history['val_acc'].append(val_accu)
                    t2 = time.time()
                    print("Epoch n°{} ({}sec) : loss = {}, validation loss = {}, accuracy = {},"
                          " validation accuracy = {}".format(str(epoch), str((t2 - t1) // 1), str(running_loss),
                                                             str(running_val_loss), str(accu), str(val_accu)))

            else:
                t2 = time.time()
                print("Epoch n°{} ({}sec) : loss ={}, Accuracy ={}".format(str(epoch), str((t2 - t1) // 1),
                                                                           str(running_loss), str(accu)))

        print("Training finished")

    def score(self, data_):
        x = torch.stack([data_[i][0] for i in range(len(data_))]).float()
        y = torch.stack([data_[i][1]
                         for i in range(len(data_))]).float()

        output = self.model(x)

        return self.loss(output, y).item()
