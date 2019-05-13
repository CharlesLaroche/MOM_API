# Author : Charles Laroche
# Last update : 03/04/2019

import torch
from torch.autograd import Variable
import numpy as np
from .MOM_sampler import momsampler
import time
from .utilities import *

random_seed = 7
np.random.seed(random_seed)


class mom_net():
    """
    This class compute the MOM training of a torch neural network.
    In fact, at each step, we select the median block and train the network on this block. If the block is too large,
    we split it into batches and trained it on every batches during the epoch.

    """
    def __init__(self, model, optimizer, loss,
                 random_seed=False, n_epochs=10,
                 batch_size=32, n_hist=30):
        """
        model : a torch model, this is the network we want to fit

        optimizer : torch.nn optimizer, the MOM training works for all the classical optimizer (SGD, Adagrad , Adam, RMSprop...)

        loss : torch.nn loss , the loss to optimize during the training (Maybe I have to add some modification to make it work           with every losses)

        K : the number of blocks, the number of blocks our database will be splited in

        n_epochs : number of epochs, number of gradient descent on the median block

        batch_size : size of the batch, if the median block is too large, we split it into batches of batch_size and trained our 
        net on each batch

        history : loss (resp val_loss) contains the loss on the training set (resp validation set) at each step of the training, 
        acc (resp val_acc) contains the accuracy on the training set (resp validation set) at each step of the training,
        hist contains the frequency of occurence in the median block for the n_hist last step of the training

        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.history = {'loss': [], 'val_loss': [],
                        'acc': [], 'val_acc': []}

    def fit(self, data_train, acc = False, data_val=[], cuda = False):
        
        val = (len(data_val) != 0)
        if val:
            val_loader = torch.utils.data.DataLoader(data_val, batch_size = len(data_val))
            for val_in,val_lab,_ in val_loader:
                val_inputs, val_labels = val_in.float(), val_lab.float()
                #if torch.cuda.is_available():
                    #val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                    
        DataLoader = torch.utils.data.DataLoader(data_train, num_workers=1,
                                                     shuffle = True,
                                                     batch_size = self.batch_size)
        n_batch = len(data_train) // self.batch_size
        for epoch in range(self.n_epochs):
            t1 = time.time()
            running_loss = 0.0
            running_val_loss = 0.0

            for i, data in enumerate(DataLoader, 0):
                # get the input
                inputs, labels, indexes = data

                inputs = Variable(inputs.float() , requires_grad = True)
                labels = labels.float()
                
                #if torch.cuda.is_available():
                    #inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                
                loss_ = self.loss(outputs , labels)

                loss_.backward()  # compute the gradient of the loss
                self.optimizer.step()

                running_loss += loss_.item()/n_batch
                
                
            if val:
                with torch.no_grad():
                    val_loss_ = self.loss(self.model(val_inputs) , val_labels)
                    running_val_loss += val_loss_.item()
                    self.history['val_loss'].append(running_val_loss)
            
            self.history['loss'].append(running_loss)
            
            accu = None
            if acc:
                accu = accuracy(self.model, data_train)
                self.history['acc'].append(accu)
            
            if val:
                with torch.no_grad():
                    val_accu = None
                    if acc:
                        val_accu = accuracy(self.model, data_val)
                    self.history['val_acc'].append(val_accu)
                    t2 = time.time()
                    print("Epoch n°"+str(epoch)+" (", (t2-t1) // 1, "sec) : loss =", running_loss,
                      ",validation loss = ", running_val_loss, ", accuracy :", accu, ", validation accuracy :", val_accu)
            else:
                t2 = time.time()
                print("Epoch n°"+str(epoch)+" (", (t2-t1) // 1, "sec) : loss =",
                      running_loss, "Accuracy =", accu)

        print("Training finished")

    def score(self, data):
        X = torch.stack([data[i][0] for i in range(len(data))]).float()
        y = torch.stack([data[i][1]
                         for i in range(len(data))]).float()

        output = self.model(X)

        return self.loss(output, y).item()
