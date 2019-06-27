# Author : Charles Laroche
# Last update : 03/04/2019


import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

plt.style.use('seaborn-darkgrid')


def accuracy(model, data, device='cpu'):
    """
    Compute the accuracy of a classification model, for instance it only works for binary classification

    """
    data_x = torch.stack([data[i][0] for i in range(len(data))]).float().to(device)
    data_y = torch.stack([data[i][1] for i in range(len(data))]).long().to(device)

    if len(np.unique(data_y)) == 2:
        output = (model(data_x).cpu().detach().numpy().flatten() > 0.5) * 1
        return accuracy_score(data_y.cpu().detach().numpy().flatten(), output)

    else:
        output = np.zeros(len(data_y.cpu().detach().numpy()))
        y_pred = model(data_x).cpu().detach().numpy()
        for i, elem in enumerate(y_pred):
            output[i] = np.argmax(elem)
            
        return accuracy_score(data_y.cpu().detach().numpy().flatten(), output)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
