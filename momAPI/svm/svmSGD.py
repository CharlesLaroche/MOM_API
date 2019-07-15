# Author : Tom Guedon
# Last modification = 15/07/2019

import numpy as np
from ..procedure.procedure_MOM import mom


def mom_svm_sgd(x, y, k, kernel, kerparam, max_iter, tol, stepsize, pen):

    loss = []
    mom_loss = []
    n_sample, p = x.shape

    # list of frequency of each observation in the mom block
    mom_freq = np.zeros(n_sample)
    mom_hist = []

    if not kernel:
        
        # first column of the features matrix is one for the intercept
        x_prime = np.ones((n_sample, p + 1))
        
        x_prime[:, 1:] = x
        
        # initialisation of the parameters to 0
        w = np.zeros(p + 1)
        n_iter = 0
        var_w = tol + 1
        
        # upload residuals according to the hinge loss
        r = -np.dot(x_prime, w)
        r = r * y + 1
        r[r < 0] = 0

        # start the SGD
        while n_iter < max_iter and var_w > tol:
            
            # select the mom blocks
            n_iter += 1
            
            indice = mom(r, k)[1]
            
            x_train = x_prime[indice, :]
            y_train = y[indice]
            
            # store the previous value
            w_j = w
            
            # iterations of the SGD
            for i in range(len(indice)):
                
                if (y_train[i]*np.dot(x_train[i], w)) < 1:
                    w = w + (0.1 / np.sqrt(n_iter)) * ((x_train[i] * y_train[i]) + (-2 * (1/n_iter) * w))
                else:
                    w = w + (stepsize / np.sqrt(n_iter)) * (-2 * (1 / n_iter) * w)
            var_w = np.sum(np.square(w - w_j))
            
            # upload residuals
            r = -np.dot(x_prime, w)
            r = r * y + 1
            r[r < 0] = 0
                    
            # update mom_freq mom_hist
            if n_iter > 20:
                for i in indice:
                    mom_freq[i] += 1
                    mom_hist.append(i)

            # update the loss list to plot
            loss.append(np.mean(r))
            mom_loss.append(np.mean(r[indice]))
            
        return {"parameters": (w[0], w[1:]), "kernel": None, "loss": loss, "mom_loss": mom_loss, "mom_freq": mom_freq,
                "mom_hist": mom_hist}

    else:
        # kernel must be added accorded to the stored ones or must be already created according to the
        # input and output that fit the structure of the code
        # kernel is a function from r^p*r^p->r

        n_sample = x.shape[0]
        p = x.shape[1]
        
        # first column of the features matrix is one for the intercept
        x_prime = np.ones((n_sample, p + 1))
        
        x_prime[:, 1:] = x
        
        # initialisation of the parameters to 1/n_sample
        
        alpha = np.zeros(n_sample)
        n_iter = 0
        var_w = tol+1
        
        # create the kernel Matrix
        ker = np.zeros((n_sample, n_sample))
        
        for i in range(n_sample):
            for j in range(i, n_sample):
                ker[i, j] = kernel(x_prime[i], x_prime[j], kerparam)
                ker[j, i] = ker[i, j]
                
        # upload residuals according to the hinge loss : (1-yi*h(xi))+
        r = np.dot(alpha, ker)
        r = r * y + 1
        r[r < 0] = 0

        loss.append(np.mean(r))
        # start the SGD
        
        while n_iter < max_iter and var_w > tol:
            n_iter += 1

            # select the mom blocks
            mom_loss_step, indice = mom(r, k)
            mom_loss.append(mom_loss_step)
            
            # restriction to the mom block
            y_train = y[indice]
            mom_ker = ker[indice][:, indice]
            mom_alpha = alpha[indice]
            
            # calculate the gradient
            grad = 2 * pen * np.dot(mom_ker, mom_alpha)
            for i in range(len(indice)):
                if -y_train[i] * np.dot(mom_alpha, mom_ker[:, i]) > -1:
                    grad += -y_train[i] * mom_ker[:, i]

            grad = np.float_(grad)
            grad *= 1/len(indice)

            # update the parameter
            alpha[indice] -= stepsize/np.sqrt(n_iter) * grad
            
            # update residuals
            r = np.dot(alpha, ker)
            r = r - y + 1
            r[r < 0] = 0
                
            # update the loss list to plot
            loss.append(np.mean(r))

            var_w = np.abs(np.mean(1 / (n_iter ** 0.3) * grad))

            # update mom_freq from 10th iteration
            if n_iter >= 10:
                for i in indice:
                    mom_freq[i] += 1
                    mom_hist.append(i)

        outliers = []
        for i in range(len(mom_freq)):
            if mom_freq[i] == 0:
                outliers.append(i)
        
        return {"parameters": (alpha[0], alpha[1:]), "kernel": ker, "loss": loss, "mom_loss": mom_loss,
                "mom_freq": mom_freq, "mom_hist": mom_hist}
