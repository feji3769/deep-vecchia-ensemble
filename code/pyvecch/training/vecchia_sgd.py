from dataclasses import dataclass
import torch
import numpy as np 
from .stopping import ExpMAStoppingCriterion
from gpytorch.mlls import ExactMarginalLogLikelihood
import random 


def fit_model(model, **kwargs):
    """
    INPUTS: 

    model : a model of type VecchiaModel.
    kwargs : {
        'lr' : learning rate for SGD,
        'batch_size' : batch size for vecchia training, 
        'optimizer' : the torch optimizer to use, defaults to Adam.
        'epochs' : number of epochs to train model for, 
        'tracking' : (boolean) log-likelihood should be tracked during training (evaluates full likelihood at each epoch), 
        'verbose' : (boolean) status should be printed during training, 
        'lengthscale_bound' : largest value lengthscale can take, defaults to 5. 
    }

    RETURNS:
    None unless tracking is true, then returns list containing log-likelihood values during training.
    """
    n = model.n
    batch_size = kwargs.get("train_batch_size", n)
    opt_func = kwargs.get("optimizer", torch.optim.Adam)
    stopping_options = {
        "maxiter": kwargs.get("max_iterations", 100), 
        "n_window":kwargs.get("n_window", 20), 
        "rel_tol":kwargs.get("rel_tol", 1e-5)}

    stop = False

    stopping_criterion = ExpMAStoppingCriterion(**stopping_options)
    optimizer = opt_func(model.parameters(), lr = kwargs.get("lr", .05))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    num_batches = n // batch_size
    ind = np.arange(n)
    np.random.shuffle(ind)
    if kwargs.get("tracking", False):
        lik = []
    e = 0
    verbose = kwargs.get("verbose", False)

    while not stop:
        if verbose and (e % 10) == 0:    
            print("epoch : {}".format(e))
            
        e += 1
        np.random.shuffle(ind)
        batches = np.array_split(ind, num_batches)
        for batch in batches:

            bs = batch.shape[0]
            optimizer.zero_grad()
            x_batch, y_batch = model.neighbor_oracle[batch]
            output = model(batch)
            loss = -1 * mll(output, y_batch) * n / bs
            loss.backward()
            optimizer.step()
            
            stop = stopping_criterion.evaluate(fvals=loss.detach())
            if stop: break
