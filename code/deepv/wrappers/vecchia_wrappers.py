"""vecchia_wrappers.py
Code wrapping pyvecch models and training loops. 
Author: Felix Jimenez
Date: 02/17/2023
"""
import logging
logger = logging.getLogger(__name__)
# gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

# pyvecch
from pyvecch.nbrs import ExactOracle, ApproximateOracle
from pyvecch.models import RFVecchia
from pyvecch.prediction import IndependentRF, VarianceCalibration
from pyvecch.training import fit_model
from pyvecch.input_transforms import Scaling, Identity
from pyvecch.training.stopping import ExpMAStoppingCriterion

#standard loads
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate(model, likelihood, x, y):
    """evaluate vecchia model on x,y pairs.
    Predict at locations x, and return nll of data y under 
    this prediction. 
    Args:
        model (vecchia GP): trained Vecchia GP to predict with. 
        likelihood (gpytorch likelihood): data likelihood. 
        x (torch.Tensor): Nxd inputs. 
        y (torch.Tensor): N targets
    Returns:
        model_nll (float): NLL of data under model predictions. 
    """
    nll_f = torch.nn.GaussianNLLLoss()

    try:
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred = likelihood(model(x))
            mu = pred.mean.squeeze()
            sig2 = pred.variance.squeeze()
            model_nll = nll_f(mu, y.squeeze(), sig2)
    except:
        n = x.shape[0]
        ind = np.arange(0,n)
        np.random.shuffle(ind)
        ind = ind[0:n//2]
        model_nll = evaluate(model, likelihood, x[ind], y[ind])



    model.train()
    likelihood.train()
    return model_nll


def get_scaled_vecchia_model(x,y, x_val = None, y_val = None,train = True,**kwargs):
    """get scaled vecchia GP. 
    Build and then train a scaled vecchia GP. 
    Args:
        x (torch.Tensor): Nxd inputs. 
        y (torch.Tensor): N targets. 
        x_val (torch.Tensor): N_val x d validation inputs. 
        y_val (torch.Tensor): N_val validation targets.
    Returns:
        model (pyvecch.models.VecchiaGP): vecchia model.
    """
    model, likelihood = build_scaled_vecchia(x,y,**kwargs)

    if x_val is not None:
        return fit_scaled_vecchia_validation(model, likelihood, x_val, y_val, **kwargs)
    else:
        return fit_model(model, **kwargs)

def build_scaled_vecchia(x,y,**kwargs):
    """build a scaled Vecchia GP. 
    Build but do not train a scaled Vecchia model. 
    Args:
        x (torch.Tensor): Nxd inputs. 
        y (torch.Tensor): N targets
    Returns:
        model (pyvecch.models.VecchiaGP): vecchia model.
        likelihood (gpytorch.likelihood): data model. 
    """
    m = kwargs.get("m", 3)
    covar_module = ScaleKernel(MaternKernel(
        ard_num_dims = x.shape[-1]
    ))
    covar_module.outputscale = 1.0

    covar_module.base_kernel.lengthscale = kwargs.get("init_len_scale", .1)
    mean_module = ZeroMean()
    likelihood = GaussianLikelihood(
        noise_constraint = Interval(
            kwargs.get("noise_lb",1e-12), 
            kwargs.get("noise_ub",5e-1)
            ), 
        noise_prior = kwargs.get("noise_prior", None)
    )
    if kwargs.get("approx_nn", False):
        neighbor_oracle = ApproximateOracle(
            x,y,m, 
            n_list=kwargs.get("n_list", 100), 
            n_probe = kwargs.get("n_probe",50)
        )
    else:
        neighbor_oracle = ExactOracle(x,y,m)
    if kwargs.get("calibrate_var", False):
        prediction_stategy = VarianceCalibration(IndependentRF())
    else:
        prediction_stategy = IndependentRF()
    if kwargs.get("scaled_nn", False):
        input_transform = Scaling(d = x.shape[-1])
    else:
        input_transform = Identity(d = x.shape[-1])

    model = RFVecchia(covar_module, mean_module, likelihood, 
        neighbor_oracle, prediction_stategy, input_transform)
    
    return model, likelihood


def fit_scaled_vecchia_validation(model, likelihood, x_val, y_val, **kwargs):
    """fit scaled Vecchia GP using validation set. 
    Use an exponential moving average with a validation 
    dataset to decide when to stop training a Vecchia GP model. 
    Args:
        model (pyvecch.models.VecchiaGP): vecchia model.
        likelihood (gpytorch.likelihood): data model. 
                x_val (torch.Tensor): N_val x d validation inputs. 
        y_val (torch.Tensor): N_val validation targets. 
    Returns:
        model (pyvecch.models.VecchiaGP): vecchia model.
        likelihood (gpytorch.likelihood): data model. 
    """
    model.train()
    likelihood.train()
    # get stopping options.
    stopping_options = {
        "maxiter": kwargs.get('maxiter', 50), 
        "n_window": kwargs.get('n_window', 3),
        "rel_tol":kwargs.get('rel_tol', 5e-3)
    }
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(**stopping_options)
    # get optimizer.
    opt = torch.optim.Adam(
        model.parameters(),
        lr = kwargs.get('lr', .1)
    )

    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[kwargs.get('maxiter', 50) * .5, kwargs.get('maxiter', 50) * .75], gamma=0.1
    )
    # marginal likelihood.
    mll = ExactMarginalLogLikelihood(likelihood, model)
    # indices for batching
    ind = np.arange(model.neighbor_oracle.n)
    num_batches = model.neighbor_oracle.n // kwargs.get("batch_size", 256)

    if kwargs.get("tracking", False):
        # should we track?
        tracked_loss = []; eval_loss = []

    # loop until convergence criterion satisfied.
    logger.info("starting training ...")
    logger.info(f"window {kwargs.get('n_window', 3)}")
    while not stop:     
        np.random.shuffle(ind)
        batches = np.array_split(ind, num_batches)
        for batch in batches[0:-1]:
            _, y_batch = model.neighbor_oracle[batch]

            # training step
            opt.zero_grad()

            output = likelihood(model(batch))

            loss = -1.0 * output.log_prob(y_batch.squeeze())/y_batch.shape[0]


            loss.backward()
            opt.step()
        
        sched.step()
        # eval step
        eval_loss_ = evaluate(model, likelihood, x_val, y_val)
        stop = stopping_criterion.evaluate(fvals=eval_loss_.detach())
        logger.info(f'eval_loss {eval_loss_.item():.5f}')
        if stop:
            break
        if kwargs.get("tracking", False):
            # store results if tracking.
            eval_loss.append(eval_loss_.item())
            tracked_loss.append(loss.item())
                
    if kwargs.get("tracking"):
        return model, likelihood, tracked_loss, eval_loss
    else:
        return model
