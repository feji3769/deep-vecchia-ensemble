"""variance_calibration.py
Variance calibration as described in [1].
[1] Scalable Bayesian Optimization Using Vecchia Approximations of Gaussian Processes
Author: Felix Jimenez
"""
import torch
from .vecchia_prediction import VecchiaPrediction
from torch.distributions import MultivariateNormal as MVN
from gpytorch.distributions import MultivariateNormal as gpytorchMVN
import numpy as np

class VarianceCalibration(VecchiaPrediction):
    def __init__(self, base_predictor, num_p = 5):
        """ variance calibration wrapper. 
        This Vecchia prediction wraps other Vecchia predictions to
        calibrate the base predition's variance estimates. 
        Args:
            base_predictor: base Vecchia prediciton object being wrapped. 
            num_p: number of points to use for internal validation set. 
        """
        self.num_p = num_p
        self.inflation_factor = torch.tensor([-1.0])
        self.base_predictor = base_predictor

    def __call__(self, test_x, model, update_calibration = False, **kwargs):
        if len(test_x.shape) < 3:
            x_p = test_x.unsqueeze(0)
        else:
            x_p = test_x
        posterior = self.base_predictor.posterior(x_p, model, **kwargs)
        if update_calibration or self.inflation_factor.item() < 0.0:
            self.update_calibration(model, **kwargs)
        posterior = self.calibrate_posterior(posterior, **kwargs)
        return posterior


    def calibrate_posterior(self, posterior, **kwargs):
        """Calibrate posterior variance.
        Add noise along diagonal based on current value of inflation factor.
        Args:
            posterior (MultivariateNormal) : uncalibrated posterior with covar shape (bs x n x n).
        Returns:
            posterior (MultivariateNormal) : calibrated posterior.
        """
        num_p = posterior.mean.shape[-1]
        posterior = torchify_distribution(posterior)
        Ib = torch.eye(num_p) * self.inflation_factor
        posterior = MVN(
            posterior.mean, 
            posterior.covariance_matrix + Ib
        )
        return posterior

    def update_calibration(self, model, nb = 1000, b_min = 0.0, b_max = 2.0, **kwargs):
        """Update variance calibration factor.
        Determine the variance calibration factor using either an 
        internal or external validation set. 
        Args:
            model (pyvecch.model.VecchiaGP): the trained model.
            nb (optional, int): number of values of b to sweep over. 
            b_min (optional, float): smallest b value to sweep over.   
            b_max (optional, float): largest b value to sweep over. 
            x_val (optional, torch.Tensor): if specified use this input as valiation inputs. 
            y_val (optional, torch.Tensor): if x_val specified use this response as valiation response. 
        """
        # get the posterior on some validation set. 
        if kwargs.get("x_val", None) is not None:
            x_val, y_val = self.use_external_val(model, **kwargs)
            is_self = False

        else:
            x_val, y_val = self.use_internal_val(model, **kwargs)
            is_self = True

        # form posterior over validation set. 
        with torch.no_grad():
            val_post = self.base_predictor.posterior(x_val, model, is_self = is_self, **kwargs)
            val_post = gpytorchMVN(val_post.mean, val_post.covariance_matrix)
            val_post = model.likelihood(val_post)

        val_post = torchify_distribution(val_post)
        n_val = x_val.shape[0]
        # additive variance.
        val_post = val_post.expand((nb,))
        b_vals = torch.linspace(b_min, b_max, nb).unsqueeze(-1).unsqueeze(-1)
        Ib = torch.eye(n_val) * b_vals
        val_post = MVN(
            val_post.mean, 
            val_post.covariance_matrix + Ib # eqn 6 in [1].
            )

        # score the posteriors.
        scores = val_post.log_prob(y_val)
        best_b = b_vals[scores.argmax()].squeeze()

        # use the one with maximum log prob.
        self.inflation_factor = best_b

    def use_internal_val(self, model, **kwargs):
        """Variance calibration on internal data. 
        Use some part of the training data to perform variance calibration.  
        Args:
            model (pyvecch.model.VecchiaGP): the trained model. 
        Returns:
            x_val (torch.Tensor): validation inputs. 
            y_val: validation outputs. 
        """
        ...
        if (self.num_p * 5) > model.n:
            raise Exception(f"Require n * 5 > num_p for variance calibration, n = {model.n}")
        if self.num_p < 2:
            raise Exception(f"Require num_p > 2, num_p = {self.num_p}")

        
        # center val. set around best x so far, assumes maximization for BO.
        x_ind = model.neighbor_oracle.y.argmax()
        x_q = model.neighbor_oracle.x[x_ind:(x_ind+1)]

        # get more than enough points for val. set and subsample.
        num_q = self.num_p * 5
        _, x_q_nbrs = model.query_neighbors(x_q, num_q, is_self = True)
        ind = np.arange(num_q)
        np.random.shuffle(ind)

        # form final validation set. 
        x_val = model.neighbor_oracle.x[ind[0:self.num_p]]; 
        y_val = model.neighbor_oracle.y[ind[0:self.num_p]]

        return x_val, y_val

    def use_external_val(self, model, **kwargs):
        """Variance calibration on external data. 
        Use an external validation set to perform the 
        variance calibration. 
        Args:
            model (pyvecch.model.VecchiaGP): the trained model. 
        Returns:
            x_val (torch.Tensor): validation inputs. 
            y_val: validation outputs. 
        """
        x_val = kwargs.get("x_val", None); 
        y_val = kwargs.get("y_val", None)
        return x_val, y_val

def torchify_distribution(dist):
    """turn to torch distribution if necessary. """
    if isinstance(dist, gpytorchMVN):
        dist = MVN(
            dist.mean, dist.covariance_matrix
        )
    return dist


