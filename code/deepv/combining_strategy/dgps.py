"""Adaptation of code from Healing-POEs-ICML Github repo. 

This file contains the implementation of DpPs combining methods. 

Author : Felix Jimenez
Date : 12/09/2022
"""

import torch
from .prediction_combiner import PredictionCombiner


class DgPs(PredictionCombiner):
    """Base class for DgPs combining methods.
    Args:
        weight_strategy (func) : a function that computes unnormalized weights 
                                 (output shape is N_models x N_test). 
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)

    def compute_precision(self, var_s, weights, prior_var):
        """Compute the precision values for the DgPs combining strategy.
        Method that each subclass needs to implement. 
        Args:
            var_s (torch.Tensor): N_models x N_test matrix of predicted means. 
            weights (torch.Tensor): N_models x N_test matrix of weights. 
            prior_var (torch.Tensor, optional) : N_models matrix of prior variances. Defaults to None.
        Returns:
            prec_s (torch.Tensor): N_test matrix of predicted precisions.
        """
        raise NotImplementedError

    def _combine(self, mu_s, var_s, noise_s, power = 1, prior_var=None, softmax=False, latent_posterior = True):
        """combine predictions from GPs.
        The procedure for combining predictions which each sublass shoudl implement.
        Args:
            mu_s (torch.Tensor): N_models x N_test matrix of predicted means. 
            var_s (torch.Tensor) : N_models x N_test matrix of predicted variances. 
            noise_s (torch.Tensor): N_models x N_test matrix of nuggets, :math:`\\tau^2`. 
            power (float, optional) : softmax weight. Defaults to 1.
            prior_var (torch.Tensor, optional) : N_models matrix of prior variances. Defaults to None.
            softmax (bool, optional) : Bool indicating softmax should be used. Defaults to False.
            latent_posterior (bool, optional): Bool indicating the posterior is for latent function, 
                                                i.e. should we add the noise. Defaults to True.
        Returns:
            mu (torch.Tensor): N_test matrix of combined predicted means. 
            var (torch.Tensor): N_test matrix of combined predicted variances
        """
        weights = self.compute_weights(mu_s, var_s, power, prior_var, softmax)
        prec = self.compute_precision(var_s, weights, prior_var)
        var = 1 / prec
        mu = var * (weights / var_s * mu_s).sum(0)
        if not latent_posterior:
            var = self.add_noise(var, noise_s)
        return mu, var

### Implemenetations of different methods. 

class PoE(DgPs):
    """Product of Experts combining strategy.
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)
    def compute_precision(self, var_s, weights, prior_var):
        """PoE method for computing precisions. 
        """
        prec = (1/var_s).sum(0)
        return prec

class GPoE(DgPs):
    """Generalized Product of Experts combining strategy.
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)
    def compute_precision(self, var_s, weights, prior_var):
        """gPoE method for computing precisions. 
        """
        prec = (weights/var_s).sum(0)
        return prec

class BCM(DgPs):
    """Bayesian committee machine combining strategy.
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)
    def compute_precision(self, var_s, weights, prior_var):
        """BCM method for computing precisions. 
        """
        N_models = var_s.shape[0]
        prec = (1/var_s + (1.0-N_models) / prior_var.unsqueeze(-1)).sum(0)
        return prec

class RBCM(DgPs):
    """Robust Bayesian committee machine combining strategy.
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)
    def compute_precision(self, var_s, weights, prior_var):
        """rBCM method for computing precisions. 
        """
        N_models = var_s.shape[0]
        prec = (weights/var_s + (1.0-weights) / prior_var.unsqueeze(-1)).sum(0)
        return prec