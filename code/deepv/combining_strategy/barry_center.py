"""Adaptation of code from Healing-POEs-ICML Github repo. 

This file contains the implementation of Barry center combining methods. 

Author : Felix Jimenez
Date : 12/09/2022
"""

import torch
from .prediction_combiner import PredictionCombiner

class BarryCenter(PredictionCombiner):
    """Barry center combining method.
    Args:
        weight_strategy (func) : a function that computes unnormalized weights 
                                 (output shape is N_models x N_test). 
    """
    def __init__(self, weight_strategy):
        super().__init__(weight_strategy)

    def _combine(self, mu_s, var_s, noise_s, power = 1, prior_var=None, softmax=False, latent_posterior = True):
        """Combine predictions using Barry centers.
        Use the 1D Barry center combining strategy to combing means and variances. 
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
        mu = (weights * mu_s).sum(0)
        var = (weights * var_s).sum(0)
        return mu, var
