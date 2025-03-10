"""Adaptation of code from Healing-POEs-ICML Github repo. 

This file contains the base class for combining methods. 

Author : Felix Jimenez
Date : 12/09/2022
"""

import torch
import scipy.special
from scipy.special import logsumexp as lse

class PredictionCombiner():
    """base class for procedures combining ensemble predictions.
    Args:
        weight_strategy (func) : a function that computes unnormalized weights 
                                 (output shape is N_models x N_test). 
    """
    def __init__(self, weight_strategy):
        self.weight_strategy = weight_strategy

    def normalize_weights(self, scores, softmax):
        """get weights by normalizing scores. 

        Use the log-sum exp trick to normalize the 
        scores across models for each test point. 

        Args:
            scores (torch.Tensor): N_models x N_test matrix of scores. 
            softmax (bool): should we use the softmax function? 
        Returns:
            weights (torch.Tensor): N_models x N_test matrix of weights (columns sum to one). 
        """
        if softmax:
            s = torch.nn.Softmax(dim = 0)
            weights = s(scores)
        else:
            weights = scores / scores.sum(0).unsqueeze(0)
        return weights

    def compute_weights(self, mu_s, var_s, power = 1, prior_var=None, softmax=False):
        """compute weights using weight strategy. 
        Use the weight strategy to get unnormalized weights, and then 
        normalize them using the LSE trick. 
        Args:
            mu_s (torch.Tensor): N_models x N_test matrix of predicted means. 
            var_s (torch.Tensor) : N_models x N_test matrix of predicted variances. 
            power (float, optional) : softmax weight. Defaults to 1.
            prior_var (torch.Tensor, optional) : N_models matrix of prior variances. Defaults to None.
            softmax (bool, optional) : Bool indicating softmax should be used. Defaults to False.
        Returns:
            weights (torch.Tensor): N_models x N_test matrix of normalized weights. 
        """
        weights = self.weight_strategy(mu_s, var_s, power, prior_var, softmax)
        weights = self.normalize_weights(weights, softmax)
        return weights

    def add_noise(self, var_s, noise_s):
        """Add the noises to the variances. 
        Combine the noises across models and then add them to the variances. 
        Args: 
            var_s (torch.Tensor): N_test matrix of posterior variances. 
            noise_s (torch.Tensor): N_models x N_test matrix of noises.
        Returns:
            var (torch.Tensor): N_test matrix of variances plus noises. 
        """
        noise_s = torch.mean(noise_s, 0)
        var = var_s + noise_s
        return var
        
    def _combine(self, mu_s, var_s, noise_s, power = 1, prior_var=None, softmax=False, latent_posterior = True):
        """Private method for combining strategies.
        """
        raise NotImplementedError

    def combine(self, mu_s, var_s, noise_s, power = 1, prior_var=None, softmax=False, latent_posterior = True, **kwargs):
        """combine predictions from GPs.
        The public interface for combining GP predictions whose behavior is determined
        by the private method _combine which each subclass must implement. 
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
            prediction (torch.distributions.Normal): Combined marginal predictions. 
        """
        mu, var = self._combine(mu_s, var_s, noise_s, power, prior_var, softmax, latent_posterior)
        prediction = torch.distributions.Normal(mu.squeeze(), var.squeeze().sqrt())
        return prediction


    def __call__(self, mu_s, var_s, noise_s, power = 1, prior_var=None, softmax=False, latent_posterior = True, **kwargs):
        return self.combine(mu_s, var_s, noise_s, power = power, prior_var=prior_var, softmax=softmax, latent_posterior = latent_posterior, **kwargs)