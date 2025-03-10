"""Adaptation of code from Healing-POEs-ICML Github repo. 

This file contains the weighting procedures. 
All weight matrices are assumed to be normalized after this. 

Author : Felix Jimenez
Date : 12/09/2022

[1]: 
Cohen, Samuel, et al. "Healing products of Gaussian process experts." 
International Conference on Machine Learning. PMLR, 2020.
"""

import torch
import scipy.special
from scipy.special import logsumexp as lse


def uniform_weight(mu_s, var_s, power, prior_var, softmax):
    # uniform weithting, not PoE because weights will sum to one. 
    w = torch.ones(mu_s.shape) / mu_s.shape[0]
    return w

def variance_weight(mu_s, var_s, power, prior_var, softmax):
    if softmax:
        # posterior var. temp. end. softmax [1]
        weight_matrix = -power * var_s
    else:
        # posterior var (small posterior var is better, so invert)
        weight_matrix = torch.clip(1/var_s, min = 1e-12, max = 1e12)

    return weight_matrix

def wass_weight(mu_s, var_s, power, prior_var, softmax):
    # wass. dist. b/w normals.
    wass = mu_s**2 + (var_s - prior_var.unsqueeze(-1))**2

    if softmax:
        # wass temp. end. softmax [1]
        weight_matrix = power * wass
    else:
        # wass. weight
        weight_matrix = wass
    return weight_matrix

def diff_entropy_weight(mu_s, var_s, power, prior_var, softmax):
    # differential entropy [1].
    delta_H =  0.5 * (torch.log(prior_var.unsqueeze(-1)) - torch.log(var_s))
    if softmax:
        # differetnial entropy with temp. end. softmax [1]
        weight_matrix = power * delta_H
    else:
        # differetnial entropy
        weight_matrix = delta_H
    return weight_matrix