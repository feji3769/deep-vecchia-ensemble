"""dve.py
Implementation of deep Vecchia ensemble for 
UCI regression tasks.
Author(s): Felix Jimenez
"""
import os
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np

import mlflow
import hydra
from omegaconf import DictConfig

import deepv
from deepv.combining_strategy import GPoE
from deepv.combining_strategy.weighting import uniform_weight, variance_weight, diff_entropy_weight, wass_weight
from deepv.wrappers import (
    build_scaled_vecchia, 
    fit_scaled_vecchia_validation, 
    get_scaled_vecchia_model
)
from pathlib import Path

from . import utils

def load_net(cfg, net=None):
    if net is None:
        net = hydra.utils.instantiate(cfg.net.model)
    sd_path = utils.save_name(cfg, prefix = cfg.net.prefix, extension = '.pth')
    net.load_state_dict(torch.load(sd_path))
    net.eval()
    return net

def subset_loader(loader, size):
    np.random.seed(42)
    ind = np.arange(len(loader.dataset))
    np.random.shuffle(ind)

    subset = torch.utils.data.Subset(loader.dataset, ind[0:size])
    subset = torch.utils.data.DataLoader(subset, shuffle = False)
    return subset

def fit_model(
    cfg: DictConfig, 
    net:nn.Module, 
    data:tuple
    ) -> nn.Module:

    train_loader, val_loader, test_loader, y_mu, y_std = data
    net = load_net(cfg, net=net)
    net.eval()
    moments = deepv.dve.fit_standardizer(train_loader, net, cfg.dve.layers)


    dve = deepv.dve.DeepV(
        net,
        train_loader,
        moments = moments,
        layers = cfg.dve.layers,
        init_fn = build_scaled_vecchia,
        train_fn = fit_scaled_vecchia_validation,
        build_args = cfg.dve.build_args,
    )
    # take a subset of the validation set for fitting gps.
    #val_subset = subset_loader(val_loader, size=cfg.dve.build_args.n_val)

    dve.fit(train_loader, val_loader)
    return dve

def load_model(cfg, data):
    train_loader, val_loader, test_loader, y_mu, y_std = data
    net = load_net(cfg, net=None)
    net.eval()
    moments = deepv.dve.fit_standardizer(train_loader, net, cfg.dve.layers)
    sd = torch.load(utils.save_name(cfg, prefix=f"{cfg.net.prefix}_dve_moments"))
    moments.load_state_dict(sd)
    dve = deepv.dve.DeepV(
        net,
        train_loader,
        moments=moments,
        layers = cfg.dve.layers,
        init_fn = build_scaled_vecchia,
        train_fn = fit_scaled_vecchia_validation,
        build_args = cfg.dve.build_args,
    )

    sd = torch.load(utils.save_name(cfg, prefix=f"{cfg.net.prefix}_dve"))
    dve.load_state_dict(sd)
    dve.eval()




    return dve

def save_model(cfg, model):

    Path(cfg.storage.weight_path).mkdir(parents=True, exist_ok=True)
    sd_path = utils.save_name(cfg, prefix=f"{cfg.net.prefix}_dve")
    torch.save(model.state_dict(), sd_path)

    sd_path = utils.save_name(cfg, prefix=f"{cfg.net.prefix}_dve_moments")
    torch.save(model.moments.state_dict(), sd_path)


def predict(cfg, model, loader):
    model.eval()
    true = []
    pred_mean = []
    pred_var = []
    with torch.no_grad():
        for x, y in loader:
            try:
                pred_ = model(x)
            except Exception as E:
                # we're in 32 bit which is giving us downstream errors. 
                # this let's us keep moving on. 
                logger.info(f"Exception raised {print(E)}")
                logger.info("Perturbing x.")
                pred_ = model(x + torch.randn_like(x) * 1e-3)

            mu = []
            sig2 = []
            for k,v in pred_.items():
                mu.append(v.mean)
                sig2.append(v.variance)
            mu = torch.cat(mu, axis = 0)
            sig2 = torch.cat(sig2, axis = 0)
            pred_mean.append(mu)
            pred_var.append(sig2)
            true.append(y)

    pred_mean = torch.cat(pred_mean, axis = -1).squeeze()
    pred_var = torch.cat(pred_var, axis = -1).squeeze()
    logger.info(f"mean shape = {pred_mean.shape}")

    combiner = GPoE(diff_entropy_weight)
    pred_dist = combiner(
        pred_mean, pred_var+model.noise.unsqueeze(-1), 0.0 * model.noise,   
        prior_var = model.prior_var+model.noise,
        latent_posterior=True,
        softmax = False,
        power = 1.0
    )

    pred = utils.Output(
        pred = pred_dist.mean,
        sigma = pred_dist.variance.sqrt()
    )

    true = torch.cat(true, axis = 0).squeeze()
    return pred, true


