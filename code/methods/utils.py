"""utils.py 
common functions for different methods.
"""
import logging
logger = logging.getLogger(__name__)
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import hydra
from dataclasses import dataclass

def scale(x,mu,sig):
    return x# * sig + mu


@dataclass
class Output:
    pred: torch.Tensor | np.ndarray
    sigma: torch.Tensor | np.ndarray | None = None

def rmse(pred, target, mu, sig):
    mse_f = nn.MSELoss()
    rmse = mse_f(
        scale(pred.pred.squeeze(), mu, sig),
        scale(target.squeeze(), mu, sig)
    ).sqrt()
    return rmse


def nll(pred, target, mu, sig):
    if pred.sigma is None:
        return 0.0
    y_hat = pred.pred #(pred.pred*sig+mu).squeeze()
    sig2_hat = pred.sigma ** 2#((pred.sigma ** 2) * (sig ** 2)).squeeze()
    y_og = scale(target.squeeze(), mu, sig)

    nll_f = nn.GaussianNLLLoss(full=True,eps=1e-12,reduction ='none')
    nll = nll_f(y_hat,y_og,sig2_hat)


    return nll.mean()

def compute_metrics(cfg, pred, true, y_mu, y_std):
    losses = {}
    for k,loss_f in cfg.metrics.items():
        losses[k] = hydra.utils.call(loss_f)(pred, true, y_mu, y_std)
    return losses

def entropy(pred):
    if pred.sigma is not None:
        d = torch.distributions.Normal(
            pred.pred,scale = pred.sigma
        )
        return d.entropy()
    else:
        return None

def compute_auroc(id_pred, ood_pred, scoring_fn = entropy):

    id_scores = scoring_fn(id_pred)
    ood_scores = scoring_fn(ood_pred)

    if id_scores is None or ood_scores is None:
        return -1

    id_labs = np.zeros((id_scores.shape[0]))
    ood_labs = np.ones((ood_scores.shape[0]))

    labs = np.concatenate((id_labs, ood_labs), axis = 0)
    scores = -1 * torch.cat((id_scores.squeeze(), ood_scores.squeeze()), axis = 0)

    auroc = roc_auc_score(labs, scores)
    return auroc

def eval_step(cfg, model, loader, y_mu, y_std):
    model.eval()
    pred, true = hydra.utils.call(cfg.method.predict)(cfg, model, loader)
    metrics = compute_metrics(cfg, pred, true, y_mu, y_std)
    return metrics

def save_name(cfg, prefix = "", extension = '.pth'):
    if len(prefix) > 0:
        prefix += "_"
    name_details = "_".join([
        f"{cfg.dataset.name}",
        f"split_{cfg.dataset.seed}",
    ])
    file_name = prefix + name_details + extension
    return os.path.join(cfg.storage.weight_path, file_name)

def mse_loss(pred, target):
    loss_f = nn.MSELoss()
    return loss_f(pred.squeeze(), target.squeeze())

def rmse_loss(pred, target):
    loss_f = nn.MSELoss()
    return loss_f(pred.squeeze(), target.squeeze()).sqrt()

def nll_loss(pred, target):
    loss_f = nn.GaussianNLLLoss()
    return loss_f(pred[:, 0].squeeze(), target.squeeze(), pred[:, 1].squeeze())