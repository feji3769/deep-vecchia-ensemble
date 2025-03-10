"""net.py
Neural network implementation for 
UCI regression.
"""
from pathlib import Path
import sys
import os

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from . import utils
from copy import deepcopy

def shuffle_loader(original_dataloader):
    dataset = original_dataloader.dataset
    dataloaders = []
    indices = torch.arange(len(dataset))
    #torch.randint(
    #    high=len(dataset), size=(len(dataset),), dtype=torch.int64
    #)
    sub_dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=original_dataloader.batch_size,
        num_workers=original_dataloader.num_workers,
        collate_fn=original_dataloader.collate_fn,
        shuffle=True,
    )
    return dataloader

def fit_model(
    cfg: DictConfig, 
    net:nn.Module, 
    data:tuple
    ) -> nn.Module:

    if torch.cuda.is_available():
        net = net.cuda()
    _=net.train()
    train_loader, val_loader, test_loader, y_mu, y_std = data
    train_loader = shuffle_loader(train_loader)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.hyps.lr, 
        weight_decay = cfg.hyps.weight_decay
    )

    opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        eta_min=cfg.hyps.eta_min,
        T_max=cfg.hyps.num_epochs
    )

    loss_f = hydra.utils.instantiate(cfg.method.loss_fn)

    pbar = tqdm(
        range(cfg.hyps.num_epochs), 
        file=open(os.devnull, 'w')
    )
    best_loss = 10e3;
    sd = net.state_dict()
    for i in pbar:
        net.train()
        # Within each iteration, we will go over each minibatch of data
        run_train_loss = []

        for x_batch, y_batch in train_loader:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            
            optimizer.zero_grad()
            pred = net(x_batch)
            loss = loss_f(pred, y_batch)
            loss.backward()
            optimizer.step()

            run_train_loss.append(loss.item())

        opt_sched.step()
        train_loss = sum(run_train_loss) / len(run_train_loss)

        mlflow.log_metric("train/loss", train_loss)
        printout = f"lr {opt_sched.get_last_lr()[0]:.2E} "
        printout += f"train/loss {train_loss:.3f} "

        net.eval()
        val_loss = utils.eval_step(cfg, net, val_loader, y_mu, y_std)
        
        for loss_n,loss_v in val_loss.items():
            mlflow.log_metric(f"val/{loss_n}", loss_v)
            printout += f"val/{loss_n} {loss_v:.3f} "
            printout += f"best/loss {best_loss:.3f} "
        pbar.set_description(printout)
        logger.info(str(pbar))
        if val_loss['rmse'] < best_loss:
            best_loss = val_loss['rmse']
            sd = deepcopy(net.state_dict())

    net.load_state_dict(sd)
    return net

def save_model(
        cfg: DictConfig,
        model: nn.Module
    )-> None:
    Path(cfg.storage.weight_path).mkdir(parents=True, exist_ok=True)
    path = utils.save_name(cfg, prefix=cfg.net.prefix)
    torch.save(
        model.state_dict(), 
        path
    )


def load_model(cfg: DictConfig, model = None):
    net = hydra.utils.instantiate(cfg.net.model)
    sd = torch.load(utils.save_name(cfg, prefix=cfg.net.prefix))
    
    net.load_state_dict(sd)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def predict(
    cfg: DictConfig, 
    model: nn.Module, 
    loader: DataLoader
    )-> torch.Tensor:

    model.eval()
    pred = []; true = []; sigma = []
    with torch.no_grad():
        for x,y in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            pred_ = model(x)
            if pred_.shape[1] > 1:
                pred.append(pred_[:, 0].cpu().squeeze())
                sigma.append(pred_[:, 1].cpu().squeeze())
            else:
                pred.append(pred_.cpu())

            true.append(y.cpu().squeeze())
    true = torch.cat(true, axis = 0)
    if len(sigma) > 0:
        sigma = torch.cat(sigma, axis = 0).squeeze().sqrt()
    else:
        sigma = None
    pred = utils.Output(
        pred = torch.cat(pred, axis = 0).squeeze(),
        sigma = sigma
    )

    return pred, true