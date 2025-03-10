"""train.py
Fitting different methods to UCI regression tasks.

Author: Felix Jimenez
"""
import logging
logger = logging.getLogger(__name__)

import os
from pathlib import Path
import sys
sys.path.append("/workspace/project/public/code/")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import uci_dataloaders
from methods.utils import compute_metrics

import mlflow
from tqdm import tqdm
import torch


import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="conf",config_name="train")
def main(cfg):
    torch.manual_seed(cfg.dataset.seed)
    logger.info(f"Training on {cfg.dataset.name}, split {cfg.dataset.seed}")

    # load data. 
    data = uci_dataloaders(
        cfg.dataset.folder, cfg.hyps.batch_size
    )
    train_loader, val_loader, test_loader, y_mu, y_std = data
    logger.info(f"number train {len(train_loader.dataset)}")
    model = hydra.utils.instantiate(cfg.net.model)

    with mlflow.start_run():
        mlflow.log_params(cfg)
        model = hydra.utils.call(cfg.method.fit_model)(cfg, model, data)

        y_val_pred, y_val  = hydra.utils.call(cfg.method.predict)(cfg, model, val_loader)
        val_loss = compute_metrics(cfg, y_val_pred, y_val, y_mu, y_std)
        
        printout = ""
        for loss_n,loss_v in val_loss.items():
            mlflow.log_metric(f"final_val/{loss_n}", loss_v)
            printout += f"final_val/{loss_n} {loss_v:.3f} "
        logger.info(printout)

        hydra.utils.call(cfg.method.save_model)(cfg, model)
        model = hydra.utils.call(cfg.method.load_model)(cfg, data)

        y_val_pred, y_val  = hydra.utils.call(cfg.method.predict)(cfg, model, val_loader)
        val_loss = compute_metrics(cfg, y_val_pred, y_val, y_mu, y_std)
        
        printout = "after reloading model: "
        for loss_n,loss_v in val_loss.items():
            mlflow.log_metric(f"final_val/{loss_n}", loss_v)
            printout += f"final_val/{loss_n} {loss_v:.3f} "
        logger.info(printout)
        return val_loss['rmse']
    
if __name__ == "__main__":
    main()