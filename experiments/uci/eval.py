"""eval.py
Evaluating different methods on UCI regression tasks.

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
from methods import utils

import mlflow
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base="1.2", config_path="conf",config_name="eval")
def main(cfg):
    logger.info(f"Evaluating on {cfg.dataset.name}, split {cfg.dataset.seed}")

    # load data. 
    data = uci_dataloaders(
        cfg.dataset.folder, cfg.hyps.batch_size
    )
    train_loader, val_loader, test_loader, y_mu, y_std = data

    model = hydra.utils.call(cfg.method.load_model)(cfg, data)

    with mlflow.start_run():
        mlflow.log_params(cfg)

        # metrics on train.
        train_loss = utils.eval_step(cfg, model, train_loader, y_mu, y_std)
        printout = ""
        for loss_n,loss_v in train_loss.items():
            mlflow.log_metric(f"train/{loss_n}", loss_v)
            printout += f"train/{loss_n} {loss_v:.3f} "
        logger.info(printout)      




        # metrics on validation.
        val_pred, val_true = hydra.utils.call(cfg.method.predict)(cfg, model, val_loader)
        val_metrics = utils.compute_metrics(cfg, val_pred, val_true, y_mu, y_std)
        printout = ""
        for loss_n,loss_v in val_metrics.items():
            mlflow.log_metric(f"val/{loss_n}", loss_v)
            printout += f"val/{loss_n} {loss_v:.3f} "
        logger.info(printout)

        # metrics on test.
        test_pred, test_true = hydra.utils.call(cfg.method.predict)(cfg, model, test_loader)
        test_metrics = utils.compute_metrics(cfg, test_pred, test_true, y_mu, y_std)
        printout = ""
        for loss_n,loss_v in test_metrics.items():
            mlflow.log_metric(f"test/{loss_n}", loss_v)
            printout += f"test/{loss_n} {loss_v:.3f} "
        logger.info(printout)



if __name__ == "__main__":
    main()