from dataloaders.mg_for_roberta import dataset_srt
from models.roberta_finetuning import finetune_RoBERTa
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.train_eval_utils import set_seed, save_config, train, evaluate

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
import utils.mg_utils as utils


class trainer(object):
    """
    Trainer class for finetuning RoBERTa for multi-label emotion recognition on the MovieGraphs dataset.
    """
    def __init__(self, config):
        """
        Initializes the trainer class with the config file.

        Args:
            config (dict): The config file with all the hyperparameters.
        """
        set_seed(config["seed"])
        self.config = config
        data_split = utils.read_train_val_test_splits(self.config["resource_path"])
        self.train_dataset = dataset_srt(self.config, data_split["train"], "train")
        self.emo2id = self.train_dataset.get_emo2id_mapping()
        self.val_dataset = dataset_srt(self.config, data_split["val"], "val", self.emo2id)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.config["batch_size"],
                                           shuffle=True,
                                           num_workers=self.config['num_cpus'],
                                           collate_fn=self.train_dataset.collate)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.config["batch_size"],
                                         shuffle=True,
                                         num_workers=self.config['num_cpus'],
                                         collate_fn=self.val_dataset.collate)
        self.device = torch.device("cuda:{}".format(self.config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.epochs = self.config["epochs"]
        self.model = finetune_RoBERTa(self.train_dataset.top_k, self.config["hugging_face_cache_path"]).to(self.device)
        self.config["model_name"] = "RoBERTa_finetuned_t{}".format(self.train_dataset.top_k)

    def setup_training(self):
        """
        Triggers the training process. This function is not called within the class.
        Wandb is initialized if wandb logging is enabled.
        Optimizer, scheduler and criterion are initialized.
        lr is fixed to 1e-6 for finetuning RoBERTa.
        A train method is called which trains the model.
        """
        if self.config["wandb"]["logging"]:
            wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
            wandb.run.name = self.config["model_name"]
        optimizer = optim.Adam(self.model.parameters(), lr=0.000001)
        criterion = nn.BCEWithLogitsLoss()
        train(epochs=self.config['epochs'], num_labels=self.train_dataset.top_k,
              train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader,
              device=self.device, emo2id=self.emo2id, model=self.model, optimizer=optimizer, scheduler=None,
              criterion=criterion, pred_thresh=self.config["target_prediction_threshold"], masking=True,
              wandb_logging=self.config["wandb"]["logging"], model_name=self.config["model_name"],
              save_path=Path(self.config["save_path"]))


def get_config():
    """
    Loads the config file and overrides the config file with the command line arguments.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    config = get_config()
    print("Current config: {}".format(config))
    save_config(config, Path(config["dumps_path"]), config["model_name"]+"__test_config.yaml")
    obj = trainer(config)
    obj.setup_training()
