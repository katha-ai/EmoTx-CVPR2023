from dataloaders.mg_emo_dataset import character_emo_dataset
from models.mlp import mlp_char, mlp_scene
from models.tx_encoder import tx_char, tx_scene
from models.emotx_1cls import Emotx_1CLS
from models.emotx import EmoTx
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.train_eval_utils import set_seed, save_config, train

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.mg_utils as utils
import wandb
import yaml


class trainer(object):
    def __init__(self, config):
        """
        Initialize the trainer class with the config file needed for training.
        Sets the seed, device, dataloaders and model.

        Args:
            config (dict): Config file containing all the hyperparameters for training.
        """
        set_seed(config["seed"])
        self.config = config
        self.config_sanity_check()
        data_split = utils.read_train_val_test_splits(config["resource_path"])
        self.train_dataset = character_emo_dataset(config=config,
                                                   movie_ids=data_split["train"],
                                                   split_type="train",
                                                   random_feat_selection=config["random_feat_selection"],
                                                   with_srt=config["use_srt_feats"])
        self.emo2id = self.train_dataset.get_emo2id_map()
        self.val_dataset = character_emo_dataset(config=config,
                                                 movie_ids=data_split["val"],
                                                 split_type="val",
                                                 random_feat_selection=False,
                                                 with_srt=config["use_srt_feats"],
                                                 emo2id=self.emo2id)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           num_workers=config["num_cpus"],
                                           collate_fn=self.train_dataset.collate)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=config["batch_size"],
                                         shuffle=False,
                                         num_workers=config["num_cpus"],
                                         collate_fn=self.val_dataset.collate)
        self.device = torch.device('cuda:{}'.format(config["gpu_id"]) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.epochs = config["epochs"]
        self.scene_feat_dim = config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
        self.char_feat_dim = config["feat_info"][config["face_feat_type"]]["face_feat_dim"]
        self.srt_feat_dim = int(config["feat_info"][config["srt_feat_model"]]["srt_feat_dim"])
        self.num_pos_embeddings = int(config["max_possible_vid_frame_no"]/config["feat_sampling_rate"])
        if config["model_no"] == 1.1:
            self.model = mlp_char(self.train_dataset.top_k, self.char_feat_dim, self.device).to(self.device)
        elif config["model_no"] == 1.2:
            self.model = mlp_scene(self.train_dataset.top_k, self.scene_feat_dim, self.device).to(self.device)
        elif config["model_no"] == 2.1:
            self.model = tx_char(self.train_dataset.top_k, self.char_feat_dim, self.device).to(self.device)
        elif config["model_no"] == 2.2:
            self.model = tx_scene(self.train_dataset.top_k, self.scene_feat_dim, self.device).to(self.device)
        elif config["model_no"] == 3.0:
            self.model = Emotx_1CLS(self.train_dataset.top_k,
                                    self.num_pos_embeddings,
                                    self.scene_feat_dim,
                                    self.char_feat_dim,
                                    self.srt_feat_dim,
                                    config["num_enc_layers"],
                                    config["num_chars"],
                                    config["max_feats"]).to(self.device)
        elif config["model_no"] == 4.0:
            self.model = EmoTx(self.train_dataset.top_k,
                               self.num_pos_embeddings,
                               self.scene_feat_dim,
                               self.char_feat_dim,
                               self.srt_feat_dim,
                               config["num_enc_layers"],
                               config["num_chars"],
                               config["max_feats"]).to(self.device)
        else:
            raise NotImplementedError("Given model number does not exists")
        model_id2st_map = {
            "1.1": "Char_MLP",
            "1.2": "Scene_MLP",
            "2.1": "Char_TxEncoder",
            "2.2": "Scene_TxEncoder",
            "3.0": "Emotx_1CLS",
            "4.0": "Emotx"
        }
        print("Selected model {}".format(model_id2st_map[str(config["model_no"])]))
        self.save_path = Path(config["save_path"])

    def config_sanity_check(self):
        """
        Sanity check for the config file to ensure that the config file is correct and no errors are made in the config file by the user.
        Check 1: If the user has selected to use scene or char features or srt features. Atleast one of them should be enabled.
        Check 2: If the user has selected to get scene or char targets. Atleast one of them should be enabled.
        Check 3: If the user has selected to use srt features, then the model number should be 3.0 or 4.0.
        Check 4: If the user has selected to use srt features, then the user should select to use srt cls only if model number is 3.0 or 4.0.
        Check 5: If model number is 3.0 or 4.0, the max_feats must be same as max_srt_feats.
        Check 6: If model number is 1.1 or 2.1, then the user should not select to use srt features and scene features.
                 Moreover joint_character_modeling should be disabled and only character targets must be enabled.
        Check 7: If model number is 1.2 or 2.2, then the user should not select to use srt features and char features. Moreover only scene targets must be enabled.
        """
        assert(self.config["use_scene_feats"] or self.config["use_char_feats"] or self.config["use_srt_feats"])
        assert(self.config["get_scene_targets"] or self.config["get_char_targets"])
        assert((self.config["model_no"] in [3.0, 4.0] and self.config["use_srt_feats"]) or not self.config["use_srt_feats"])
        assert(self.config["model_no"] not in [3.0, 4.0] or (self.config["model_no"] in [3.0, 4.0] and \
                                                             self.config["use_srt_cls_only"]))
        assert(self.config["model_no"] not in [3.0, 4.0] or (self.config["model_no"] in [3.0, 4.0] and \
                                                             self.config["max_feats"] == self.config["max_srt_feats"]))
        assert(self.config["model_no"] not in [1.1, 2.1] or (self.config["model_no"] in [1.1, 2.1] and \
                                                             self.config["use_char_feats"] and \
                                                             self.config["get_char_targets"] and \
                                                             not self.config["joint_character_modeling"] and \
                                                             not self.config["use_scene_feats"] and \
                                                             not self.config["use_srt_feats"] and \
                                                             not self.config["get_scene_targets"]))
        assert(self.config["model_no"] not in [1.2, 2.2] or (self.config["model_no"] in [1.2, 2.2] and \
                                                             self.config["use_scene_feats"] and \
                                                             self.config["get_scene_targets"] and \
                                                             not self.config["use_char_feats"] and \
                                                             not self.config["use_srt_feats"] and \
                                                             not self.config["get_char_targets"]))

    def trigger_training(self):
        """
        Triggers the training process. This function is not called within the class.
        Wandb is initialized if wandb logging is enabled.
        Optimizer, scheduler and criterion are initialized.
        A train method is called which trains the model.
        """
        if self.config["wandb"]["logging"] and (not self.config["wandb"]["sweeps"]):
            wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"], config=self.config)
            wandb.run.name = self.config["model_name"]
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=0.001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(self.config["pos_weight"][str(self.train_dataset.top_k)]).to(self.device))
        train(epochs=self.epochs, num_labels=self.train_dataset.top_k,
              train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader,
              device=self.device, emo2id=self.emo2id, model=self.model, optimizer=optimizer, scheduler=scheduler,
              criterion=criterion, pred_thresh=self.config["target_prediction_threshold"], masking=True,
              wandb_logging=self.config["wandb"]["logging"], model_name=self.config["model_name"], save_path=Path(self.config["save_path"]))


def fill_model_name(config):
    """
    Fills the model name based on the config file.
    """
    target_type = {(True, True): "SC", (True, False): "S", (False, True): "C"}
    random_frame_selection = {True: "R", False: "nR"}
    feat_type = {"emo": 'e', "generic": 'g', "clip": 'c', "resnet50_places": 'r', "resnet50_fer": 'r', "concat": "co", "independent": "in", "mvit_v1": "m"}
    srt_feat_type = {True: 'P', False: 'F'}
    srt_feat_model = {'roberta': 'r', 'clip': 'c'}
    if float(config["model_no"]) in [3.0, 4.0]:
        model_no = '_'.join(str(config["model_no"]).split('.'))
        model_name = "M{}.L{}.N{}.e{}.t{}.{}S.{}C.{}{}Sr.{}".format(model_no,
                                                                    config["num_enc_layers"],
                                                                    config["num_chars"],
                                                                    len(np.format_float_positional(config["lr"]))-2,
                                                                    config["top_k"] if not config["use_emotic_mapping"] else 26,
                                                                    feat_type[config["scene_feat_type"]],
                                                                    feat_type[config["face_feat_type"]],
                                                                    feat_type[config["srt_feat_type"]],
                                                                    srt_feat_type[config["srt_feat_pretrained"]]+srt_feat_model[config["srt_feat_model"]],
                                                                    config["feat_sampling_rate"])
    else:
        model_no = '_'.join(str(config["model_no"]).split('.'))
        model_name = "M{}.{}.t{}".format(model_no,
                                         feat_type[config["face_feat_type"]],
                                         config["top_k"] if not config["use_emotic_mapping"] else 26)
    model_name += "_local" if not config["wandb"]["logging"] else ""
    model_name += "_{}".format(config["model_name_suffix"])
    return model_name


def get_config():
    """
    Loads the config file and updates it with the command line arguments.
    The model name is also updated. The config is then converted to a dictionary.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    OmegaConf.update(updated_conf, "model_name", fill_model_name(updated_conf))
    return OmegaConf.to_container(updated_conf)


def sweep_agent_manager():
    """
    This function is called by wandb agent. It initializes wandb and then calls the trainer class.
    Run name is also updated.
    """
    wandb.init()
    config = dict(wandb.config)
    run_name = fill_model_name(config)
    wandb.run.name = run_name
    config["model_name"] = run_name
    obj = trainer(config)
    obj.trigger_training()


if __name__ == "__main__":
    config = get_config()
    if config["wandb"]["sweeps"]:
        wandb.agent(sweep_id=config["wandb"]["sweep_id"], function=sweep_agent_manager, count=config["wandb"]["sweep_run_count"])
    else:
        print("Current config: {}".format(yaml.dump(config)))
        save_config(config, Path(config["dumps_path"]), config["model_name"]+"__test_config.yaml")
        obj = trainer(config)
        obj.trigger_training()
