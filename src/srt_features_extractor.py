from models.roberta_finetuning import featExtract_finetuned_RoBERTa, featExtract_pretrained_RoBERTa
from omegaconf import OmegaConf
from pathlib import Path
from transformers import RobertaTokenizer
from utils.scene_srt_reader import clip_srt

import os
import pickle
import torch
import time
import yaml


class srts_feat_extraction(object):
    """
    Class for extracting features from the finetuned or pretrained RoBERTa models.
    """
    def __init__(self, config):
        """
        Initializes the class with the config file.

        Args:
            config (dict): The config file with all the hyperparameters.
        """
        self.config = config
        self.concat = config["srt_feat_type"] == "concat"
        self.srts_path = Path(config["clip_srts_path"])
        self.srts_feat_save_path = Path(config["save_path"])/"srt_feats"
        self.srts_feat_save_path = self.srts_feat_save_path/"concat" if self.concat else self.srts_feat_save_path/"independent"
        self.top_k = config["top_k"] if not config["use_emotic_mapping"] else 26
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=config["hugging_face_cache_path"])
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else 'cpu')
        if not config["srt_feat_pretrained"]:
            print("Selected finetuned RoBERTa model for top-{} emotions".format(self.top_k))
            self.srts_feat_save_path = self.srts_feat_save_path/("finetuned_t{}".format(self.top_k))
            model_name = "RoBERTa_finetuned_t{}.pt".format(self.top_k)
            self.model = featExtract_finetuned_RoBERTa(self.top_k, Path(config["saved_model_path"])/model_name)
        else:
            print("Selected the pretrained-RoBERTa-base model.")
            self.srts_feat_save_path = self.srts_feat_save_path/"pretrained"
            self.model = featExtract_pretrained_RoBERTa(config["hugging_face_cache_path"])
        self.model = self.model.eval().to(self.device)

    def save_feats(self, save_path, scene, times, feats):
        """
        Saves the timestamps for every utterance and the extracted features to a pickle file.
        """
        with open(save_path/(scene+".pkl"), 'wb') as f:
            pickle.dump(times, f)
            pickle.dump(feats, f)

    @torch.no_grad()
    def extract_features(self, movies):
        """
        Extracts the utterance features for the given list of movies.

        Args:
            movies (list): List of movies for which the features are to be extracted.
        """
        srt_reader_obj = clip_srt(self.config, movies)
        pst = time.perf_counter()
        for movie in movies:
            save_path = self.srts_feat_save_path/movie
            save_path.mkdir(parents=True, exist_ok=True)
            st = time.perf_counter()
            print("Started extracting features for: {}".format(movie), end=" | ")
            srt_files = os.listdir(self.srts_path/movie)
            scenes = [".".join(filename.split(".")[:-1]) for filename in srt_files]
            for scene in scenes:
                srts = srt_reader_obj.get_srt(Path(movie)/scene, concat=self.concat)
                times, feats = list(), list()
                if not self.concat:
                    times = [pair[0] for pair in srts]
                    srts = [pair[1] for pair in srts]
                if srts:
                    tokenized_srts = self.tokenizer(srts, padding='longest', truncation=True, return_tensors="pt")
                    feats = self.model(tokenized_srts["input_ids"].to(self.device), tokenized_srts["attention_mask"].to(self.device))
                    feats = feats.detach().cpu()
                self.save_feats(save_path, scene, times, feats)
            print("Completed in {:.4f} sec.".format(time.perf_counter()-st))
        print("Finished extraction in {:.4f} sec.".format(time.perf_counter()-pst))


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    cnfg = get_config()
    mvs = os.listdir(cnfg["clip_srts_path"])
    obj = srts_feat_extraction(cnfg)
    obj.extract_features(mvs)
