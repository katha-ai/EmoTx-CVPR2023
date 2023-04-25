from pathlib import Path
from transformers import RobertaTokenizer
from utils.scene_srt_reader import clip_srt

import pickle
import torch


class srt_dataset(object):
    """
    Dataset class to load subtitle features for given movie ids.
    """
    def __init__(self, config, movie_ids, top_k):
        """
        Args:
            config (dict): config file
            movie_ids (list): list of movie ids
            top_k (int): top k clips to use
        """
        self.config = config
        self.feat_model = config["srt_feat_model"]
        self.data_path = Path(config["data_path"])
        self.srt_feat_path = self.data_path/config["srt_feat_dir"]
        self.max_features = config["max_srt_feats"]
        self.srt_feat_dim = config["feat_info"][self.feat_model]["srt_feat_dim"]
        self.top_k = top_k
        self.srt_type = config["srt_feat_type"]
        self.srt_feats_dir = "pretrained" if config["srt_feat_pretrained"] else "finetuned_t{}".format(self.top_k)
        self.use_cls_only = config["use_srt_cls_only"]
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=config["hugging_face_cache_path"])
        self.clip_srt_obj = clip_srt(config, movie_ids)

    def get_srt_feats(self, scenes):
        """
        Operates in two modes, depending on the value of self.use_cls_only.
        If self.use_cls_only is True, it returns the CLS emb. combined/independent uttarances from the dialogue. Used to get features extracted from pretrained/finetuned model.
        If self.use_cls_only is False, it returns tokenized ids of words within the utterance. Used as an input to Roberta model.

        Args:
            scenes (list): list of scenes
        
        Returns:
            srt_feats (torch.tensor): srt features. Shape (max_features, srt_feat_dim)
            times (torch.tensor): mid timestamp for individual utterances. Shape (max_features,)
            srt_pad_mask (torch.tensor): padding mask. Shape (max_features,)
        """
        srt_pad_mask = torch.zeros(self.max_features)
        if self.use_cls_only:
            srt_feats = torch.empty((0, self.srt_feat_dim))
            times = torch.empty((0,))
            for scene in scenes:
                with open(self.srt_feat_path/self.srt_type/self.srt_feats_dir/(str(scene)+".pkl"), 'rb') as f:
                    time = pickle.load(f)
                    feats = pickle.load(f)
                if len(feats):
                    srt_feats = torch.cat([srt_feats, feats], dim=0)
                    times = torch.cat([times, torch.tensor(time)], dim=0)
            srt_feats = srt_feats[:self.max_features, :]
            times = times[:self.max_features]
            srt_pad_mask[srt_feats.shape[0]:] = 1
            pad_len = self.max_features - srt_feats.shape[0]
            srt_feats = torch.cat([srt_feats, torch.zeros(pad_len, srt_feats.shape[-1])])
            times = torch.cat([times, torch.zeros([pad_len])])
        else:
            srts = list()
            times = list()
            for scene in scenes:
                time_srt_pairs = self.clip_srt_obj.get_srt(scene, concat=False)
                for time, srt in time_srt_pairs:
                    enc = self.tokenizer(srt, return_tensors="pt")
                    srts.extend(enc.input_ids)
                    times.extend([time]*enc.input_ids.shape[1])
            srt_feats = torch.cat(srts, dim=0)[:self.max_features] if srts else torch.empty((0,))
            times = torch.tensor(times)[:self.max_features] if times else torch.empty((0,))
            srt_pad_mask[srt_feats.shape[0]:] = 1
            pad_len = self.max_features - srt_feats.shape[0]
            srt_feats = torch.cat([srt_feats, torch.zeros(pad_len)]).to(torch.int)
            times = torch.cat([times, torch.zeros([pad_len])])
        return srt_feats, times, srt_pad_mask
