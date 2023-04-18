from pathlib import Path
from utils.train_eval_utils import select_feat_ndxs

import numpy as np
import torch


class scene_dataset(object):
    """
    Dataset class to load scene features.
    """
    def __init__(self, config, random_feat_selection=True):
        """
        Args:
            config (dict): Configuration dictionary.
            random_feat_selection (bool): Whether to select features randomly or uniformly based on sampling rate.
        """
        self.config = config
        self.data_path = Path(config["data_path"])
        self.scene_feat_path = self.data_path/(config["scene_feat_type"]+config["scene_feat_dir"])
        self.scene_feat_dim = config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
        self.max_features = config["max_feats"]
        self.feat_sampling_rate = config["feat_sampling_rate"]
        self.random_feat_selection = random_feat_selection


    def get_scene_feat(self, scenes):
        """
        Get scene features for a list of scenes.
        if scene_feat_type is mvit_v1, then scene features are selected based on action features.
        else scene features are selected based on sampling strategy from some pretrained model.

        Args:
            scenes (list): List of scenes.

        Returns:
            scene_feat (torch.Tensor): Scene features. Shape: (num_scenes, num_features, feat_dim).
            selected_frames (torch.Tensor): Selected frames. Shape: (num_scenes, num_features).
            padding_mask (torch.Tensor): Padding mask. Shape: (num_scenes, num_features).
        """
        scene_feat = torch.empty((0, self.scene_feat_dim))
        selected_frames = None
        padding_mask = torch.zeros(self.max_features)
        for scene in scenes:
            feat = np.load(self.scene_feat_path/(str(scene)+".npy"))
            scene_feat = torch.vstack([scene_feat, torch.tensor(feat)])
        frame_ndxs = select_feat_ndxs(scene_feat.shape[0], self.feat_sampling_rate, self.max_features, self.random_feat_selection)
        if self.config["scene_feat_type"] != "mvit_v1":
            selected_frames = frame_ndxs
            scene_feat = scene_feat[frame_ndxs]
        else:
            selected_frames = torch.cumsum(torch.tensor([self.config["action_feat_group_size"]//2]+[self.config["action_feat_group_size"]//2]*(scene_feat.shape[0]-1)), dim=0)
            scene_feat = scene_feat[:self.max_features]
            selected_frames = selected_frames[:self.max_features]
        padding_mask[scene_feat.shape[0]:] = 1
        padding_len = max(0, self.max_features - scene_feat.shape[0])
        pad_vec = torch.zeros((padding_len, self.scene_feat_dim))
        scene_feat = torch.vstack([scene_feat, pad_vec])
        selected_frames = torch.cat([selected_frames, padding_mask[:padding_len]])
        return scene_feat, padding_mask, selected_frames
