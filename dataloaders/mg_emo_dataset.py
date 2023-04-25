from dataloaders.char_dataset import char_dataset
from dataloaders.scene_dataset import scene_dataset
from dataloaders.srt_dataset import srt_dataset
from pathlib import Path
from torch.utils.data import Dataset
from utils.movie_scene_mapper import movie_scene_mapper

import time
import torch
import yaml


class character_emo_dataset(Dataset):
    """
    Dataset object used to load scene, character and subtitle features along with multilabel targets for emotion recogntion.
    """
    def __init__(self, config, movie_ids, split_type, random_feat_selection=True, with_srt=True, emo2id=dict()):
        """
        Args:
            config (dict): Configuration dictionary.
            movie_ids (list): List of movie ids.
            split_type (str): Split type. Can be train, val or test.
            random_feat_selection (bool): Whether to select features randomly or uniformly based on sampling strategy.
            with_srt (bool): Whether to load subtitle features.
            emo2id (dict): Dictionary mapping emotion to id.
        """
        self.config = config
        self.data_path = Path(config["data_path"])
        self.scene_feat_dim = config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
        self.char_feat_dim = config["feat_info"][config["face_feat_type"]]["face_feat_dim"]
        self.srt_feat_dim = config["feat_info"][config["srt_feat_model"]]["srt_feat_dim"]
        self.feat_sampling_rate = config["feat_sampling_rate"]
        self.num_chars = config["num_chars"]
        self.with_srt = with_srt
        self.max_features = config["max_feats"]
        self.vid_fps = config["vid_fps"]
        self.bin_size = self.feat_sampling_rate/self.vid_fps
        self.num_bins = int(self.config["max_possible_vid_frame_no"]/self.feat_sampling_rate)
        self.fps_boundaries = torch.cumsum(torch.tensor([0]+[self.bin_size]*self.num_bins), dim=0)
        self.split_type = split_type
        self.char_dataset_obj = char_dataset(config, movie_ids, random_feat_selection)
        self.movie_scene_map = movie_scene_mapper(self.config, movie_ids, emo2id,
                                                  char_level=True,
                                                  extended_named_track_info=self.char_dataset_obj.new_named_track_cache,
                                                  emotic_mapped=config["use_emotic_mapping"])
        self.top_k = self.movie_scene_map.top_k
        self.scene_dataset_obj = scene_dataset(config, random_feat_selection)
        self.srt_dataset_obj = srt_dataset(config, movie_ids, self.top_k)

    def get_emo2id_map(self):
        """
        Method to get emotion to id mapping.

        Returns:
            dict: Dictionary mapping emotion to id.
        """
        return self.movie_scene_map.get_emo2id_mapping()

    def collate(self, batches):
        """
        Method to collate batches.
        Based on config parameters, it loads scene, character and subtitle features along with their respective targets.

        Args:
            batches (list): List of batches.

        Returns:
            collated_data (dict): Collated batches with features, masks and targets.
        """
        char_count_in_scenes = [len(batch["targets"][0]) for batch in batches]
        if self.config["get_scene_targets"]:
            scene_target_mask = torch.cat([torch.ones((len(batches), 1, self.top_k)), torch.zeros((len(batches), self.num_chars, self.top_k))], dim=1).flatten().type(torch.bool)
            scene_targets = torch.stack([resp["targets"][1] for resp in batches])
        else:
            scene_target_mask = torch.zeros((len(batches)*(self.num_chars+1)*self.top_k)).type(torch.bool)
            scene_targets = torch.zeros((0, self.top_k))
        if self.config["get_char_targets"]:
            char_targets_raw = [torch.stack(list(resp["targets"][0].values()))[:self.num_chars] for resp in batches]
            char_target_mask = torch.stack([torch.vstack([torch.ones(target.shape), torch.zeros((max(0,self.num_chars-target.shape[0]),self.top_k))]) for target in char_targets_raw])
            char_target_mask = torch.cat([torch.zeros((len(batches), 1, self.top_k)), char_target_mask], dim=1).flatten().type(torch.bool)
            char_targets = torch.cat([torch.stack(list(resp["targets"][0].values())[:self.num_chars]) for resp in batches], dim=0)
        else:
            char_target_mask = torch.zeros((len(batches), self.num_chars+1, self.top_k)).flatten().type(torch.bool)
            char_targets = torch.zeros((0, self.top_k))
        if self.config["use_scene_feats"]:
            scene_feats = torch.stack([resp["feats"][1] for resp in batches]).type(torch.float)
            scene_frames = torch.stack([resp["timestamps"][1] for resp in batches])
            scene_frames = torch.bucketize(scene_frames/self.vid_fps, self.fps_boundaries)
            scene_masks = torch.stack([resp["masks"][1] for resp in batches])
        else:
            scene_feats = torch.zeros((len(batches), self.max_features, self.scene_feat_dim)).type(torch.float)
            scene_frames = torch.zeros((len(batches), self.max_features)).to(torch.long)
            scene_masks = torch.ones((len(batches), self.max_features))
        if self.config["use_char_feats"]:
            if self.config["joint_character_modeling"]:
                empty_char_feat = torch.zeros((self.max_features, self.char_feat_dim))
                padded_char_mask = torch.ones((self.max_features))
                char_feats = [torch.stack(list(resp["feats"][0].values()))[:self.num_chars] for resp in batches]
                char_feats = torch.stack([torch.vstack([feat, empty_char_feat.repeat(max(0,self.num_chars-feat.shape[0]),1,1)]) for feat in char_feats]).type(torch.float)
                char_frames = [torch.stack(list(resp["timestamps"][0].values()))[:self.num_chars] for resp in batches]
                char_frames = torch.stack([torch.vstack([frames, torch.zeros((self.max_features)).repeat(max(0,self.num_chars-frames.shape[0]),1)]) for frames in char_frames])
                char_frames = torch.bucketize(char_frames/self.vid_fps, self.fps_boundaries)
                char_masks = [torch.stack(list(resp["masks"][0].values()))[:self.num_chars] for resp in batches]
                char_masks = torch.stack([torch.vstack([mask, padded_char_mask.repeat(max(0,self.num_chars-mask.shape[0]),1)]) for mask in char_masks])
                if self.config["shuffle_characters"]:
                    shuffled_ndxs = torch.randperm(self.num_chars)
                    char_feats = char_feats[:,shuffled_ndxs,:,:]
                    char_frames = char_frames[:,shuffled_ndxs,:]
                    char_masks = char_frames[:,shuffled_ndxs,:]
                    char_target_mask = char_target_mask.reshape(char_feats.shape[0], self.num_chars+1, self.top_k)[:,1:,:]
                    char_target_mask = char_target_mask[:,shuffled_ndxs,:]
                    char_target_mask = torch.cat([torch.zeros((scene_feats.shape[0], 1, self.top_k)), char_target_mask], dim=1).flatten().type(torch.bool)
                    char_targets = torch.stack([torch.vstack([target, torch.zeros((self.top_k)).repeat(max(0,self.num_chars-target.shape[0]),1)]) for target in char_targets_raw])
                    char_targets = char_targets[:,shuffled_ndxs,:]
                    stacked_tgt = torch.cat([torch.zeros(scene_feats.shape[0], 1, self.top_k), char_targets], dim=1)
                    char_targets = stacked_tgt.flatten()[char_target_mask].reshape(-1, self.top_k)
            else:
                scene_feats = torch.cat([feat.repeat(char_count_in_scenes[ndx], 1, 1) for ndx, feat in enumerate(scene_feats)])
                scene_frames = torch.cat([frames.repeat(char_count_in_scenes[ndx], 1) for ndx, frames in enumerate(scene_frames)])
                scene_masks = torch.cat([mask.repeat(char_count_in_scenes[ndx], 1) for ndx, mask in enumerate(scene_masks)])
                scene_target_mask = torch.cat([mask.repeat(char_count_in_scenes[ndx], 1) for ndx, mask in enumerate(scene_target_mask.reshape(len(batches), self.num_chars+1, self.top_k))]).flatten().to(torch.bool)
                scene_targets = torch.cat([target.repeat(char_count_in_scenes[ndx], 1) for ndx, target in enumerate(scene_targets)]) if scene_targets.shape[0] else scene_targets
                char_feats = torch.cat([torch.stack(list(resp["feats"][0].values())) for resp in batches], dim=0)
                char_feats = torch.cat([char_feats.unsqueeze(1), torch.zeros((char_feats.shape[0], self.num_chars-1, self.max_features, self.char_feat_dim))], dim=1).type(torch.float)
                char_frames = torch.cat([torch.stack(list(resp["timestamps"][0].values())) for resp in batches], dim=0)
                char_frames = torch.cat([char_frames.unsqueeze(1), torch.zeros((char_frames.shape[0], self.num_chars-1, self.max_features))], dim=1)
                char_frames = torch.bucketize(char_frames/self.vid_fps, self.fps_boundaries)
                char_masks = torch.cat([torch.stack(list(resp["masks"][0].values())) for resp in batches], dim=0)
                char_masks = torch.cat([char_masks.unsqueeze(1), torch.ones((char_masks.shape[0]), self.num_chars-1, self.max_features)], dim=1)
                char_target_mask = torch.cat([torch.ones((char_feats.shape[0], 1, self.top_k)), torch.zeros((char_feats.shape[0], self.num_chars-1, self.top_k))], dim=1)
                char_target_mask = torch.cat([torch.zeros((scene_feats.shape[0], 1, self.top_k)), char_target_mask], dim=1).flatten().type(torch.bool)
                char_targets = torch.cat([torch.stack(list(resp["targets"][0].values())) for resp in batches], dim=0)
        else:
            char_feats = torch.zeros((scene_feats.shape[0], self.num_chars, self.max_features, self.char_feat_dim)).type(torch.float)
            char_frames = torch.zeros((scene_feats.shape[0], self.num_chars, self.max_features)).to(torch.long)
            char_masks = torch.ones((scene_feats.shape[0], self.num_chars, self.max_features))
        if self.config["use_srt_feats"]:
            srt_feats = torch.stack([resp["feats"][2] for resp in batches])
            srt_times = torch.stack([resp["timestamps"][2] for resp in batches])
            srt_bins = torch.bucketize(srt_times, self.fps_boundaries)
            srt_masks = torch.stack([resp["masks"][2] for resp in batches])
            if not self.config["joint_character_modeling"]:
                srt_feats = torch.cat([feat.repeat(char_count_in_scenes[ndx], 1, 1) for ndx, feat in enumerate(srt_feats)])
                srt_bins = torch.cat([bins.repeat(char_count_in_scenes[ndx], 1) for ndx, bins in enumerate(srt_bins)])
                srt_masks = torch.cat([mask.repeat(char_count_in_scenes[ndx], 1) for ndx, mask in enumerate(srt_masks)])
        else:
            srt_feats = torch.zeros((scene_feats.shape[0], self.max_features, self.srt_feat_dim))
            srt_times = torch.zeros((scene_feats.shape[0], self.max_features))
            srt_masks = torch.ones((scene_feats.shape[0], self.max_features))
            srt_bins = torch.zeros((scene_feats.shape[0], self.max_features)).to(torch.long)
        collated_data = {
            "feats": [char_feats, scene_feats, srt_feats, char_frames, scene_frames, srt_bins],
            "masks": [char_masks, scene_masks, srt_masks, char_target_mask, scene_target_mask],
            "targets": [char_targets, scene_targets],
        }
        return collated_data

    def __getitem__(self, ndx):
        """
        Returns a dictionary of features, masks, and targets for the given index.

        Args:
            ndx (int): Index of the item to be returned.

        Returns:
            response (dict): A dictionary of features, masks, and targets.
        """
        scenes, targets = self.movie_scene_map[ndx]
        chars = list(targets['chars'].keys())
        scene_feat, scene_feat_mask, scene_frames = None, None, None
        char_feats, char_feat_masks, char_frames = None, None, None
        srt_feats, srt_pad_mask, srt_times = None, None, None
        if self.config["use_scene_feats"]:
            scene_feat, scene_feat_mask, scene_frames = self.scene_dataset_obj.get_scene_feat(scenes)
        if self.config["use_char_feats"]:
            char_feats, char_feat_masks, char_frames = self.char_dataset_obj.get_character_feat(scenes, chars)
        if self.config["use_srt_feats"]:
            srt_feats, srt_times, srt_pad_mask = self.srt_dataset_obj.get_srt_feats(scenes)
        response = {
            "feats": [char_feats, scene_feat, srt_feats],
            "masks": [char_feat_masks, scene_feat_mask, srt_pad_mask],
            "timestamps": [char_frames, scene_frames, srt_times],
            "targets": [targets["chars"], targets["scene"]],
        }
        return response

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.movie_scene_map)
