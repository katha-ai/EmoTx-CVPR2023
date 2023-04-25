from collections import defaultdict
from pathlib import Path
from utils.train_eval_utils import select_feat_ndxs

import itertools
import numpy as np
import os
import pickle
import time
import torch


class char_dataset(object):
    """
    Dataset class to load character features for given movie ids.
    """
    def __init__(self, config, movie_ids, random_feat_selection=True):
        """
        Args:
            config (dict): config dictionary
            movie_ids (list): list of movie ids
            random_feat_selection (bool): Whether to select features randomly or uniformly based on sampling rate.
        """
        self.config = config
        self.data_path = Path(config["data_path"])
        self.tracks_path = self.data_path/config["tracks_dir"]
        self.tracks_name_path = self.data_path/config["track_names_dir"]
        self.track_name_metadata_path = self.data_path/config["track_name_metadata_dir"]
        self.face_feat_path = self.data_path/(config["face_feat_type"]+config["face_feat_dir"])
        self.char_feat_dim = config["feat_info"][config["face_feat_type"]]["face_feat_dim"]
        self.feat_sampling_rate = config["feat_sampling_rate"]
        self.max_features = config["max_feats"]
        self.random_feat_selection = random_feat_selection
        self.new_named_track_cache = self.process_track_name_probablities(movie_ids)

    def process_track_name_probablities(self, movie_ids):
        """
        Process track name probabilities for given movie ids.
        Since the track names were clustered, certain probablity exists that a given name maps to which track.
        A name for a give track is selected if the probability is greater than a threshold. Else, the track is ignored.

        Args:
            movie_ids (list): list of movie ids
        
        Returns:
            new_named_track_cache (dict): dictionary containing track name probabilities for given movie ids.
        """
        cache = dict()
        track_selection_thresh = self.config["character_name_selection_threshold"]
        st = time.perf_counter()
        for movie in movie_ids:
            name_to_ndx, ndx_to_name = self.get_movie_track_name_metadata(movie)
            scenes = os.listdir(self.tracks_path/movie)
            scenes = [".".join(scene.split('.')[:-1]) for scene in scenes]
            named_track_scenes = os.listdir(self.tracks_name_path/movie)
            named_track_scenes = [".".join(scene.split('.')[:-1]) for scene in named_track_scenes]
            cache[movie] = dict()
            for scene in scenes:
                cache[movie][scene] = dict()
                if scene not in named_track_scenes:
                    continue
                named_tracks = np.load(self.tracks_name_path/movie/(scene+".npy"))
                for char, ndx in name_to_ndx.items():
                    track_probab = named_tracks[:, ndx]
                    mask = track_probab > track_selection_thresh
                    if mask.sum():
                        cache[movie][scene][char] = np.nonzero(mask)[0].tolist()
        print("Prepared cache for retrieving new named-tracks in {:.4f}sec.".format(time.perf_counter() - st))
        return cache

    def read_char_tracks(self, scene):
        """
        Read character tracks for given scene.

        Args:
            scene (Path): "movie_id/scene_id" as a path object.

        Returns:
            ptracks (np.ndarray): person body tracks.
            ftracks (np.ndarray): face tracks.
        """
        f = open(self.tracks_path/(str(scene)+".pkl"), "rb")
        ptracks = pickle.load(f)
        ftracks = pickle.load(f)
        f.close()
        return ptracks, ftracks

    def get_movie_track_name_metadata(self, movie):
        """
        Read track name metadata for given movie.
        This meta data includes character name to their corresponding track index and vice-versa in track name vector.

        Args:
            movie (str): movie id.

        Returns:
            name_to_ndx (dict): character name to their corresponding track index.
            ndx_to_name (dict): track index to their corresponding character name.
        """
        f = open(self.track_name_metadata_path/(movie+".pkl"), "rb")
        name_to_ndx = pickle.load(f)
        ndx_to_name = pickle.load(f)
        f.close()
        return name_to_ndx, ndx_to_name

    def distribute_feat_to_track_ids(self, feat, stacked_tracks):
        """
        Distribute face features to their corresponding track ids.

        Args:
            feat (np.ndarray): face features.
            stacked_tracks (np.array): stacked character tracks.

        Returns:
            track_sorted_feats (dict): dictionary containing face features for each track id.
            track_frame_map (dict): dictionary containing frame numbers corresponding to face featues for each track id.
        """
        track_ids = stacked_tracks[:, -1].astype(np.int32)
        frame_nos = stacked_tracks[:, -3].astype(np.int32)
        track_sorted_feats, track_frame_map = dict(), dict()
        for ndx, tr_id in enumerate(track_ids):
            if tr_id in track_sorted_feats:
                track_sorted_feats[tr_id] = np.vstack([track_sorted_feats[tr_id], feat[ndx]])
                track_frame_map[tr_id].append(frame_nos[ndx])
            else:
                track_sorted_feats[tr_id] = feat[ndx]
                track_frame_map[tr_id] = [frame_nos[ndx]]
        for tid, feats in track_sorted_feats.items():
            if len(feats.shape) == 1:
                track_sorted_feats[tid] = np.expand_dims(track_sorted_feats[tid], axis=0)
        return track_sorted_feats, track_frame_map

    def collect_char_feat(self, movie_scene):
        """
        Collect character features and frame numbers for given movie scene mapped to their corresponding track ids.

        Args:
            movie_scene (Path): "movie_id/scene_id" as a path object.

        Returns:
            track_feat_map (dict): dictionary containing face features for each track id.
            track_frame_map (dict): dictionary containing frame numbers corresponding to face featues for each track id.
        """
        ptracks, ftracks = self.read_char_tracks(movie_scene)
        f_feat = np.load(self.face_feat_path/(str(movie_scene)+".npy"))
        track_feat_map, track_frame_map = self.distribute_feat_to_track_ids(f_feat,
                                                                            np.vstack(list(ftracks.values())))
        return track_feat_map, track_frame_map

    def get_character_feat(self, scenes, chars):
        """
        For given scenes and character names, extract character features and frame numbers mapped to their corresponding track ids.

        Args:
            scenes (list): list of scene ids. Each scene id is a path object.
            chars (list): list of character names.

        Returns:
            char_feats (dict): dictionary containing character features for each character.
            char_feat_pad_mask (dict): dictionary containing padding mask for each character.
            selected_frames (dict): dictionary containing frame numbers corresponding to character featues for each character.
        """
        name_to_ndx, ndx_to_name = self.get_movie_track_name_metadata(str(scenes[0].parent))
        char_feats = dict([(char, torch.empty((0, self.char_feat_dim))) for char in chars])
        char_feat_pad_mask, selected_frames = dict(), defaultdict(list)
        for scene in scenes:
            ptracks, ftracks = self.read_char_tracks(scene)
            if not ftracks: # This is required since a ClipGraph can consist of multiple clips among which few may not have faces
                continue
            stacked_ftracks = np.vstack(list(ftracks.values()))
            track_feat_map, track_frame_map = self.collect_char_feat(scene)
            char_tid_map = self.new_named_track_cache[str(scene.parent)][str(scene.name)]
            for char in chars:
                if char not in char_tid_map.keys():
                    continue
                tids = char_tid_map[char]
                frames = list(itertools.chain.from_iterable([track_frame_map[tid+1] for tid in tids]))
                feat = torch.cat([torch.tensor(track_feat_map[tid+1]) for tid in tids])
                char_feats[char] = torch.vstack([char_feats[char], feat])
                selected_frames[char] += frames
        for char in char_feats.keys():
            selected_feat_ndxs = select_feat_ndxs(char_feats[char].shape[0], self.feat_sampling_rate, self.max_features, self.random_feat_selection)
            selected_frames[char] = torch.tensor(selected_frames[char])[selected_feat_ndxs]
            char_feats[char] = char_feats[char][selected_feat_ndxs]
            char_feat_pad_mask[char] = torch.zeros(self.max_features)
            char_feat_pad_mask[char][char_feats[char].shape[0]:] = 1
            pad_len = self.max_features - char_feats[char].shape[0]
            pad_vec = torch.zeros((pad_len, self.char_feat_dim))
            char_feats[char] = torch.vstack([char_feats[char], pad_vec])
            selected_frames[char] = torch.cat([selected_frames[char], torch.zeros(pad_len)])
        return char_feats, char_feat_pad_mask, selected_frames
