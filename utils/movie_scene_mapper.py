from collections import defaultdict
from pathlib import Path

import utils.mg_utils as utils
import sys
import torch
import yaml


class movie_scene_mapper(object):
    """
    This class is used to get movie_id/scene Path objects and corresponding scene and character emotion targets.
    """
    def __init__(self, config, movie_ids, emo2id=dict(), char_level=False, extended_named_track_info=None, emotic_mapped=False):
        """
        Args:
            config (dict): Configuration dictionary.
            movie_ids (list): List of movie ids.
            emo2id (dict): Dictionary mapping emotion to id.
            char_level (bool): Whether to return character targets or not.
            extended_named_track_info (dict): Dictionary containing extended named track information.
            emotic_mapped (bool): Whether to use emotic emotions or not.
        """
        self.config = config
        self.emotic_mapped = emotic_mapped
        self.movie_ids = movie_ids
        self.movie_graphs = utils.read_mg_pkl(config["pkl_path"], config["pkl_name"])
        self.custom_mapping = None if not self.emotic_mapped else utils.get_emotic_emotions_map(config["emotic_mapping_path"])
        self.top_k_emotions = self.get_top_k_emotions()
        self.emo2id = emo2id
        self.char_level = char_level
        self.new_named_tracks = extended_named_track_info
        self.top_k = len(self.top_k_emotions) if not self.emotic_mapped else len(self.custom_mapping)
        self.cgs = dict()
        self.scenes = list()
        self.targets = list()
        self.prepare_mappings()

    def get_top_k_emotions(self):
        """
        Get top k emotions from the movie graphs.
        Operated in two modes: emotic_mapped and non-emotic_mapped.
        In emotic_mapped mode, the MovieGraphs emotions are mapped to the emotic emotions.
        In non-emotic_mapped mode, the MovieGraphs emotions are used as is, based on top-k values.

        Returns:
            top_k_emotions (list): list of top k emotions
        """
        top_k_emotions = list()
        if self.emotic_mapped:
            for _, emos in self.custom_mapping.items():
                top_k_emotions += emos
        else:
            top_k_emotions = utils.get_top_k_emotions(self.movie_graphs, self.config["top_k"])
        return top_k_emotions

    def filter_cgs(self):
        """
        Filter the clip graphs from movies to retain only the clips that have the top-k emotions.
        """
        all_mapped_clip_graphs = utils.get_clip_graphs(self.movie_graphs, self.movie_ids)
        for m_id, cgs in all_mapped_clip_graphs.items():
            filtered_clip_graphs, _, _ = utils.get_emo_filtered_cgs(cgs, self.top_k_emotions)
            self.cgs[m_id] = filtered_clip_graphs

    def map_emo_with_id(self):
        """
        Map the emotions to ids based on emotic-mapped bool flag.
        In case of emotic-mapped, 181 MovieGraphs emotions are mapped to respective 26 Emotic classes.
        In case of non-emotic-mapped, the top-k emotions are mapped to ids (based of ranking).
        """
        if self.emotic_mapped:
            for ndx, pair in enumerate(self.custom_mapping.items()):
                group_label, emos = pair
                for emo in emos:
                    self.emo2id[emo] = ndx
        else:
            for ndx, emo in enumerate(self.top_k_emotions):
                self.emo2id[emo] = ndx

    def build_target_vector(self, emotions):
        """
        Build the target vector for the clip graphs.
        The target vector is a one-hot vector of size top-k (or 26 for Emotic) with the emotions present in the clip graph set to 1.

        Args:
            emotions (list): list of emotions present in the clip graph.

        Returns:
            vector (torch.tensor): target vector for the clip graph.
        """
        vector = torch.zeros(self.top_k)
        for emo in emotions:
            if emo in self.emo2id.keys():
                vector[self.emo2id[emo]] = 1
        return vector

    def get_character_emo_map(self, char_emo_pairs):
        """
        Build the target vectors for characters.
        MovieGraphs contains character emotions in the form of pairs (character, emotion).
        A character can have multiple emotions, therefore multiple pairs.
        This function filters multiple emotions of different characters and builds the target vectors.

        Args:
            char_emo_pairs (list): list of character-emotion pairs.

        Returns:
            char_emo_target (dict): dictionary mapping character to target vector.
        """
        char_emo_map = defaultdict(list)
        char_emo_target = dict()
        for character, emotion in char_emo_pairs:
            char_emo_map[character].append(emotion)
        for char, emos in char_emo_map.items():
            target = self.build_target_vector(emos)
            if target.sum():
                char_emo_target[char] = target
        return char_emo_target

    def collect_data(self):
        """
        Collect the movie/scene_id Path objects and scene and character targets.
        """
        for m_id in self.movie_ids:
            for cg in self.cgs[m_id]:
                scenes = [Path(m_id)/vid_name[:-4] for vid_name in cg.video['fname']]
                if m_id == 'tt1568346' and cg.video['fname'][0] == 'scene-079.ss-0893.es-0935.mp4':
                    scenes = [Path(m_id)/'scene-079.ss-0893.es-0935']
                emotions = [pair[1] for pair in utils.get_emotions_from_cg(cg)]
                target = self.build_target_vector(emotions)
                if self.char_level:
                    target = {'scene': target, 'chars': dict()}
                    char_emo_map = self.get_character_emo_map(utils.get_emotions_from_cg(cg))
                    cg_names = set(char_emo_map.keys())
                    ft_scene_chars = [self.new_named_tracks[m_id][str(scene.name)].keys() for scene in scenes if scene.name in self.new_named_tracks[m_id].keys()]
                    ft_names = list()
                    _ = [ft_names.extend(list(names)) for names in ft_scene_chars]
                    ft_names = set(ft_names)
                    intersection = cg_names.intersection(ft_names)
                    if len(intersection) == 0:
                        continue
                    for name in intersection:
                        target['chars'][name] = char_emo_map[name]
                self.scenes.append(scenes)
                self.targets.append(target)

    def get_emo2id_mapping(self):
        """
        Get the emotion to id mapping.

        Returns:
            emo2id (dict): dictionary mapping emotion to id.
        """
        return self.emo2id

    def prepare_mappings(self):
        """
        Prepare the mappings for the dataset.
        """
        self.filter_cgs()
        if not self.emo2id.keys():
            self.map_emo_with_id()
        self.collect_data()

    def __getitem__(self, ndx):
        """
        Get the movie_id/scene Path object and scene and (if requested) character targets for the given index.

        Args:
            ndx (int): index of the scene.

        Returns:
            (tuple): at [0] it contains Path object for movie_id/scene.
                at [1] it contains scene and (if requested) character targets.
        """
        return (self.scenes[ndx], self.targets[ndx])

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.targets)
