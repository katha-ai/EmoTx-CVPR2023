## Import statements
from collections import Counter, defaultdict
from matplotlib import colors
from scipy import stats

import matplotlib
matplotlib.use('Agg')

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import sys
import yaml


def get_default_vars():
    """
    This loads a dictionary names mg_var which contains different
    PATHs to access MovieGraph Dataset
    """
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config["mg_vars"]


def import_graphClass(ROOT):
    """
    Imports the GraphClass script provided with the data.
    """
    sys.path.insert(1, ROOT)
    import GraphClasses


def read_mg_pkl(DATA_ROOT, PKL_NAME):
    """
    Loads the pickle file that contains the entire MovieGraph dataset.
    """
    import_graphClass(DATA_ROOT)
    all_mg = None
    with open(DATA_ROOT + PKL_NAME, 'rb') as fid:
        all_mg = pickle.load(fid, encoding="latin1")
    return all_mg


def read_custom_mapped_emotions(RESOURCE_PATH):
    """
    Reads a json file names custom_mapped_emotions.json which maps
    560 different emotions to 7 basic classes:
    '', 'disgusted', 'scared', 'sad', 'surprised', 'angry', 'happy'
    """
    with open(RESOURCE_PATH + 'custom_mapped_emotions.json', 'r') as f:
        custom_mapped_emotions = json.load(f)
    return custom_mapped_emotions


def read_emotion_att_list(RESOURCE_PATH):
    """
    Loads emotoin_attributs_list.txt file which contains
    all the emotions present in the MovieGraph dataset.
    """
    with open(RESOURCE_PATH + 'emotion_attributes_list.txt', 'r') as f:
        attrs = [line.strip() for line in f.readlines()]
    return attrs


def read_train_val_test_splits(PATH):
    """
    Reads the slipt.json file which contains the distribution of movies
    across train-val-test sets and returns that as a dictionary.
    Count of movies in respective splits-
    Train: 34
    Val: 7
    Test: 10
    """
    with open(PATH + 'split.json', 'r') as f:
        data_split = json.load(f)
    return data_split


def get_clip_graphs(movie_graphs, m_ids):
    """
    Returns the COMBINED  clip_graphs from multiple movies who's
    movie-ID is provided in m_id attribute.
    """
    clip_graphs = dict()
    for m_id in m_ids:
        clip_graphs[m_id] = list()
        mg = movie_graphs[m_id]
        for cg_id in mg.clip_graphs:
            clip_graphs[m_id].append(mg.clip_graphs[cg_id])
    return clip_graphs


def get_cgs_for_data_split(all_mg, MG_ROOT):
    """
    Returns a dictionary for train-val-test splits which maps these splits
    with a list of all COMBINED clip_graphs in that set. Irrespective of movies.
    """
    data_split = read_train_val_test_splits(MG_ROOT)
    train_cgs, val_cgs, test_cgs = list(), list(), list()
    grouped_train_cgs = get_clip_graphs(all_mg, data_split['train'])
    grouped_val_cgs = get_clip_graphs(all_mg, data_split['val'])
    grouped_test_cgs = get_clip_graphs(all_mg, data_split['test'])
    for cgs in grouped_train_cgs.values():
        train_cgs += cgs
    for cgs in grouped_val_cgs.values():
        val_cgs += cgs
    for cgs in grouped_test_cgs.values():
        test_cgs += cgs
    cgs = {'train': train_cgs,
           'val': val_cgs,
           'test': test_cgs}
    return cgs


def get_emotions_from_cg(clip_graph):
    """
    Returns all the emotions from a clip grpah object
    """
    emo_attributes = clip_graph.find_all_entity_attribute_pairs(subtypes=['emo'],
                                                                return_names=True)
    return emo_attributes


def get_emo_filtered_cgs(cgs, list_of_emo_to_filter=None):
    """
    IF list_of_emo_to_filter attribute is None,
        This method filters the clip_graphs that do not have any
        emotion_attribute in them.
    ELSE
        This method filters the clip_graphs that do not have emotions
        intersecting with the given list of emotions (also filters empty ones)
    """
    filtered_cgs = list()
    non_empty_count, empty_count = 0, 0
    for cg in cgs:
        emos = get_emotions_from_cg(cg)
        if emos:
            emos = [pair[1] for pair in emos]
            if list_of_emo_to_filter:
                check_passed = False
                for emo in emos:
                    if emo in list_of_emo_to_filter:
                        check_passed = True
                        break
                if not check_passed:
                    empty_count += 1
                    continue
            non_empty_count += 1
            filtered_cgs.append(cg)
        else:
            empty_count += 1
    return filtered_cgs, non_empty_count, empty_count


def get_all_emotions(all_mg):
    """
    This method returns a list of all the unique emotions present in the
    MovieGraph dataset by iterating over all the ClipGraphs from all the
    MovieGraph objects.
    - All mg and cgs are stored as OrderedDict, therefore iterating over
    them will return the id which can be used to get the mg/cg.
    """
    all_paired_emos = list()
    for mg_id in all_mg:
        mg = all_mg[mg_id]
        cgs = mg.clip_graphs
        for cg_id in cgs:
            all_paired_emos += get_emotions_from_cg(cgs[cg_id])
    all_emos = [pair[1] for pair in all_paired_emos]
    unique_emotions = list(set(all_emos))
    return all_emos, unique_emotions


def get_top_k_emotions(all_mg, top_k=50):
    """
    Returns the top_k emotions from entire MovieGraph dataset.
    """
    all_emos, _ = get_all_emotions(all_mg)
    sorted_emo_freq = sorted(Counter(all_emos).items(), key=lambda x: -x[1])
    top_k_emo = [emo for emo, _ in sorted_emo_freq][:top_k]
    return top_k_emo


def get_emotions_from_multiple_cgs(clip_graphs):
    """
    Returns combined list of emotions extracted from multiple clip_graphs
    """
    paired_emotions = list()
    for clip_graph in clip_graphs:
        paired_emotions += get_emotions_from_cg(clip_graph)
    emotions = [pair[1] for pair in paired_emotions]
    return emotions


def get_emotic_emotions_map(MAPPING_FILE_PATH):
    """
    Returns a dictionary with new grouped Emotic-emotion labels mapped to individual
    emotions in MovieGraph dataset.
    """
    with open(MAPPING_FILE_PATH, 'r') as f:
        emotic_map = json.load(f)
    return emotic_map


def get_custom_mapped_emos_from_cgs(cgs, RESOURCE_PATH):
    """
    Returns the emotions mapped according to the 7 classes in the
    custom_mapped_emotions.
    """
    custom_mapped_emotions = read_custom_mapped_emotions(RESOURCE_PATH)
    custom_mapped_cg_emos = list()
    for cg in cgs:
        character_emo_pairs = get_emotions_from_cg(cg)
        for char, emo in character_emo_pairs:
            try:
                basic_emotion = custom_mapped_emotions[emo]
                if basic_emotion:
                    custom_mapped_cg_emos.append(basic_emotion)
                else:
                    raise KeyError
            except KeyError:
                custom_mapped_cg_emos.append('UNK')
    return custom_mapped_cg_emos


def get_char_scene_emotions(all_mg, imdb_key, custom_mapped_emotions):
    """
    - Returns a dictionary of dictionary where primary keys are integer indexes
      corresponding to the scene numbers.
    - Each scene number maps to a dictionary which maps characters to their
      emotions in that scene.
    - These emotions are filtered using the custom_mapped_emotions.
    """
    char_scene_emotions = defaultdict(lambda: defaultdict(list))
    mg = all_mg[imdb_key]
    for sid in mg.clip_graphs:
        cg = mg.clip_graphs[sid]
        emo_attrs = get_emotions_from_cg(cg)
        for (character, emo) in emo_attrs:
            try:
                basic_emotion = custom_mapped_emotions[emo]
                if basic_emotion:
                    char_scene_emotions[sid][character].append(basic_emotion)
            except KeyError:
                pass    
    return char_scene_emotions


def get_char_emos(char_scene_emotions, char_names):
    """
    Filters specific characters from the entire movie and retuns a
    dictionary with character_names mapped to their emotions arranged
    according to scene numbers.
    """
    label_id_mapping = {
        'angry': 0,
        'disgusted': 1,
        'happy': 2,
        'sad': 3,
        'scared': 4,
        'surprised': 5,
        'UNK': 6
    }
    char_emo_dict = {}
    for name in char_names:
        char_vals = []
        for sid in sorted(char_scene_emotions.keys()):
            char_emos = char_scene_emotions[sid][name]
            label_id = [label_id_mapping[emo] for emo in char_emos]
            if not label_id:
                char_vals.append(label_id_mapping['UNK'])
            else:
                emo_mode = stats.mode(label_id)[0][0]
                char_vals.append(emo_mode)
        char_emo_dict[name] = char_vals
    return char_emo_dict
