from collections import Counter
from omegaconf import OmegaConf
from pathlib import Path
from utils import mg_utils as utils

import matplotlib; matplotlib.use('Agg')

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_emotic_emo_freq(all_mg, emotic_mapping_path):
    """
    Plots the frequency of emotions in the Emotic dataset.
    The plot is saved in plots/emotic_emo_freq.pdf

    Args:
        all_mg (list): List of all the MovieGraph objects.
        emotic_mapping_path (str): Path to the emotic_mapping.json file.
    """
    FILENAME = "emotic_emo_freq.pdf"
    emos = ['loving', 'friendly', 'angry', 'resentful', 'outraged', 'vengeful', 'annoyed', 'annoying', 'frustrated', 'irritated', 'agitated', 'bitter', 'insensitive', 'exasperated',
            'displeased', 'optimistic', 'hopeful', 'imaginative', 'eager', 'disgusted', 'horrified', 'hateful', 'confident', 'proud', 'stubborn', 'defiant', 'independent', 'convincing',
            'disapproving', 'hostile', 'unfriendly', 'mean', 'disrespectful', 'mocking', 'condescending', 'cunning', 'manipulative', 'nasty', 'deceitful', 'conceited', 'sleazy', 'greedy',
            'rebellious', 'petty', 'indifferent', 'bored', 'distracted', 'distant', 'uninterested', 'self-centered', 'lonely', 'cynical', 'restrained', 'unimpressed', 'dismissive', 'worried',
            'nervous', 'tense', 'anxious', 'afraid', 'alarmed', 'suspicious', 'uncomfortable', 'hesitant', 'reluctant', 'insecure', 'stressed', 'unsatisfied', 'solemn', 'submissive', 'confused',
            'skeptical', 'indecisive', 'embarrassed', 'ashamed', 'humiliated', 'curious', 'serious', 'intrigued', 'persistent', 'interested', 'attentive', 'fascinated', 'respectful', 'grateful',
            'excited', 'enthusiastic', 'energetic', 'playful', 'impatient', 'panicky', 'impulsive', 'hasty', 'tired', 'sleepy', 'dizzy', 'scared', 'fearful', 'timid', 'terrified', 'cheerful',
            'delighted', 'happy', 'amused', 'laughing', 'thrilled', 'smiling', 'pleased', 'overwhelmed', 'ecstatic', 'exuberant', 'hurt', 'content', 'relieved', 'relaxed', 'calm', 'quiet',
            'satisfied', 'reserved', 'carefree', 'funny', 'attracted', 'aroused', 'hedonistic', 'pleasant', 'flattered', 'entertaining', 'mesmerized', 'sad', 'melancholy', 'upset',
            'disappointed', 'discouraged', 'grumpy', 'crying', 'regretful', 'grief-stricken', 'depressed', 'heartbroken', 'remorseful', 'hopeless', 'pensive', 'miserable', 'apologetic',
            'nostalgic', 'offended', 'insulted', 'ignorant', 'disturbed', 'abusive', 'offensive', 'surprise', 'surprised', 'shocked', 'amazed', 'startled', 'astonished', 'speechless',
            'disbelieving', 'incredulous', 'kind', 'compassionate', 'supportive', 'sympathetic', 'encouraging', 'thoughtful', 'understanding', 'generous', 'concerned', 'dependable',
            'caring', 'forgiving', 'reassuring', 'gentle', 'jealous', 'determined', 'aggressive', 'desperate', 'focused', 'dedicated', 'diligent']
    all_emos, _ = utils.get_all_emotions(all_mg)
    freq = Counter(all_emos)
    with open(emotic_mapping_path, 'r') as f:                                                                                                                                                 
        mapping = json.load(f)
    rev_map = dict()
    for emot, emos in mapping.items():
        for emo in emos:
            rev_map[emo] = emot
    top_k_emo_freq_pairs = dict()
    for emot, emos in mapping.items():
        top_k_emo_freq_pairs[emot] = 0
        for emo in emos:
            top_k_emo_freq_pairs[emot] += freq[emo]
    top_k_emo_freq_pairs = sorted(top_k_emo_freq_pairs.items(), key=lambda x: -x[1])
    plt.figure(figsize=(20, 10), dpi=300)
    x = [emo[0] for emo in top_k_emo_freq_pairs]
    y = [frq[1] for frq in top_k_emo_freq_pairs]
    plt.bar(range(len(x)), y)
    for i, h in enumerate(y):
        plt.text(i, h, s=h, color="black", horizontalalignment="center", fontsize=20)
    plt.xticks(range(len(x)), x, rotation=90, fontsize=30)
    plt.yticks(fontsize=20)
    plt.ylabel("Frequency of movie scenes", fontsize=30)
    plt.savefig("plots/" + FILENAME, bbox_inches="tight")
    print("Generated plots/" + FILENAME)


def plot_t10_25_emo_freq(all_mg):
    """
    Plot the top-10 and top-25 emotions from MovieGraphs dataset based on frequency.
    The plot is saved in plots/top_10_25_emo_freq.pdf.

    Args:
        all_mg (list): List of MovieGraphs.
    """
    FILENAME = "top_10_25_emo_freq.pdf"
    all_emos, _ = utils.get_all_emotions(all_mg)
    sorted_emo_freq = sorted(Counter(all_emos).items(), key=lambda x: -x[1])
    top_k_emo_freq_pairs_25 = [(emo, freq) for emo, freq in sorted_emo_freq][:25]
    top_k_emo_freq_pairs_10 = [(emo, freq) for emo, freq in sorted_emo_freq][:10]
    plt.figure(figsize=(20, 10), dpi=300)
    x = [emo[0] for emo in top_k_emo_freq_pairs_25]
    y = [frq[1] for frq in top_k_emo_freq_pairs_25]
    plt.bar(range(len(x)), y, label="Top-25")
    x1 = [emo[0] for emo in top_k_emo_freq_pairs_10]
    y1 = [frq[1] for frq in top_k_emo_freq_pairs_10]
    plt.bar(range(len(x1)), y1, label="Top-10")
    for i, h in enumerate(y):
        plt.text(i, h, s=h, color="black", horizontalalignment="center", fontsize=20)
    plt.xticks(range(len(x)), x, rotation=90, fontsize=30)
    plt.yticks(fontsize=20)
    plt.ylabel("Frequency of movie scenes", fontsize=30)
    plt.legend(fontsize=25)
    plt.savefig("plots/" + FILENAME, bbox_inches="tight")
    print("Generated plots/" + FILENAME)


def plot_char_vs_clips_stats(all_mg, MG_ROOT):
    """
    Plot the number of characters vs number of scenes in MovieGraphs dataset.
    The plot is saved in plots/char_freq_MG.pdf.

    Args:
        all_mg (list): List of MovieGraphs.
        MG_ROOT (str): Path to MovieGraphs resources.
    """
    FILENAME = "char_freq_MG.pdf"
    data_split = utils.read_train_val_test_splits(MG_ROOT)
    movie_char_freq_map = dict()
    movie_char_freq, movie_char_counts = list(), list()
    for movie_id in all_mg:
        cgs = all_mg[movie_id].clip_graphs
        characters = list()
        for cg_id in cgs:
            characters += [pair[0].lower() for pair in cgs[cg_id].get_characters()]
        char_freq = Counter(characters)
        char_freq = dict(sorted(char_freq.items(), key=lambda x: -x[1]))
        movie_char_freq_map[movie_id] = char_freq
        movie_char_freq.append(list(char_freq.values()))
        movie_char_counts.append(len(char_freq.keys()))
    movie_char_counts = np.array(movie_char_counts)
    movie_char_counts[::-1].sort()
    ## Plotting average occurances of first top 25 (min) character's occurences
    avg_occurences = np.zeros(min(movie_char_counts)) # Min: 25 | Max: 122
    max_occurences = np.zeros(min(movie_char_counts))
    min_occurences = np.ones(min(movie_char_counts))*9999
    for ndx in range(min(movie_char_counts)):
        freqs = movie_char_freq[ndx][:min(movie_char_counts)]
        avg_occurences += freqs
        min_occurences = np.minimum(min_occurences, freqs)
        max_occurences = np.maximum(max_occurences, freqs)
    avg_occurences /= len(movie_char_freq)
    plt.figure(figsize=(15, 8), dpi=300)
    x = np.arange(min(movie_char_counts))
    plt.bar(x, avg_occurences)
    for i, h in enumerate(avg_occurences):
        text = "{:.1f}".format(avg_occurences[i])
        plt.text(i, h, s=text, color="black", horizontalalignment="center", fontsize=15)
    plt.xticks(x, x+1, fontsize=20)
    plt.yticks(fontsize=22)
    plt.xlabel("Character index", fontsize=25)
    plt.ylabel("Average number of movie scenes \nin which character appears", fontsize=25) # appearance across movie scenes
    plt.savefig("plots/" + FILENAME, bbox_inches="tight")
    print("Generated plots/" + FILENAME)


def plot_emo_vs_clip_stats(all_mg, MG_ROOT):
    """
    Plot the number of emotions vs number of scenes in MovieGraphs dataset.
    Generates two plots, one is saved in plots/emo_freq_MG.pdf and the other
    is saved at plots/emo_freq_MG_from_train_val_test.pdf.

    Args:
        all_mg (list): List of MovieGraphs.
        MG_ROOT (str): Path to MovieGraphs resources.
    """
    data_split = utils.read_train_val_test_splits(MG_ROOT)
    FILENAME_MG = "emo_counts_in_ClipGraphs.pdf"
    FILENAME_SPLIT = "emo_counts_in_ClipGraphs_from_train_val_test.pdf"
    movie_emoCounts_map, emoCounts = dict(), list()
    train_emo_counts, val_emo_counts, test_emo_counts = list(), list(), list()
    for movie_id in all_mg:
        cgs = all_mg[movie_id].clip_graphs
        movie_emoCounts_map[movie_id] = list()
        for cg_id in cgs:
            emotions = utils.get_emotions_from_cg(cgs[cg_id])
            movie_emoCounts_map[movie_id].append(len(emotions))
        emoCounts += movie_emoCounts_map[movie_id]
    for movie_id in data_split["train"]:
        train_emo_counts += movie_emoCounts_map[movie_id]
    for movie_id in data_split["val"]:
        val_emo_counts += movie_emoCounts_map[movie_id]
    for movie_id in data_split["test"]:
        test_emo_counts += movie_emoCounts_map[movie_id]
    emoCounts_freq = Counter(emoCounts)
    emoCounts_freq = dict(sorted(emoCounts_freq.items(), key=lambda x: x[0]))
    tr_emoCounts_fr = Counter(train_emo_counts)
    tr_emoCounts_fr = dict(sorted(tr_emoCounts_fr.items(), key=lambda x: x[0]))
    vl_emoCounts_fr = Counter(val_emo_counts)
    vl_emoCounts_fr = dict(sorted(vl_emoCounts_fr.items(), key=lambda x: x[0]))
    ts_emoCounts_fr = Counter(test_emo_counts)
    ts_emoCounts_fr = dict(sorted(ts_emoCounts_fr.items(), key=lambda x: x[0]))
    ## Plotting trend for entire MG
    x = np.arange(len(emoCounts_freq.keys()))
    y = np.array(list(emoCounts_freq.values()))
    plt.figure(figsize=(15, 8), dpi=300)
    plt.bar(x, y)
    for i, h in enumerate(y):
        plt.text(i, h, s=h, color="black", horizontalalignment="center", fontsize=20)
    plt.xticks(x, list(emoCounts_freq.keys()), fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("Count of emotion labels", fontsize=30)
    plt.ylabel("Number of scenes", fontsize=30)
    # plt.title("Number of emotions present in {} ClipGraphs".format(len(emoCounts)))
    plt.savefig("plots/" + FILENAME_MG)
    print("Generated plots/" + FILENAME_MG)
    ## Plotting trend for train_val_test splits
    x = np.arange(len(emoCounts_freq.keys()))
    y = {"train":list(), "val":list(), "test":list()}
    for k in emoCounts_freq.keys():
        y["train"].append(tr_emoCounts_fr[k] if k in tr_emoCounts_fr else 0)
        y["val"].append(vl_emoCounts_fr[k] if k in vl_emoCounts_fr else 0)
        y["test"].append(ts_emoCounts_fr[k] if k in ts_emoCounts_fr else 0)
    train_y = np.array(y["train"])
    val_y = np.array(y["val"])
    test_y = np.array(y["test"])
    plt.figure(figsize=(15, 8), dpi=300)
    plt.bar(x-0.25, train_y, width=0.25, label="Train: {} Scenes".format(train_y.sum()))
    plt.bar(x, val_y, width=0.25, label="Val: {} Scenes".format(val_y.sum()))
    plt.bar(x+0.25, test_y, width=0.25, label="Test: {} Scenes".format(test_y.sum()))
    plt.legend(fontsize=25)
    plt.xticks(x, list(emoCounts_freq.keys()), fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("Count of emotion labels", fontsize=30)
    plt.ylabel("Number of scenes", fontsize=30)
    # plt.title("Number of emotions in ClipGraphs from Train-Val-Test splits")
    plt.savefig("plots/" + FILENAME_SPLIT, bbox_inches="tight")
    print("Generated plots/" + FILENAME_SPLIT)


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    save_path = Path("./plots/")
    save_path.mkdir(parents=True, exist_ok=True)
    print("Created a new directory: {} to save generated plots".format(save_path))
    config = get_config()
    custom_mapped_emotions = utils.read_custom_mapped_emotions(config['resource_path'])
    all_mg = utils.read_mg_pkl(config['pkl_path'], config['pkl_name'])
    plot_char_vs_clips_stats(all_mg, config["resource_path"])
    plot_emo_vs_clip_stats(all_mg, config["resource_path"])
    plot_t10_25_emo_freq(all_mg)
    plot_emotic_emo_freq(all_mg, config["emotic_mapping_path"])
    print("Done!")
