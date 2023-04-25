from models.mlp import mlp_char, mlp_scene
from models.tx_encoder import tx_char, tx_scene
from models.emotx_1cls import Emotx_1CLS
from models.emotx import EmoTx
from omegaconf import OmegaConf
from pathlib import Path

import matplotlib; matplotlib.use("Agg")

import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import yaml


def plot_tx_AP(config, base_path, filenames, plot_name, emo_count, figsize=(20,5), fontsize=13, xlbl_rotation=45, model_names=list(), emotic=False):
    """
    Plots the class-wise AP scores for the given models.

    Args:
        config (dict): The config file with all the hyperparameters.
        base_path (str): The path to the directory containing the pickle files.
        filenames (list): The list of pickle file names containing the metadata.
        plot_name (str): The name of the plot to be saved.
        emo_count (int): The number of emotions.
        figsize (tuple): The figure size.
        fontsize (int): The fontsize for text within the plot.
        xlbl_rotation (int): The rotation for the x-axis labels.
        model_names (list): The list of model names to be displayed in the legend.
        emotic (bool): Whether the Emotic dataset is used.
    """
    emo2ids, eval_metrics = list(), list()
    for filename in filenames:
        with open(base_path/filename, 'rb') as f:
            emo2ids.append(pickle.load(f))
            _ = pickle.load(f)
            eval_metrics.append(pickle.load(f))
    scene_mAP_scores = list()
    for ndx in range(len(filenames)):
        metrics = eval_metrics[ndx]['AP']
        scene_mAP_scores.append(metrics[1])
    dct = {model_names[ndx]: scene_mAP_scores[ndx] for ndx in range(len(filenames))}
    labels = list(emo2ids[0].keys())
    if emotic:
        with open(config["emotic_mapping_path"], 'r') as f:
            mapping = json.load(f)
        labels = list(mapping.keys())
    df = pd.DataFrame(dct, index=labels)
    df = df.sort_values(model_names[0], ascending=False)
    legend_labels = list()
    for model_name in model_names:
        legend_labels.append(model_name)
    plt.rcParams["figure.dpi"] = 400
    plot = df.plot(kind='bar', figsize=figsize, width=0.8)
    plot.legend(legend_labels, fontsize=fontsize)
    plot.set_ylabel("AP Scores", fontsize=fontsize+5)
    plot.set_xticklabels(plot.get_xticklabels(), fontsize=fontsize, rotation=xlbl_rotation)
    plot.xaxis.labelpad = 5
    plt.yticks(fontsize=fontsize)
    # plt.xticks(np.arange(len(scene_mAP_scores[0])), plot.get_xticklabels(), rotation=xlbl_rotation)
    patches = list(plot.patches)
    if emotic:
        patches = list(plot.patches)
        for i in range(0, len(scene_mAP_scores[0])):
            plot.annotate("{:.1f}".format(patches[i].get_height()*100), (patches[i].get_x(), patches[i].get_height()*1.005), fontsize=15)
    else:
        for i in range(0, len(scene_mAP_scores[0])):
            heights = [patches[i+(emo_count*bar_count)].get_height() for bar_count in range(len(filenames))]
            max_height = max(heights)
            base = heights.index(max_height)
            plot.annotate("{:.1f}".format(max_height*100), (patches[i+(base*emo_count)].get_x(), max_height*1.005), fontsize=fontsize)
    plt.savefig("./plots/" + plot_name, bbox_inches="tight")
    plt.cla()
    print("Saved {}".format(plot_name))


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    save_path = Path("./plots")
    save_path.mkdir(parents=True, exist_ok=True)
    config = get_config()
    base_path = Path(config["saved_model_path"])/"metadata"
    model_names = ["Our model with individual CLS",
                   "Our model with 1 CLS",
                   "Single Tx Encoder",
                   "MLP (2 Lin)"]
    filenames_t10_camReady = [
        "EmoTx_t10.pkl",
        "emotx1cls_t10.pkl",
        "tx_encoder_scene_t10.pkl",
        "mlp_scene_t10.pkl",
    ]
    filenames_t25_camReady = [
        "EmoTx_t25.pkl",
        "emotx1cls_t25.pkl",
        "tx_encoder_scene_t25.pkl",
        "mlp_scene_t25.pkl",
    ]
    filenames_emotic_camReady = ["EmoTx_Emotic26.pkl"]
    plot_name_t10 = "t10_APs_sorted.pdf"
    plot_name_t25 = "t25_APs_sorted.pdf"
    plot_name_emotic = "emotic_APs_sorted.pdf"
    figsize_t10 = (20, 7)
    figsize_t25 = (20, 5)
    figsize_emotic = (20, 7)
    xlbl_rotation_t10 = 45
    xlbl_rotation_t25 = 45
    xlbl_rotation_emotic = 90
    fontsize_t10 = 20
    fontsize_t25 = 15
    fontsize_emotic = 20
    plot_tx_AP(config, base_path, filenames_t10_camReady, plot_name_t10, 10, figsize_t10, fontsize_t10, xlbl_rotation_t10, model_names)
    plot_tx_AP(config, base_path, filenames_t25_camReady, plot_name_t25, 25, figsize_t25, fontsize_t25, xlbl_rotation_t25, model_names)
    plot_tx_AP(config, base_path, filenames_emotic_camReady, plot_name_emotic, 26, figsize_emotic, fontsize_emotic, xlbl_rotation_emotic, [model_names[0]], emotic=True)
