#!/usr/bin/env python

"""qualitative_analysis.py: Codes for qualitative analysis done for Project
`How you feelin'? Learning Emotion and Mental states in MOvie Charatcers`."""

import torch
import typing
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from scipy.interpolate import interp1d
from omegaconf import OmegaConf
from pathlib import Path


def corrHeatMap(arr: np.ndarray,
                labels: list,
                title: str = None,
                robust: bool = False,
                annot: bool = False,
                rotationx: float = 45,
                rotationy: float = 45,
                save: bool = False,
                save_path: Path = None,
                fontsize: int = 36,
                figsize: tuple = (10, 10)
                ) -> None:
    """
    see the contribution of each emotion on a given emotion = score/sum(row)
    since, the emotion hitting it most may not be expressive, hence
    we need to divide it by max so that the entry can max-value in color scale.
    And most importantly, we need to read it row-wise (no upper triangular and low triangular)
    """
    corr = np.zeros((arr.shape[1], arr.shape[1]))
    for row in arr:
        corr += np.outer(row, row) - np.diag(row**2)
    corr /= corr.sum(axis=1, keepdims=True)
    corr /= corr.max(axis=1, keepdims=True)
    sns.set()
    plt.figure(figsize=figsize, dpi=400)
    ax = sns.heatmap(corr, robust=robust, annot=annot, cmap="Blues",
                     xticklabels=labels, yticklabels=labels, cbar=False)  # fmt=".2f"
    ax.set_yticklabels(ax.get_yticklabels(), rotation=rotationy,
                       horizontalalignment='right', fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotationx,
                       horizontalalignment='center', fontsize=fontsize)
    if save:
        plt.savefig(save_path/(title+".pdf"), bbox_inches='tight')
    plt.show()

# ====================================================================================================

def plot_expressiveness(attn_array: torch.Tensor,
                        mask_array: torch.Tensor,
                        labels: torch.Tensor,
                        labels_lst: torch.Tensor,
                        save: bool,
                        save_path: Path,
                        errorbar: bool,
                        palette: str,
                        n_emotions: int = 10,
                        figsize: typing.Tuple = (10, 10)
                        ) -> None:
    """
    Find out which emotion are significantly more expressive than others.
    --------------------------------------------------------------------------------
    Args:
        - attn_array: torch.Tensor, attention array of shape (num_scenes, num_tokens, num_tokens).
        - mask_array: torch.Tensor, mask array of shape (num_scenes, num_tokens).
        - labels: torch.Tensor, labels of shape (num_scenes, num_emotions).
        - labels_lst: torch.Tensor, list of labels of shape (num_emotions,).
        - save: bool, whether to save the plot or not.
        - errorbar: bool, whether to plot errorbar or not.
        - palette: str, color palette to use for plotting.
        - n_emotions: int, number of emotions.
        - figsize: tuple, figure size.

    Returns:
        - None
    """
    e_arr = []
    express_scores, express_std = [], []
    for scene_row in range(attn_array.shape[0]):
        scene_mask = mask_array[scene_row].logical_not()
        scene_arr = attn_array[scene_row][:n_emotions]*scene_mask
        # scene_arr = (scene_arr.T[scene_mask]).T
        num = scene_arr[:, 2*n_emotions+300:2*(n_emotions+300)].sum(dim=1) +\
            scene_arr[:, 2*(n_emotions+300)+n_emotions:3*(n_emotions+300)].sum(dim=1) +\
            scene_arr[:, 3*(n_emotions+300)+n_emotions:4*(n_emotions+300)].sum(dim=1) +\
            scene_arr[:, 4*(n_emotions+300)+n_emotions:5*(n_emotions+300)].sum(dim=1)
        den = scene_arr[:, n_emotions:300 + n_emotions].sum(dim=1) + scene_arr[:, -300:].sum(dim=1)
        # den[den==0] = np.nan
        assert torch.all((num <= 1).bool()), "num should be less than 1"
        assert torch.all((den <= 1).bool()), "den should be less than 1"
        e = num/den
        e_arr.append(e)
    e_arr = torch.vstack(e_arr)
    for em in range(n_emotions):
        em_label = labels[:, em].bool()
        em_arr = e_arr[:, em][em_label]
        express_scores.append(em_arr.mean())
        # express_std.append(em_arr.std())
    express_scores = torch.Tensor(express_scores)
    sort_idx = torch.argsort(express_scores, descending=True)
    labels_lst = [c for _, c in sorted(zip(express_scores.tolist(), labels_lst), reverse=True)]
    plt.figure(figsize=figsize, dpi=400)
    sns.set(context='notebook', style=None, palette=sns.color_palette(palette, n_emotions),
            font='sans-serif', font_scale=1, color_codes=False, rc=None)
    ax = sns.barplot(x=labels_lst, y=express_scores[sort_idx].tolist())
    if errorbar:
        ax.errorbar(x=labels_lst, y=express_scores[sort_idx].tolist(),
                    yerr=express_std[sort_idx].tolist(), fmt='none', ecolor='black', capsize=3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       horizontalalignment='center')
    ax.set_ylabel('expressiveness')
    # ax.axhline(y=1, color='r', linestyle='--')
    if save:
        if errorbar:
            plt.savefig(save_path/f'expressivenessWerr_{n_emotions}.pdf', bbox_inches='tight')
        else:
            plt.savefig(save_path/f'val_test_expressiveness_{n_emotions}.pdf', bbox_inches='tight')
    plt.show()

# ====================================================================================================

def id2idx(id: int,
           num_cls_tokens: int = 10,
           num_general_tokens: int = 300):
    """
    Convert id to index.
    --------------------------------------------------------------------------------
    Args:
        - id: int, id of the token.
        - num_cls_tokens: int, number of tokens that are `[CLS]` tokens.
        - num_general_tokens: int, number of tokens that are scene/char tokens.

    Returns:
        - idx: int, index of the `[CLS]` token in attention array.
    """
    return (id + 1)*(num_general_tokens + num_cls_tokens)

# Load all the arrays
def load_arrays(base_path: Path = Path(''),
                scene_attn_file: str = 'scene-009_att_new.pt',
                scene_bins_file: str = 'scene-009_scene_bins_new.pt',
                char_bins_file: str = 'scene-009_char_bins_new.pt',
                srt_bins_file: str = 'scene-009_srt_bins_new.pt',
                key_pad_mask_file: str = 'scene-009_keyPadMask.pt',
                num_cls_tokens: int = 10,
                num_general_tokens: int = 300,
                verbose: bool = True,
                cls_ndxs_file: typing.Optional[str] = 'scene-009_cls_ndxs.pt',
               )-> typing.Dict[str, torch.Tensor]:
    
    # Load all the arrays
    attn_arr = torch.load(base_path/scene_attn_file).squeeze()
    scene_bins = torch.load(base_path/scene_bins_file).squeeze()
    char_bins = torch.load(base_path/char_bins_file).squeeze()
    srt_bins = torch.load(base_path/srt_bins_file).squeeze()
    key_pad_mask = torch.load(base_path/key_pad_mask_file).squeeze()
    inv_key_pad_mask = key_pad_mask.logical_not().int().squeeze()
    attn_arr = attn_arr*inv_key_pad_mask
    # Ensuring irrelevant tokens are not considered
    scene_bins[inv_key_pad_mask[num_cls_tokens:
                                num_cls_tokens+num_general_tokens].logical_not()] = 0
    
    # Verbosity
    if verbose:
        print(f"scene attn shape: {attn_arr.shape}")
        print(f"scene bins shape: {scene_bins.shape}")
        print(f"scene char_bins shape: {char_bins.shape}")
        print(f"scene srt_bins shape: {srt_bins.shape}")
        print(f"scene key_pad_mask shape: {key_pad_mask.shape}")
        print(f"inv_key_pad_mask shape: {inv_key_pad_mask.shape}")
        cls_ndxs = torch.load(base_path/cls_ndxs_file).squeeze()
        print(f"cls_ndxs.shape: {cls_ndxs.shape}")
        print(f"cls_ndxs: {cls_ndxs}")
    min_bin_idx, max_bin_idx = min(scene_bins), max(scene_bins)
    # Here 1/3 accounts for 0.33sec for each bin. Meaning 1 bin = 0.33sec, or 3 bins = 1sec
    time_axis = np.arange(min_bin_idx+1, max_bin_idx+2)*(1/3)
    ref_time = torch.from_numpy(np.arange(min_bin_idx, max_bin_idx+1))
    return {'attn_arr': attn_arr, 'scene_bins': scene_bins,
            'char_bins': char_bins, 'srt_bins': srt_bins,
            'key_pad_mask': key_pad_mask, 'inv_key_pad_mask': inv_key_pad_mask,
            'cls_ndxs': cls_ndxs, 'min_bin_idx': min_bin_idx,
            'max_bin_idx': max_bin_idx, 'time_axis': time_axis, 'ref_time': ref_time}

def align_scores_according_to_time(bin_indices, attn_scores, ref_time):
    """
    Align attention of `[CLS]` on scene/char tokens w.r.t time = `ref_time`.
    --------------------------------------------------------------------------------
    Args:
        - bin_indices: torch.Tensor of shape (num_cls_tokens,) that tells
          in which time-bin each token is.
        - attn_scores: torch.Tensor of shape (num_general_tokens,) that tells
          attention scores for each token.
        - ref_time: torch.Tensor of shape (max_bin_idx-min_bin_idx,) = timea-axis
          in terms of bin indices.          
    
    Returns:
        - time_dict: dict of shape (max_bin_idx-min_bin_idx,) = at each time-step
          what is the attention score for each `[CLS]` on scene/char tokens.
    """
    time_dict = {int(k): 0 for k in ref_time}
    for u, v in zip(bin_indices, attn_scores):
        if u in ref_time:
            if int(u) in time_dict:
                time_dict[int(u)] = max(float(v), time_dict[int(u)])
            else:
                time_dict[int(u)] = float(v)
    return list(time_dict.values())


def interpolate_scene(scene_bin_indices, scene_attn_scores):
    """
    Interpolate attention scores for scene tokens across time.
    --------------------------------------------------------------------------------
    Args:
        - scene_bin_indices: torch.Tensor of shape (num_general_tokens,) that tells
            in which time-bin each scene token is.
        - scene_attn_scores: torch.Tensor of shape (num_general_tokens,) that tells
            how much `[CLS]` attends to each scene token.

    Returns:
        - f: function that takes time as input and returns attention score.
    """
    xx = scene_bin_indices[scene_bin_indices != 0]*(1/3)
    f = interp1d(xx.detach().cpu().numpy(),
                 scene_attn_scores[(scene_bin_indices != 0).squeeze()].detach().cpu().numpy(),
                 fill_value='extrapolate')
    return f

def preprocess(char_emo_idx: typing.Dict[str, typing.List[typing.Tuple[str, int]]],
               array_dict: typing.Dict[str, torch.Tensor],
               max_num_characters: int = 2,
               num_cls_tokens: int = 10,
               num_general_tokens: int = 300
              )-> typing.DefaultDict[str,
                  typing.Dict[str, typing.List[typing.List[float]]]]:
    """
    Preprocess and align attention scores for each character and scene so as to
    prepare it for plotting.
    -----------------------------------------------------------------------------------
    Args:
        - char_emo_idx: dict of shape (num_characters,) that tells
            position in attention matrix corresponding to emotion of a character.
        - array_dict: dict consisting of all relevant info.
        - max_num_characters: int that tells maximum number of characters to consider.
        - num_cls_tokens: int that tells number of `[CLS]` tokens.
        - num_general_tokens: int that tells number of general(char/scene) tokens.

    Returns:
        - emo_vals: defaultdict(dict) of shape (num_characters,) that tells
            attention scores for each character.
    """
    emo_vals = defaultdict(dict)
    num_characters = len(char_emo_idx)
    assert num_characters <= max_num_characters, "Number of characters exceeds max_num_characters."
    for key, val in char_emo_idx.items():
        for v in val:
            emo_vals[key][v[0]] = []
            emo_vals[key][v[0]].append(interpolate_scene(array_dict['scene_bins'],
                                        array_dict['attn_arr'][v[1]][num_cls_tokens:id2idx(0)]))
            for i in range(num_characters):
                emo_vals[key][v[0]].append(align_scores_according_to_time(array_dict['char_bins'][i],
                                    array_dict['attn_arr'][v[1]][id2idx(i)+num_cls_tokens:
                                    id2idx(i)+num_cls_tokens+num_general_tokens],
                                    array_dict['ref_time']))
            emo_vals[key][v[0]].append(align_scores_according_to_time(array_dict['srt_bins'],
                                array_dict['attn_arr'][v[1]][id2idx(max_num_characters):],
                                array_dict['ref_time']))
    return emo_vals

# Plot Attention scores for each character with their respective emotion
def plotAttention(emotion: str,
                  time_axis: np.ndarray,
                  cls_dict_vals: typing.List[typing.List[float]],
                  interpolation_func: typing.Callable,
                  char_names: typing.List[str],
                  main_char_name: str,
                  fig_style: str = 'woa',
                  save_fig: bool = False,
                  save_path: str = ''
                 )->None:
    """
    This function plot how [CLS] token attend to various emotion for
    each character throughout the entire scene of a movie.
    --------------------------------------------------------------------------------
    Args:
        - emotions: str, emotion to plot attention for.
        - time_axis: np.ndarray, time-axis for the scene.
        - cls_dict_vals: List[List[float]], attention scores for each character
            for each time-step.
        - interpolation_func: function, interpolation function for scene tokens.
        - char_names: names of characters in the scene.
        - main_char_name: str, name of the main character of the scene.
        - fig_style: style of figure to plot. Default is 'woa'.
        - save_fig: whether to save figure or not. Default is False.
        - save_path: path to save figure. Default is ''.

    Returns:
        - None
    """
    plt.figure(figsize=(15, 1.5), dpi = 400)
    emotion = emotion.upper()
    plt.plot(time_axis, interpolation_func(time_axis),
            label=f'{main_char_name} {emotion} CLS on Video tokens')
    for i, val in enumerate(cls_dict_vals[:-1]):
        plt.plot(time_axis, val, label=f'{main_char_name} {emotion} CLS on {char_names[i]} tokens')
    plt.plot(time_axis, cls_dict_vals[-1], label=f'{main_char_name} {emotion} CLS on Dialog SRT tokens')
    if fig_style == 'wa':
        plt.legend() # loc='upper left', bbox_to_anchor=(1, 1)
        plt.xticks(np.arange(0, int(max(time_axis))+1, 2))
        plt.xlabel('Time (s)')
        plt.ylabel('Attention')
    elif fig_style == 'woa':
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    else:
        raise ValueError(f"Invalid fig_style: {fig_style}. Valid options are 'wa' and 'woa'.")
    plt.xlim(0, 50)
    if save_fig:
        plt.savefig(f'{save_path}/{main_char_name}CLS_{emotion}.svg', bbox_inches='tight')
    plt.show()

def compile(char_emo_idx: typing.Dict[str, typing.List[typing.Tuple[str, int]]],
            array_dict: typing.Dict[str, np.ndarray],
            max_num_characters: int = 10,
            num_cls_tokens: int = 1,
            num_general_tokens: int = 1,
            save_path: str = ''
           )->None:
    char_vals = preprocess(char_emo_idx, array_dict, max_num_characters=max_num_characters,
                           num_cls_tokens=num_cls_tokens, num_general_tokens=num_general_tokens)

    for (k, v) in char_vals.items():
        for key, val in v.items():
            plotAttention(key, array_dict['time_axis'],
                          val[1:], val[0], list(char_emo_idx.keys()), k,
                          fig_style='woa', save_fig=True, save_path=save_path)


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == '__main__':

    config = get_config()
    labels10 = {'Happy': 0, 'Worried': 1, 'Calm': 2, 'Excited': 3, 'Quiet': 4, 'Amused': 5,
                'Curious': 6, 'Confused': 7, 'Serious': 8, 'Surprise': 9}
    labels25 = {**labels10, **{"Friendly": 10, "Angry": 11, "Annoyed": 12, "Shocked": 13,
                               "Confident": 14, "Sad": 15, "Nervous": 16, "Scared": 17,
                               "Cheerful": 18, "Upset": 19, "Polite": 20, "Honest": 21,
                               "Helpful": 22, "Determined": 23, "Alarmed": 24}}
    BASE_PATH = Path(config["saved_model_path"])/"metadata"
    SAVE_PATH = Path("./plots/")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------------------------------------------------
    print("\nGenerating temporally varying attention plots for movie scenes...")
    num_cls_tokens = 10
    num_general_tokens = 300
    max_num_characters = 4

    # Forrest Gump is first in the list with Happy emotion = 310,
    # while Mom is second with worried token = 621...
    char_emo_idx = {"forrest_gump":[('Happy', id2idx(0)+labels25['Happy'])],
                    "mom":[('Worried', id2idx(1)+labels25['Worried'])]}

    # Load attention scores for scene tokens
    array_dict = load_arrays(base_path=BASE_PATH,
                             scene_attn_file="forrest_gump_scene009_cont/scene-009_att_new.pt",
                             scene_bins_file="forrest_gump_scene009_cont/scene-009_scene_bins_new.pt",
                             char_bins_file="forrest_gump_scene009_cont/scene-009_char_bins_new.pt",
                             srt_bins_file="forrest_gump_scene009_cont/scene-009_srt_bins_new.pt",
                             key_pad_mask_file="forrest_gump_scene009_cont/scene-009_keyPadMask.pt",
                             cls_ndxs_file="forrest_gump_scene009_cont/scene-009_cls_ndxs.pt")
    compile(char_emo_idx, array_dict, max_num_characters=max_num_characters,
            num_cls_tokens=num_cls_tokens, num_general_tokens=num_general_tokens,
            save_path=SAVE_PATH)

    # Dylan is first in the list with Excited emotion = 313,
    # while Jamie is second with Happy token = 620...
    char_emo_idx1 = {"dylan": [('Excited', id2idx(0)+labels25['Excited'])],
                     "jamie": [('Happy', id2idx(1)+labels25['Happy'])]}
    
    # Load attention scores for scene tokens
    array_dict1 = load_arrays(base_path=BASE_PATH,
                              scene_attn_file="fwb_scene125_cont/scene-125_att_new.pt",
                              scene_bins_file="fwb_scene125_cont/scene-125_scene_bins_new.pt",
                              char_bins_file="fwb_scene125_cont/scene-125_char_bins_new.pt",
                              srt_bins_file="fwb_scene125_cont/scene-125_srt_bins_new.pt",
                              key_pad_mask_file="fwb_scene125_cont/scene-125_keyPadMask.pt",
                              cls_ndxs_file="fwb_scene125_cont/scene-125_cls_ndxs.pt")
    compile(char_emo_idx1, array_dict1, max_num_characters=max_num_characters,
            num_cls_tokens=num_cls_tokens, num_general_tokens=num_general_tokens,
            save_path=SAVE_PATH)
    
    # -------------------------------------------------------------------------------------------
    
    print("\nGenerating expressiveness score for Top 25 Emotions...")
    l2_25_attn = torch.load(BASE_PATH/"test_val_t25_att.pt")
    l2_25_mask = torch.load(BASE_PATH/"test_val_t25_mask.pt")
    l2_25_labels = torch.load(BASE_PATH/"test_val_sceneTgt_t25.pt")
    print(f"Shape of attention matrix for Top 25 EMotion: {l2_25_attn.shape}")
    print(f"Shape of mask matrix for Top 25 Emotion: {l2_25_mask.shape}")
    print(f"Shape of labels for Top 25 Emotion: {l2_25_labels.shape}")
    plot_expressiveness(attn_array=l2_25_attn,
                        mask_array=l2_25_mask,
                        labels=l2_25_labels,
                        labels_lst=labels25,
                        save=True,
                        save_path=SAVE_PATH,
                        errorbar=False,
                        palette='husl',
                        n_emotions=25,
                        figsize=(10, 4))
    
    # -------------------------------------------------------------------------------------------
    print("\nGenerating correlation heatmaps for Top 10 Emotions...")
    ch10_train = torch.load(BASE_PATH/"t10_train_char_targets.pt")
    ch10_val = torch.load(BASE_PATH/"t10_val_char_targets.pt")
    sc10_train = torch.load(BASE_PATH/"t10_train_scene_targets.pt")
    sc10_val = torch.load(BASE_PATH/"t10_val_scene_targets.pt")
    print("10 character training set shape:", ch10_train.shape)
    print("10 character validation set shape:", ch10_val.shape)
    print("10 character training scene shape:", sc10_train.shape)
    print("10 character validation scene shape:", sc10_val.shape)
    ch10_arr = np.concatenate((ch10_train, ch10_val), axis=0)
    sc10_arr = np.concatenate((sc10_train, sc10_val), axis=0)
    _ = corrHeatMap(sc10_arr, labels10, title="scene_10", save=True,
                    save_path=SAVE_PATH, rotationx=90, rotationy=0, fontsize=36, figsize=(9, 9))
    _ = corrHeatMap(ch10_arr, labels10, title="characters_10", save=True,
                    save_path=SAVE_PATH, rotationx=90, rotationy=0, fontsize=36, figsize=(9, 9))
    
    print("\nGenerating correlation heatmaps for Top 25 Emotions...")
    ch25_train = torch.load(BASE_PATH/"t25_train_char_targets.pt")
    ch25_val = torch.load(BASE_PATH/"t25_val_char_targets.pt")
    sc25_train = torch.load(BASE_PATH/"t25_train_scene_targets.pt")
    sc25_val = torch.load(BASE_PATH/"t25_val_scene_targets.pt")
    print("25 character training set shape:", ch25_train.shape)
    print("25 character validation set shape:", ch25_val.shape)
    print("25 character training scene shape:", sc25_train.shape)
    print("25 character validation scene shape:", sc25_val.shape)
    ch25_arr = np.concatenate((ch25_train, ch25_val), axis=0)
    sc25_arr = np.concatenate((sc25_train, sc25_val), axis=0)
    _ = corrHeatMap(sc25_arr, labels25, title="scene_25", save_path=SAVE_PATH,
                    save=True, rotationx=90, rotationy=0, fontsize=32, figsize=(15, 15))
    _ = corrHeatMap(ch25_arr, labels25, title="characters_25", save_path=SAVE_PATH,
                    save=True, rotationx=90, rotationy=0, fontsize=32, figsize=(15, 15))
