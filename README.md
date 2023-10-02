<div align="center">
    <h1> :movie_camera: <a href="https://arxiv.org/abs/2304.05634">How you feelin'? Learning Emotions and Mental States in Movie Scenes</a><br>
    <a href="https://img.shields.io/badge/python-3.6-blue"><img src="https://img.shields.io/badge/python-3.6-blue"></a>
    <a href="https://img.shields.io/badge/made_with-pytorch-red"><img src="https://img.shields.io/badge/made_with-pytorch-red"></a>
    <a href="https://img.shields.io/badge/dataset-MovieGraphs-orange"><img src="https://img.shields.io/badge/dataset-MovieGraphs-orange"></a>
    <a href="https://katha-ai.github.io/projects/emotx"><img src="https://img.shields.io/website?up_message=up&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Femotx&link=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Femotx"></a>
    <a href="https://arxiv.org/abs/2304.05634"><img src="https://img.shields.io/badge/arXiv-2304.05634-f9f107.svg"></a>
    <a href="https://youtu.be/7OATfNOdAzI"><img src="https://badges.aleen42.com/src/youtube.svg"></a>
    <a href="https://katha-ai.github.io/projects/emotx"><img src="docs/assets/emotx.gif" style="width: 800px;"></a>
</div>

## :bookmark: Contents
1. [About](#robot-about)
2. [Setting up the repository](#toolbox-setting-up-the-repository)
    1. [Create a virtual environment](#earth_asia-create-a-python-virtual-environment)
    2. [Setup the data directory](#stars-download-the-moviegraphs-features)
    3. [Update the config template](#book-create-the-configyaml)
3. [Feature Extraction](#bomb-feature-extraction)
4. [Train EmoTx with different configurations!](#weight_lifting-train)
5. [Download](#mag-download)
    1. [EmoTx pre-trained weights](#rocket-emotx-pre-trained-weights)
    2. [Pre-trained feature backbones](#open_hands-pre-trained-feature-backbones)
6. [Bibtex](#round_pushpin-cite)

## :robot: About
This is the official code repository for CVPR-2023 accepted paper ["How you feelin'? Learning Emotions and Mental States in Movie Scenes"](https://arxiv.org/abs/2304.05634). This repository contains the implementation of EmoTx, a Transformer-based model designed to predict emotions and mental states at both the scene and character levels. Our model leverages multiple modalities, including visual, facial, and language features, to capture a comprehensive understanding of emotions in complex movie environments. Additionally, we provide the pre-trained weights for EmoTx and all the pre-trained feature backbones used in this project. We also provide the extracted features for scene (full frame), character faces and subtitle from MovieGraphs dataset.
<br>

## :toolbox: Setting up the repository
### :earth_asia: Create a python-virtual environment
1. Clone the repository and change the working directory to be project's root.
```
$ git clone https://github.com/katha-ai/EmoTx-CVPR2023.git
$ cd EmoTx-CVPR2023
```
2. This project strictly requires `python==3.6`.

Create a virtual environment using Conda-
```
$ conda create -n emotx python=3.6
$ conda activate emotx
(emotx) $ pip install -r requirements.txt
```
OR

Create a virtual environment using pip (make sure you have Python3.6 installed)
```
$ python3.6 -m pip install virtualenv
$ python3.6 -m virtualenv emotx
$ source emotx/bin/activate
(emotx) $ pip install -r requirements.txt
```

### :stars: Download the MovieGraphs features
You can also use `wget` to download these files-
```
$ wget -O <FILENAME> <LINK>
```

|File name | Contents | Comments |
|----------|---------------|----------|
| [EmoTx_min_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EdbbcQvEaBlIg6Sktxw60lQBiOUyDWdbKf3GhF88mrEhaA?download=1) | <ul><li>Extended character tracks</li><li>`emotic_mapping.json`</li><li>MovieGraphs pickle</li><li>Scene (full frame) features extracted from MViT_v1 model pre-trained on _Kinetics400 dataset</li><li>Character face features extracted from ResNet50 pre-trained on VGGFace, FER13 and SFEW datasets</li><li>Subtitle features (from both pre-trained and fine-tuned RoBERTa)</li><li>All pre-trained backbones used in EmoTx</li></ul> | contains `data/` directory which will occupy 167GB of disk space. |
| [InceptionResNetV1_VGGface_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaTYYi2G2DNCpUaO7-Hl9PgBaXgCedq3QMxbuVos3Sfa7A?download=1) | Character face features extracted from InceptionResNet_v1 model pre-trained on VGGface2 dataset. | Contains `generic_face_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_face_features/` will occupy 32GB of disk space. |
| [VGG-vm_FER13_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/Ecx72Es1AdBGnT1Zr0U2R0cBjsx5WXP1nNAHjW2-3CdtbA?download=1) | Character face features  extracted from VGG-vm model pretrained on VGGFace and FER13 datasets | Contains `emo_face_features/` directory. TO use these features with EmoTx, move this directory inside `data/` extracted with `EmoTx_min_feats.tar.gz`. After extraction, `emo_face_features/` will occupy 254GB of disk sace.|
| [ResNet152_ImgNet_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaPRbRO5hmVFtOXgkb-mLCkBX11y2T4dzwdXXSLbX0eAtw?download=1) | Scene (full frame) features extracted from ResNet152 model pre-trained on ImageNet dataset | Contains `generic_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_scene_features/` will occupy 72GB of disk space.  |
| [ResNet50_PL365_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaSkvXnjw6tBsXsRYf41yTgBbafy63-Nen_MRXulx0ycQA?download=1) | Scene (full frame) features extracted from ResNet50 model pre-trained on Places365 dataset. | Contains `resnet50_places_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `resnet50_places_scene_features/` will occupy 143GB of disk space. |
<br>

### :book: Create the `config.yaml`
1. Create a copy of the given config template
```
(emotx) $ cp config_base.yaml config.yaml
```
2. Edit the lines `2-9` in config as directed in the comments. If you have extracted the `EmoTx_min_feats.tar.gz` in `/home/user/data`, then the path variables in `config.yaml` would be-
```
# Path variables
data_path: /home/user/data
resource_path: /home/user/data/MovieGraph/resources/
clip_srts_path: /home/user/data/MovieGraph/srt/clip_srt/
emotic_mapping_path: /home/user/data/emotic_mapping.json
pkl_path: /home/user/data/MovieGraph/mg/py3loader/
save_path: /home/user/checkpoints/
saved_model_path: /home/user/data/pretrained_models/
hugging_face_cache_path: /home/user/.cache/
dumps_path: "./dumps"

# Directory names
...
```
Refer the full `config_base.yaml` for the default parameter configuration.

## :bomb: Feature Extraction
Follow the instructions in [feature_extractors/README.md](feature_extractors/README.md) to extract required features from MovieGraphs dataset. Note that we have already provided the pre-extracted features above and therefore you need not extract the features again.

## :weight_lifting: Train
After extracting the features and creating the config, you can train EmoTx on a 12GB GPU!<br>
You can also use the pre-trained weights provided in the [Download](#mag-download) section.<br>
Note: the `Eval_mAP: [[A,B], C]` in log line (printed during training) represents the char_mAP, scene_mAP and average of both respectively.<br>
Note: it is recommended to use [wandb](https://wandb.ai)<br>

Using the default values given in the `config_base.yaml`
1. To train EmoTx for MovieGraphs-top10 emotion label set, use the default config (no argument required)
```
(emotx) $ python trainer.py
```
2. To train EmoTx with MovieGraphs-top25 emotion label set-
```
(emotx) $ python trainer.py top_k=25
```
3. To use EmoticMapping label set-
```
(emotx) $ python trainer.py use_emotic_mapping=True
```
4. To use different scene features (valid keywords- `mvit_v1`, `resnet50_places`, `generic`) [generic=ResNet150_ImageNet]
```
(emotx) $ python trainer.py scene_feat_type="mvit_v1"
```
5. To use different character face features (valid keywords- `resnet50_fer`, `emo`, `generic`) [emo=VGG-vm_FER13, generic=InceptionResNetV1_VGGface]
```
(emotx) $ python trainer.py scene_feat_type="resnet50_fer"
```
6. To use fine-tuned/pre-trained subtitle features (valid choices- `False` (to use fine-tuned RoBERTa) | `True` (to use pre-trained RoBERTa))
```
(emotx) $ python trainer.py srt_feat_pretrained=False
```
7. Train with only scene features
```
(emotx) $ python trainer.py use_char_feats=False use_srt_feats=False get_char_targets=False
```
8. To train with only character face features
```
(emotx) $ python trainer.py use_scene_feats=False use_srt_feats=False get_scene_targets=False
```
9. To train with scene and subtitle features
```
(emotx) $ python trainer.py use_char_feats=False get_char_targets=False
```
10. Enable wandb logging (recommended)
```
(emotx) $ python trainer.py wandb.logging=True wandb.project=<PROJECT_NAME> wandb.entity=<WANDB_USERNAME>
```
All the above arguments can be combined to train with different configurations.

## :mag: Download
### :rocket: EmoTx pre-trained weights
| File name | Comments | Training command |
|-----------|----------|----------|
| [EmoTx_Top10.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EUVbB6Mbwy5OpzAoqAz0jxYBsiUCach0dbeTaZWyE9316Q?download=1) | EmoTx trained on MovieGraphs-top10 emotion label set | `(emotx) $ python trainer.py model_no=4.0 top_k=10 ` |
| [EmoTx_Top25.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/ETiTg6kxt89JrJgFV0jkOGkBPXW7f36CxpW2vi_2VrLzYQ?download=1) | EmoTx trained on MovieGraphs-top25 emotion label set | `(emotx) $ python trainer.py model_no=4.0 top_k=25 ` |
| [EmoTx_Emotic.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EYFhKfYaDLxCpAfS6CPLk2ABw5BU_MFkaEjpJGvHyuAYUg?download=1) | EmoTx trained on EmoticMapping emotion label set | `(emotx) $ python trainer.py model_no=4.0 use_emotic_mapping=True ` |

These models can be loaded using the following code-
```python
import torch
from models.emotx import EmoTx

model_checkpoint_filepath = "<PATH_TO_CHECKPOINT>.pt"
chkpt = torch.load(model_checkpoint_filepath)

model = EmoTx(
    num_labels=chkpt["num_labels"],
    num_pos_embeddings=chkpt["num_pos_embeddings"],
    scene_feat_dim=chkpt["scene_feat_dim"],
    char_feat_dim=chkpt["char_feat_dim"],
    srt_feat_dim=chkpt["srt_feat_dim"],
    num_chars=chkpt["num_chars"],
    num_enc_layers=chkpt["num_enc_layers"],
    max_individual_tokens=chkpt["max_individual_tokens"],
    hidden_dim=chkpt["hidden_dim"]
)
model.load_state_dict(chkpt["state_dict"])
```

### :open_hands: Pre-trained feature backbones
| File name | Comments |
|-----------|----------|
| [ResNet50_PL365.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EXMhpmrJqPlMvTnwKg3cwI8B10jZjeYOQR3P7VW9nbayUQ?download=1) | ResNet50 trained on Places365 dataset |
| [MViT_v1_Kinetics400.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EVi2ME3a1N9HnpC4hWDtPikB_27E7HE2AGiv2tvUvEg_hg?download=1) | MViT_v1 trained on Kinetics400 dataset |
| [ResNet50_FER.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EXD44dJWuMxMvPgvHCh8fikB6tWpPzsAvrIULQXmYJ-4VQ?e=umNkgK?download=1) | ResNet50 trained on VGGFace, FER2013 and SFEW datasets |
| [InceptionResNetV1_VGGface.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EbGhXirljZlAq-J0VJUm0IsBBC3zvnC8lZqburadJwjnxg?download=1) | InceptionResNetV1 trained on VGGFace2 dataset |
| [VGG-vm_FER13.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EbGLXEXEW0xNktis5XqgGA8BdhfIELJ0_FfR17RMTgC8tQ?download=1) | VGG-vm trained on VGGface and FER2013 dataset |
| [MTCNN.pth](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EYpoawuM-BNJslyUpKBCkuwBNLAzdYFW1TSa93Vim12Tsg?download=1) and [MTCNN.json](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/Ee6h8i_9hJJMpoKiWQwnhWEBgonsRy0nkz2wCRCvSmu0GA?download=1) | MTCNN model and config used for face detection |
| [Cascade_RCNN_movienet.pth](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EdwBB5CS115Ht6GxyckNq6wBXXg5ia6gBckQHlGO2vGI6Q?download=1) and [Cascade_RCNN_movienet.json](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EQc9VXAMNaVBqrZk0NaYLgEBFgkjkv7va6lO5CAptyj8zA?download=1) | Config and person detection model pre-trained on MovieNet character annotations |
| [RoBERTa_finetuned_t10.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EUVZuWL49VVDsSYJAyXWp9ABl04IHgZAxcqiRGy7J6BRJQ?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Top-10 label set |
| [RoBERTa_finetuned_t25.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EQyudvEGE55MqfgOH8KmHP0Bo7yWo6Ee9eMjVd42z1rtCw?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Top-25 label set |
| [RoBERTa_finetuned_Emotic.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EZhITIWmEMNMpxsblIiGapIB0a_4wmJP4pqptVCSJSlyHw?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Emotic-Mapped label set |


## :round_pushpin: Cite
If you find any part of this repository useful, please cite the following paper!
```
@inproceedings{dhruv2023emotx,
title = {{How you feelin'? Learning Emotions and Mental States in Movie Scenes}},
author = {Dhruv Srivastava and Aditya Kumar Singh and Makarand Tapaswi},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
```
