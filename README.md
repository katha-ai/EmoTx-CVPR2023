<div align="center">
    <h1> :movie_camera: <a href="https://arxiv.org/abs/2304.05634">How you feelin'? Learning Emotions and Mental States in Movie Scenes</a><br>
    <a href="https://img.shields.io/badge/python-3.6-blue"><img src="https://img.shields.io/badge/python-3.6-blue"></a>
    <a href="https://img.shields.io/badge/made_with-pytorch-red"><img src="https://img.shields.io/badge/made_with-pytorch-red"></a>
    <a href="https://img.shields.io/badge/dataset-MovieGraphs-orange"><img src="https://img.shields.io/badge/dataset-MovieGraphs-orange"></a>
    <a href="https://katha-ai.github.io/projects/emotx"><img src="https://img.shields.io/website?up_message=up&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Femotx&link=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Femotx"></a>
    <a href="https://arxiv.org/abs/2304.05634"><img src="https://img.shields.io/badge/arXiv-2304.05634-f9f107.svg"></a>
    <a href="https://katha-ai.github.io/projects/emotx"><img src="docs/assets/emotx.gif" style="width: 800px;"></a>
</div>

## :bookmark: Contents
1. [About](#about)
2. [Setting up the repository](#setting-up-the-repository)
    1. [Create a virtual environment](#create-a-python-virtual-environment)
    2. [Setup the data directory](#download-the-moviegraphs-features)
    3. [Update the config template](#create-the-configyaml)
    4. [Train!](#train)
4. [Download](#download)
    1. [EmoTx pre-trained weights](#emotx-pre-trained-weights)
    2. [Pre-trained feature backbones](#pre-trained-feature-backbones)
3. [Bibtex](#cite)

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
| [EmoTx_min_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EdbbcQvEaBlIg6Sktxw60lQBiOUyDWdbKf3GhF88mrEhaA?download=1) | <ul><li>Extended character tracks</li><li>`emotic_mapping.json`</li><li>MovieGraphs pickle</li><li>MViT_v1_Kinetics400-SceneFeatures</li><li>ResNet50-FER13-FaceFeatures</li><li>Subtitle features</li><li>All pre-trained backbones used in EmoTx</li></ul> | contains `data/` directory which will occupy 167GB of disk space. |
| [InceptionResNetV1_VGGface_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaTYYi2G2DNCpUaO7-Hl9PgBaXgCedq3QMxbuVos3Sfa7A?download=1) | Character face features extracted from InceptionResNet_v1 model pre-trained on VGGface2 dataset. | Contains `generic_face_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_face_features/` will occupy 32GB of disk space. |
| [VGG-vm_FER13_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/Ecx72Es1AdBGnT1Zr0U2R0cBjsx5WXP1nNAHjW2-3CdtbA?download=1) | Character face features  extracted from VGG-vm model pretrained on VGGFace and FER13 datasets | Contains `emo_face_features/` directory. TO use these features with EmoTx, move this directory inside `data/` extracted with `EmoTx_min_feats.tar.gz`. After extraction, `emo_face_features/` will occupy xxGB of disk sace.|
| [ResNet150_ImgNet_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaPRbRO5hmVFtOXgkb-mLCkBX11y2T4dzwdXXSLbX0eAtw?download=1) | Scene (full frame) features extracted from ResNet150 model pre-trained on ImageNet dataset | Contains `generic_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_scene_features/` will occupy xxGB of disk space.  |
| [ResNet50_PL365_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaSkvXnjw6tBsXsRYf41yTgBbafy63-Nen_MRXulx0ycQA?download=1) | Scene (full frame) features extracted from ResNet50 model pre-trained on Places365 dataset. | Contains `resnet50_places_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `resnet50_places_scene_features/` will occupy xxGB of disk space. |
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

### :weight_lifting: Train
After extracting the features and creating the config, you can train EmoTx on a 12GB GPU!
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

## :mag: Download
### :rocket: EmoTx pre-trained weights
| File name | Comments | Training command |
|-----------|----------|----------|
| [EmoTx_Top10.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EflUuVhuoyZCsIoWuwIfGtYB-NcGHQGym_D3Ww5bxXbsRg?download=1) | EmoTx trained on MovieGraphs-top10 emotion label set | `(emotx) $ python trainer.py model_no=4.0 top_k=10 ` |
| [EmoTx_Top25.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EVe0p54Qd0VGlROxoOywzesBYl5PuDf-wivTciH81p5eZA?download=1) | EmoTx trained on MovieGraphs-top25 emotion label set | `(emotx) $ python trainer.py model_no=4.0 top_k=25 ` |
| [EmoTx_Emotic.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EetOBsV898hOhiV1oGEoJ3MBZkLx3n8dnLk_zxuzsBVoGQ?download=1) | EmoTx trained on EmoticMapping emotion label set | `(emotx) $ python trainer.py model_no=4.0 use_emotic_mapping=True ` |


### :running: Pre-trained feature backbones
| File name | Comments |
|-----------|----------|
| [ResNet50_PL365.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EXMhpmrJqPlMvTnwKg3cwI8B10jZjeYOQR3P7VW9nbayUQ?download=1) | ResNet50 trained on Places365 dataset |
| [MViT_v1_Kinetics400.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EVi2ME3a1N9HnpC4hWDtPikB_27E7HE2AGiv2tvUvEg_hg?download=1) | MViT_v1 trained on Kinetics400 dataset |
| [ResNet50_FER.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EXD44dJWuMxMvPgvHCh8fikB6tWpPzsAvrIULQXmYJ-4VQ?e=umNkgK?download=1) | ResNet50 trained on VGGFace, FER2013 and SFEW datasets |
| [InceptionResNetV1_VGGface.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EbGhXirljZlAq-J0VJUm0IsBBC3zvnC8lZqburadJwjnxg?download=1) | InceptionResNetV1 trained on VGGFace2 dataset |
| [VGG-vm_FER13.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EbGLXEXEW0xNktis5XqgGA8BdhfIELJ0_FfR17RMTgC8tQ?download=1) | VGG-vm trained on VGGface and FER2013 dataset |
| [RoBERTa_finetuned_t10.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EUVZuWL49VVDsSYJAyXWp9ABl04IHgZAxcqiRGy7J6BRJQ?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Top-10 label set |
| [RoBERTa_finetuned_t25.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EQyudvEGE55MqfgOH8KmHP0Bo7yWo6Ee9eMjVd42z1rtCw?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Top-25 label set |
| [RoBERTa_finetuned_Emotic.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EZhITIWmEMNMpxsblIiGapIB0a_4wmJP4pqptVCSJSlyHw?download=1) | RoBERTa fine-tuned on MovieGraphs dataset with Emotic-Mapped label set |


## :v: Cite
If you find any part of this repository useful, please cite the following paper!
```
@inproceedings{dhruv2023emotx,
title = {{How you feelin'? Learning Emotions and Mental States in Movie Scenes}},
author = {Dhruv Srivastava and Aditya Kumar Singh and Makarand Tapaswi},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
```
