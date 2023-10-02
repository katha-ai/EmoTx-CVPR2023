# Feature extraction from the MovieGraphs dataset

This module was used to extract scene, face and subtitle (dialogue) features from the MovieGraphs dataset. <br>
Note: we have already provided all the extracted features (refer [../README.md](https://github.com/katha-ai/EmoTx-CVPR2023/blob/master/README.md#stars-download-the-moviegraphs-features)).
Therefore, you need not extract the features again. <br>

> Make sure your working directory is in project root and not inside `feature_extractors` module

## Save the pre-trained checkpoints
Download the required checkpoint from [here](https://github.com/katha-ai/EmoTx-CVPR2023/blob/master/README.md#open_hands-pre-trained-feature-backbones) and save it in the directory mentioned at `config.yaml:saved_model_path`.

## Create the new environment
This module is separate from EmoTx. <br>
For feature extraction, we require `python>=3.8`. Therefore, you will have to create another environment for this.

Using Coda-
```
$ conda create -n mg_ft_extr python=3.8
$ conda activate mg_ft_extr
(mg_ft_extr) $ pip install -r feature_extractors/feat_extraction_requirements_py38.txt
```
Or using pip (make sure you have `python3.8` installed)-
```
$ python3.8 -m pip install virtualenv
$ python3.8 -m virtualenv mg_ft_extr
$ source mg_ft_extr/bin/activate
(mg_ft_extr) $ pip install -r feature_extractors/feat_extraction_requirements_py38.txt
```

## Extract scene features
1. Extract features from MViT_V1 pre-trained on Kinetics400 dataset
```
(mg_ft_extr) $ python -m feature_extractors.action_feat_extractor scene_feat_type="mvit_v1"
```

2. Extract features from ResNet50 pre-trained on Places365 dataset
```
(mg_ft_extr) $ python -m feature_extractors.scene_feat_extractor scene_feat_type="resnet50_places"
```

3. Extract features from ResNet152 pre-trained on ImageNet dataset
```
(mg_ft_extr) $ python -m feature_extractors.scene_feat_extractor scene_feat_type="generic"
```

## Extract subtitle features
If you wish to extract the dialogue features from a fine-tuned RoBERTa (fine-tuned using [../finetune_roberta.py](https://github.com/katha-ai/EmoTx-CVPR2023/blob/master/finetune_roberta.py)),
Make sure the fine-tuned model with the appropriate name is saved in the path mentioned in `config.yaml:saved_model_path`<br>
The command line argument `srt_feat_pretrained=True` implies we will use the pre-trained RoBERTa checkpoint whereas `srt_feat_pretrained=False` implies we will use the fine-tuned RoBERTa checkpoint.

1. Extract utterance level features from the fine-tuned RoBERTa model-
```
(mg_ft_extr) $ python -m feature_extractors.srt_features_extractor srt_feat_type="independent" srt_feat_pretrained=False
```
2. Extract concatenated utterance features from the fine-tuned RoBERTa model-
```
(mg_ft_extr) $ python -m feature_extractors.srt_features_extractor srt_feat_type="concat" srt_feat_pretrained=False
```
3. Extract utterance level features from pre-trained RoBERTa-base checkpoint-
```
(mg_ft_extr) $ python -m feature_extractors.srt_features_extractor srt_feat_type="independent" srt_feat_pretrained=True
```
4. Extract concatenated utterance features from pre-trained RoBERTa-base checkpoint-
```
(mg_ft_extr) $ python -m feature_extractors.srt_features_extractor srt_feat_type="concat" srt_feat_pretrained=True
```

## Face detection and tracking
We use MTCNN for Face detection and a CascadeRCNN pre-trained with MovieNet annotations for Person detection. We first detect the person bbox and then detect faces within the person box. This minimizes the false-positive face detections.
1. Edit the `config.yaml:char_detection.save_path`
2. Perform character detection-
```
(mg_ft_extr) $ python -m feature_extractors.character_detector
```
3. Once the detection is over, proceed to character tracking
```
(mg_ft_extr) $ python -m feature_extractors.character_tracker
```
Note: This will generate a directory named `character_tracks/` in the `config.yaml:save_path`. Move it inside the `data/` directory that is mentioned at `config.yaml:data_path`.

## Extract face features
Make sure you have performed character detection and tracking before performing this and moved the `character_tracks/` directory to `data/` directory.
1. Extract face features from InceptionResNet_v1 pre-trained on VGG-Face2 dataset-
```
(mg_ft_extr) $ python -m feature_extractors.face_feat_extractor face_feat_type="generic"
```
2. Extract face features from ResNet50 pre-trained on SFEW, FER13 and VGG-Face datasets
```
(mg_ft_extr) $ python -m feature_extractors.face_feat_extractor face_feat_type="resnet50_fer"
```
3. Extract face features from VGG-m pre-trained on FER13 and VGG-Face datasets
```
(mg_ft_extr) $ python -m feature_extractors.face_feat_extractor face_feat_type="emo"
``` 
