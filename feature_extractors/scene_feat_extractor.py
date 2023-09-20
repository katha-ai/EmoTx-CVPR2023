from models.pretrained_models import Places365_ResNet50, ImageNet_ResNet152
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from transformers import CLIPProcessor
from decord import VideoReader

import numpy as np
import os
import sys
import time
import torch
import yaml


class SceneNormalizer_ResNet50_places365(object):
    def __init__(self):
        self.roi_shape = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        self.norm = T.Normalize(self.mean, self.std)

    def __call__(self, img):
        return self.norm(self.transform(img))


class SceneNormalizer_ResNet152_ImageNet(object):
    def __init__(self):
        self.roi_shape = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.norm = T.Compose([T.ToTensor(),
                               T.Normalize(mean=self.mean,
                                           std=self.std)])

    def __call__(self, img):
        return self.norm(img)


class scene_feture_extractor(object):
    def __init__(self, config):
        self.config = config
        self.movie_dir = Path(config["data_path"])/config["mg_videos_dir"]
        self.tracks_path = Path(config["data_path"])/config["tracks_dir"]
        self.save_path = Path(config["save_path"])
        self.emb_save_dir = config["scene_feat_type"]+config["scene_feat_dir"]
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.batch_size = self.config['batch_size']
        if config["scene_feat_type"] == "resnet50_places":
            print("Selected ResNet50_Places365 for scene feature extraction")
            self.context_model = Places365_ResNet50(str(Path(self.config["saved_model_path"])/self.config["feat_info"][config["scene_feat_type"]]["weights"]), self.device).eval().to(self.device)
            self.context_normalizer = SceneNormalizer_ResNet50_places365()
            self.context_roi_shape = self.context_normalizer.roi_shape
            self.context_embedding_size = self.config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
        elif config["scene_feat_type"] == "generic":
            print("Selected ResNet152_ImageNet for scene feature extraction")
            self.context_model = ImageNet_ResNet152().eval().to(self.device)
            self.context_normalizer = SceneNormalizer_ResNet152_ImageNet()
            self.context_roi_shape = self.context_normalizer.roi_shape
            self.context_embedding_size = self.config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
        else:
            raise NotImplementedError("Scene feature type {} not implemented".format(config["scene_feat_type"]))

    def get_video_object(self, movie_id, scene):
        clip_path = str(self.movie_dir/movie_id/(scene+'.mp4'))
        vid = VideoReader(clip_path)
        return vid

    @torch.no_grad()
    def get_context_embeddings(self, vid):
        frame_ndxs = list(range(len(vid)))
        embeddings = np.empty((0, self.context_embedding_size))
        for batch_ndx in range(0, len(vid), self.batch_size):
            batch_frames = torch.empty((0, 3, self.context_roi_shape[1], self.context_roi_shape[0]))
            for fr_ndx in frame_ndxs[batch_ndx:batch_ndx+self.batch_size]:
                fr = np.array(Image.fromarray(vid[fr_ndx].asnumpy()).resize((self.context_roi_shape[1], self.context_roi_shape[0])))
                fr = self.context_normalizer(fr)
                batch_frames = torch.vstack([batch_frames, fr.unsqueeze(0)])
            emb = self.context_model(batch_frames.to(self.device))
            embeddings = np.vstack([embeddings, emb.detach().cpu()])
        return embeddings

    def context_feat_extractor(self, movie_id):
        print("Triggered context feature extraction for {}".format(movie_id))
        save_path = self.save_path/self.emb_save_dir/movie_id
        save_path.mkdir(parents=True, exist_ok=True)
        track_pkls = os.listdir(self.tracks_path/movie_id)
        scene_names = ['.'.join(pkl.split('.')[:-1]) for pkl in track_pkls]
        for scene_name in scene_names:
            start = time.perf_counter()
            video = self.get_video_object(movie_id, scene_name)
            emb = self.get_context_embeddings(video)
            np.save(save_path/scene_name, emb)
            log_line = "{}/{} | Embeddings shape: {} | total_frames: {} | time taken {:.4f} sec."
            print(log_line.format(movie_id, scene_name, emb.shape, len(video), time.perf_counter()-start))

    def runner(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        movies = os.listdir(self.tracks_path)
        for movie in movies:
            self.context_feat_extractor(movie)


if __name__ == "__main__":
    configname = sys.argv[1] if sys.argv[1:] else "config.yaml"
    with open(configname, 'r') as f:
        config = yaml.safe_load(f)
    obj = scene_feture_extractor(config)
    obj.runner()
