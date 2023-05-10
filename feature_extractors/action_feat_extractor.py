from models.pretrained_models import MViT_v1
from pathlib import Path
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from decord import VideoReader

import os
import numpy as np
import time
import torch
import yaml


class ActionFeatNormalizer_MViT(object):
    def __init__(self):
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = (224,224)
        self.frame_group_size = 32
        self.transform =  Compose([
                            Lambda(lambda x: x/255.0),
                            NormalizeVideo(self.mean, self.std),
                            ShortSideScale(size=256),
                            CenterCropVideo(crop_size=self.crop_size),
                        ])

    def __call__(self, frames):
        return self.transform(frames)


class motion_feat_extractor(object):
    def __init__(self, config):
        self.config = config
        self.save_path = Path(config["save_path"])
        self.movie_dir = Path(config["data_path"])/config["mg_videos_dir"]
        self.tracks_path = Path(config["data_path"])/config["tracks_dir"]
        self.emb_save_dir = config["scene_feat_type"]+config["scene_feat_dir"]
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        if config["scene_feat_type"] == "mvit_v1":
            print("Selected MViT_v1 for action feature extraction")
            self.action_model = MViT_v1(str(Path(self.config["saved_model_path"])/self.config["mvit_v1_weights"])).eval().to(self.device)
            self.action_normalizer = ActionFeatNormalizer_MViT()
            self.action_embedding_size = self.config["feat_info"][config["scene_feat_type"]]["scene_feat_dim"]
            self.frame_group_size = self.action_normalizer.frame_group_size

    def get_video_object(self, movie_id, scene):
        clip_path = str(self.movie_dir/movie_id/(scene+".mp4"))
        vid = VideoReader(clip_path)
        return vid

    @torch.no_grad()
    def get_action_embeddings(self, vid):
        embeddings = np.empty((0, self.action_embedding_size))
        for batch_ndx in range(0, len(vid), self.frame_group_size//2):
            frames = vid[batch_ndx:batch_ndx+self.frame_group_size].asnumpy()
            frames = torch.FloatTensor(frames).permute(3,0,1,2)
            frames = self.action_normalizer(frames)
            pad_frame_count = self.frame_group_size-frames.shape[1]
            h, w = self.action_normalizer.crop_size[1], self.action_normalizer.crop_size[0]
            pad_vec = torch.zeros(3, pad_frame_count, h, w)
            frames = torch.cat([frames, pad_vec], dim=1)
            emb = self.action_model(frames.unsqueeze(0).to(self.device))
            embeddings = np.vstack([embeddings, emb.detach().cpu()])
        return embeddings

    def action_feat_extractor(self, movie_id):
        print("Triggered action feature extraction for {}".format(movie_id))
        save_path = self.save_path/self.emb_save_dir/movie_id
        save_path.mkdir(parents=True, exist_ok=True)
        track_pkls = os.listdir(self.tracks_path/movie_id)
        scene_names = ['.'.join(pkl.split('.')[:-1]) for pkl in track_pkls]
        for scene_name in scene_names:
            start = time.perf_counter()
            vid = self.get_video_object(movie_id, scene_name)
            emb = self.get_action_embeddings(vid)
            np.save(save_path/scene_name, emb)
            # log_line = "{}/{} | Embeddings shape: {} | time taken {:.4f} sec."
            # print(log_line.format(movie_id, scene_name, emb.shape, time.perf_counter()-start))

    def runner(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        movies = os.listdir(self.tracks_path)
        for movie in movies:
            st = time.perf_counter()
            self.action_feat_extractor(movie)
            print("Completed {} in {:.4f} sec.".format(movie, time.perf_counter()-st))


if __name__ == "__main__":
    configname = "config.yaml"
    with open(configname, 'r') as f:
        config = yaml.safe_load(f)
    obj = motion_feat_extractor(config)
    obj.runner()
