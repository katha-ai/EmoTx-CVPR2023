from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from models.pretrained_models import VGGF2_face, Resnet50_FER
from torchvision import transforms as T
from decord import VideoReader

import numpy as np
import os
import pickle
import sys
import time
import torch
import yaml


class FaceNormalizer_IRV1(object):
    def __init__(self):
        self.roi_shape = (160, 160)
        self.normalizer = fixed_image_standardization

    def __call__(self, img):
        return T.ToTensor()(self.normalizer(np.float32(img)))


class FaceNormalizer_VGGF2(object):
    def __init__(self):
        self.roi_shape = (224, 224)
        self.mean = [131.45376586914062, 103.98748016357422, 91.46234893798828]
        self.std = [1,1,1]
        self.normalizer = T.Normalize(mean=self.mean, std=self.std)

    def __call__(self, img):
        tensor_imgs = T.ToTensor()(img)
        rev_norm_imgs = tensor_imgs*255.0
        norm_imgs = self.normalizer(rev_norm_imgs)
        return norm_imgs

class FaceNormalizer_ResNet50(object):
    def __init__(self):
        self.roi_shape = (224, 224)
        self.mean = [131.88330078125, 105.51170349121094, 92.56940460205078]
        self.std = [1, 1, 1]
        self.normalizer = T.Normalize(mean=self.mean, std=self.std)

    def __call__(self, img):
        tensor_imgs = T.ToTensor()(img)
        rev_norm_imgs = tensor_imgs*255.0
        norm_imgs = self.normalizer(rev_norm_imgs)
        return norm_imgs


class character_feture_extractor(object):
    def __init__(self, config):
        self.config = config
        self.movie_dir = Path(config["data_path"])/config['mg_videos_dir']
        self.tracks_path = Path(config["data_path"])/"character_tracks"
        self.save_path = Path(config['save_path'])
        self.emb_save_dir = config["face_feat_type"]+config["face_feat_dir"]
        self.device = torch.device('cuda:{}'.format(config['gpu_id']) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.batch_size = self.config['batch_size']
        if config["face_feat_type"] == "resnet50_fer":
            print("Selected ResNet50 model pretrained on FER, SFEW and VGG Face datasets")
            weights_file = self.config["feat_info"][self.config["face_feat_type"]]['weights']
            self.face_model = Resnet50_FER(str(Path(self.config["saved_model_path"])/weights_file)).eval().to(self.device)
            self.normalizer = FaceNormalizer_ResNet50()
            self.roi_shape = self.normalizer.roi_shape
        elif config["face_feat_type"] == "emo":
            print("Selected VGG-m model pretrained on FER VGG Face datasets")
            weights_file = self.config["feat_info"][self.config["face_feat_type"]]['weights']
            self.face_model = VGGF2_face(str(Path(self.config["saved_model_path"])/weights_file)).eval().to(self.device)
            self.normalizer = FaceNormalizer_VGGF2()
            self.roi_shape = self.normalizer.roi_shape
        elif config["face_feat_type"] == "generic":
            print("Selected InceptionResNet-V1 (generic) pretrained on VGGFace2 dataset")
            self.face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.normalizer = FaceNormalizer_IRV1()
            self.roi_shape = self.normalizer.roi_shape
        else:
            raise NotImplementedError
        self.embedding_size = self.config["feat_info"][self.config["face_feat_type"]]['face_feat_dim']

    def get_video_object(self, movie_id, scene):
        clip_path = str(self.movie_dir/movie_id/(scene+'.mp4'))
        vid = VideoReader(clip_path)
        return vid

    def read_tracks(self, movie_id, scene):
        f = open(self.tracks_path/movie_id/(scene+".pkl"), 'rb')
        _ = pickle.load(f)
        ftracks = pickle.load(f)
        f.close()
        return ftracks

    def extract_regions(self, frame, tracks, img_shape=None):
        h, w = frame.shape[:2]
        regions = list()
        for track in tracks:
            x1, y1, x2, y2 = track[:4].astype(np.int32)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            region = frame[y1:y2, x1:x2, :]
            if img_shape:
                region = np.array(Image.fromarray(region).resize(img_shape))
            regions.append(region)
        return regions

    @torch.no_grad()
    def get_embeddings(self, vid, tracks, normalizer, img_shape, model, emb_size):
        frames_ndxs = list(tracks.keys())
        embeddings = np.empty((0, emb_size))
        for batch_ndx in range(0, len(frames_ndxs), self.batch_size):
            rois = torch.empty((0, 3, img_shape[1], img_shape[0]))
            for fr_ndx in frames_ndxs[batch_ndx:batch_ndx+self.batch_size]:
                frame = np.array(Image.fromarray(vid[fr_ndx].asnumpy()))
                region_crops = self.extract_regions(frame, tracks[fr_ndx], img_shape=img_shape)
                for roi in region_crops:
                    roi = normalizer(roi)
                    rois = torch.vstack([rois, roi.unsqueeze(0)])
            emb = model(rois.to(self.device))
            embeddings = np.vstack([embeddings, emb.detach().cpu()])
        return embeddings

    def feat_extractor(self, movie_id):
        print("Triggered Face feature extraction for {}".format(movie_id))
        save_path = self.save_path/self.emb_save_dir/movie_id
        save_path.mkdir(parents=True, exist_ok=True)
        track_pkls = os.listdir(self.tracks_path/movie_id)
        scene_names = ['.'.join(pkl.split('.')[:-1]) for pkl in track_pkls]
        for scene_name in scene_names:
            start = time.time()
            video = self.get_video_object(movie_id, scene_name)
            tracks = self.read_tracks(movie_id, scene_name)
            emb = self.get_embeddings(video, tracks, self.normalizer, self.roi_shape, self.face_model, self.embedding_size)
            np.save(save_path/scene_name, emb)
            print("{}/{} | Embeddings shape: {} | time taken {:.4f} sec.".format(movie_id, scene_name, emb.shape, time.time()-start))

    def extract_char_face_features(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        movies = os.listdir(self.tracks_path)
        for movie in movies:
            start = time.time()
            self.feat_extractor(movie)
            time_taken = (time.time()-start)/3600
            print("Finished movie: {} | Time taken: {:.4f}\n".format(movie, time_taken))


if __name__ == "__main__":
    config_filename = sys.argv[1] if sys.argv[1:] else "config.yaml"
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    obj = character_feture_extractor(config)
    obj.extract_char_face_features()
