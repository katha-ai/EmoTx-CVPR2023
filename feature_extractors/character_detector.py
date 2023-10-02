from feature_extractors import get_config
from utils.movienet_tools_subset.facedetector import FaceDetector
from utils.movienet_tools_subset.persondetector import PersonDetector
from pathlib import Path

import cv2
import numpy as np
import os
import pickle
import sys
import time
import torch


class CharacterDetector(object):
    def __init__(self, config):
        self.config = config
        self.gpu_id = self.config["gpu_id"]
        self.extns = ["mp4", "avi"]
        self.movies_dir = Path(config["data_path"])/config["mg_videos_dir"]
        self.model_dir = Path(config['saved_model_path'])
        self.per_det_cfg = self.model_dir/config["char_detection"]["person_config"]
        self.face_cfg = self.model_dir/config["char_detection"]["face_config"]
        self.per_det_model = str(self.model_dir/config["char_detection"]["person_det_model"])
        self.face_det_model = str(self.model_dir/config["char_detection"]["face_det_model"])
        self.per_detector = PersonDetector('rcnn', self.per_det_cfg, self.per_det_model, gpu=self.gpu_id)
        self.face_detector = FaceDetector(self.face_cfg, self.face_det_model, gpu=self.gpu_id)
        self.save_path = Path(config["char_detection"]["save_path"])

    def detect_persons(self, frame, frame_no):
        person_detections = self.per_detector.detect(frame, conf_thr=0.90)
        per_imgs = self.per_detector.crop_person(frame, person_detections, img_scale=None)
        if len(per_imgs):
            frame_per_marking = np.vstack([np.array([frame_no, ndx]) for ndx in range(person_detections.shape[0])])
            person_detections = np.hstack([person_detections[:, :4], frame_per_marking])
        return person_detections, per_imgs

    def detect_face_fom_person(self, per_img, per_frame_coord, frame_no, per_no):
        top_left_x, top_left_y = max(0, per_frame_coord[0]), max(0, per_frame_coord[1])
        faces, _ = self.face_detector.detect(per_img, conf_thr=0.70)
        faces[:, 0] += max(0, top_left_x)
        faces[:, 1] += max(0, top_left_y)
        faces[:, 2] += max(0, top_left_x)
        faces[:, 3] += max(0, top_left_y)
        selected_face = np.empty((0, 7))
        if faces.size:
            selected_face = faces[np.argmax(faces[:, 4])]
            selected_face = np.concatenate([selected_face, np.array([frame_no, per_no])])
        return selected_face

    def detector(self, save_path, movie_id, scene_name):
        clip = str(self.movies_dir/movie_id/(scene_name+".mp4"))
        video = cv2.VideoCapture(clip)
        frame_no = 0
        pdets, fdets = dict(), dict()
        while True:
            stat, frame = video.read()
            if not stat:
                break
            per_detections, per_imgs = self.detect_persons(frame, frame_no)
            face_detections = np.empty((0, 7))
            for ndx, per_img in enumerate(per_imgs):
                if per_img.shape[0] >=50 and per_img.shape[1] >= 20:
                    face = self.detect_face_fom_person(per_img, per_detections[ndx], frame_no, ndx)
                    face_detections = np.vstack([face_detections, face])
            if per_detections.size:
                pdets[frame_no] = per_detections
            if face_detections.size:
                fdets[frame_no] = face_detections
            frame_no += 1
        print(save_path/(scene_name+".pkl"))
        f = open(save_path/(scene_name+".pkl"), "wb")
        pickle.dump(pdets, f)
        pickle.dump(fdets, f)
        f.close()

    def start_detection_over_MovieGraphs(self):
        print("Creating folder {} and ignoring if it already exists.".format(self.save_path))
        self.save_path.mkdir(parents=True, exist_ok=True)
        movie_dirs = os.listdir(self.movies_dir)
        pending_count = len(movie_dirs)
        for movie_id in movie_dirs:
            pending_count -= 1
            if not os.path.isdir(self.movies_dir/movie_id):
                continue
            print("Starting movie: {} | Pending: {}".format(movie_id, pending_count))
            save_path = self.save_path/movie_id
            save_path.mkdir(parents=True, exist_ok=True)
            files = os.listdir(self.movies_dir/movie_id)
            for file in files:
                if file.split('.')[-1] in self.extns:
                    scene_name = ".".join(file.split(".")[:-1])
                    self.detector(save_path, movie_id, scene_name)
        print("Finished all movies")


if __name__ == "__main__":
    config = get_config()
    obj = CharacterDetector(config)
    obj.start_detection_over_MovieGraphs()
