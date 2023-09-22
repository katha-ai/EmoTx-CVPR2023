from decord import VideoReader
from multiprocessing import Process, current_process
from pathlib import Path
from utils.sort import *

import os
import pickle
import sys
import time
import yaml


class char_tracker(object):
    def __init__(self, config):
        self.detections_path = Path(config["char_detection"]["save_path"])
        self.movies_dir = Path(config["data_path"])/config["mg_videos_dir"]
        self.save_path = Path(config["save_path"])/"character_tracks"
        self.num_cpus = config["num_cpus"]

    def get_shot_frames(self, videvents):
        with open(videvents, "r") as f:
            shot_frames = f.readlines()
        shot_frames = [int(st.split(' ')[0]) for st in shot_frames[1:]]
        shot_frames.reverse()
        return shot_frames

    def get_detections(self, dets):
        with open(dets, "rb") as f:
            ptracks = pickle.load(f)
            ftracks = pickle.load(f)
        return ptracks, ftracks

    def person_tracker(self, dets, shot_frames, clip, max_age=5, min_hits=0, iou_thresh=0.3):
        tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
        tracker.flush_existing_trackers()
        total_frames = len(VideoReader(clip))
        det_frames = list(dets.keys())
        det_frames.reverse()
        frame_tracks = dict()
        bboxes = list()
        frame_pid_track_mapping, fr_pid_trs = dict(), list()
        for i in range(total_frames):
            if len(shot_frames) and i >= shot_frames[-1]:
                _ = shot_frames.pop()
                tracker.flush_existing_trackers()
                bboxes = list()
            if len(det_frames) and i == det_frames[-1]:
                bboxes, fr_pid_trs = tracker.update(dets[i])
                _ = det_frames.pop()
            else:
                empty_dets = np.empty((0, 5))
                bboxes, fr_pid_trs = tracker.update(empty_dets)
            if len(bboxes):
                for mp in fr_pid_trs:
                    frame_pid_track_mapping[(mp[0])] = mp[1]
                frame_tracks[i] = bboxes
        tracker.reset_counting()
        return frame_tracks, frame_pid_track_mapping

    def map_ptracks_with_fdets(self, fdets, fr_pid_tid_map):
        ftracks = dict()
        for frame_no, dets in fdets.items():
            tracks = np.empty((0, 8))
            for ndx, det in enumerate(dets):
                tid = fr_pid_tid_map[(frame_no, det[-1])]
                tracks = np.vstack([tracks, np.concatenate([fdets[frame_no][ndx], [tid]])])
            ftracks[frame_no] = tracks
        return ftracks

    def save_tracks(self, save_path, scene_name, person_tracks, face_tracks):
        f = open(save_path/(scene_name+".pkl"), "wb")
        pickle.dump(person_tracks, f)
        pickle.dump(face_tracks, f)
        f.close()

    def runner(self, movies):
        total_movies = len(movies)
        for movie in movies:
            start_time = time.time()
            print("Working on movie: {} | Pending: {}".format(movie, total_movies-1))
            scenes = [".".join(sc.split('.')[:-1]) for sc in os.listdir(self.detections_path/movie)]
            save_path = self.save_path/movie
            save_path.mkdir(parents=True, exist_ok=True)
            for scene in scenes:
                clip = str(self.movies_dir/movie/(scene+".mp4"))
                videvents = self.movies_dir/movie/(scene+".videvents")
                detections = self.detections_path/movie/(scene+".pkl")
                shot_frames = self.get_shot_frames(videvents)
                pdets, fdets = self.get_detections(detections)
                ptracks, frame_pid_track_mapping = self.person_tracker(pdets, shot_frames[:], clip)
                ftracks = self.map_ptracks_with_fdets(fdets, frame_pid_track_mapping)
                self.save_tracks(save_path, scene, ptracks, ftracks)
            end_time = time.time()
            print("Completed movie: {} | Time taken: {:.4f} sec.\n".format(movie, end_time-start_time))
            total_movies -= 1
        print("Finished all movies in {}\n".format(current_process().name))

    def start_tracking(self):
        print("Making the directory {} and ignoring if it exists".format(self.save_path))
        self.save_path.mkdir(parents=True, exist_ok=True)
        movie_dirs = os.listdir(self.detections_path)
        groups = list(np.array_split(movie_dirs, self.num_cpus))
        processes = list()
        for pr_ndx in range(self.num_cpus):
            pr = Process(target=self.runner, args=(groups[pr_ndx], ), name="Process_{}".format(pr_ndx))
            pr.start()
            processes.append(pr)
            print("Started process {}".format(pr))
        for pr in processes:
            pr.join()
        print("Finished Execution")


if __name__ == "__main__":
    config_name = sys.argv[1] if sys.argv[1:] else "config.yaml"
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    tracker_obj = char_tracker(config)
    tracker_obj.start_tracking()
