from pathlib import Path

import os
import pysrt
import re
import yaml
import utils.mg_utils as utils


class clip_srt(object):
    """
    Class to read the srt files and return the text.
    """
    def __init__(self, config, movie_ids):
        """
        Args:
            config (dict): Configuration file.
            movie_ids (list): List of movie ids.
        """
        self.CLIP_SRT_PATH = Path(config['clip_srts_path'])
        self.movie_ids = movie_ids
        self.srts = dict()
        self.time_srt = dict()
        self.read_srts()

    def process_sub(self, sub):
        """
        Process the pysrt subtitle object to get the text.
        The subtitles may break a continuous statement into new lines and can be
        interrupted by special characters. This methods filters the utterances to get
        the correct start and end timestamps along with the proper utterance string.

        Args:
            sub (pysrt subtitle object): Contains utterances with start and end timestamp.

        Returns:
            (list): List of tuples containing start and end timestamp and utterance string.
        """
        quote_matches = re.compile('<.+?>')
        text = sub.text.strip()
        # Get rid of text in parentheses or square brackets
        text = re.sub(r"\([^\)]+\)", "", text)
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"<i>", "", text)
        text = re.sub(r"</i>", "", text)
        text = re.sub(r"\xe2|\x99|\xaa|#", '', text)
        text = quote_matches.sub('', text)
        # If there's nothing left after the bracket elimination, skip the line
        if not text:
            return []
        # If there's no '-', then everything is said by one character, so combine
        # the lines into one long line (could have multiple sentences).
        if text[0] != '-':
            text = ' '.join(text.split('\n'))
            return [(sub.start, sub.end, text)]
        else:
            parts_of_dialog = text.split('\n')
            parts_of_dialog = [part[1:].strip() for part in parts_of_dialog]
            if not parts_of_dialog:
                return []
            if len(parts_of_dialog) == 1:
                return [(sub.start, sub.end, parts_of_dialog[0])]
            if not parts_of_dialog[0] and not parts_of_dialog[1]:
                return []
            elif parts_of_dialog[0] and not parts_of_dialog[1]:
                return [(sub.start, sub.end, parts_of_dialog[0])]
            elif parts_of_dialog[1] and not parts_of_dialog[0]:
                return [(sub.start, sub.end, parts_of_dialog[1])]
            else:
                part_lengths = [len(part) for part in parts_of_dialog]
                total_length = float(sum(part_lengths))
                duration = sub.end - sub.start
                ratio1 = part_lengths[0] / total_length
                ratio2 = part_lengths[1] / total_length
                return [(sub.start, sub.start + duration * ratio1, parts_of_dialog[0]),
                        (sub.start + duration * ratio1, sub.end, parts_of_dialog[1])]

    def read_srts(self):
        """
        Read the srt files and store the text in a dictionary.
        """
        srts = list()
        issues = list()
        for movie_id in self.movie_ids:
            srt_files = os.listdir(self.CLIP_SRT_PATH/movie_id)
            for srt_file in srt_files:
                scene_name = ".".join(srt_file.split('.')[:-1])
                key = Path(movie_id)/scene_name
                srt = pysrt.open(self.CLIP_SRT_PATH/movie_id/srt_file, encoding='cp1252')
                self.srts[key] = ""
                self.time_srt[key] = list()
                for sub in srt:
                    parsed_srt_objs = self.process_sub(sub)
                    self.time_srt[key] += [((obj[0].ordinal+(obj[1].ordinal-obj[0].ordinal)/2)/1000, obj[-1]) for obj in parsed_srt_objs]
                    self.srts[key] += " ".join([obj[-1] for obj in parsed_srt_objs]) + " "
                self.srts[key] = self.srts[key].strip()

    def get_srt(self, movie_scene_id, concat=True):
        """
        Get the text from the srt file.
        Operates in two modes:
            1. Concatenated mode: Returns the entire text from the srt file.
            2. Non-concatenated mode: Returns the text from the srt file along with the
                mid timestamp.

        Args:
            movie_scene_id (Path): Path to the srt file.
            concat (bool): If True, returns the concatenated text. Else, returns the
                text along with the mid timestamp.

        Returns:
            (str or list): Concatenated text or text along with mid timestamp.
        """
        if concat:
            return [self.srts[movie_scene_id]]
        else:
            return self.time_srt[movie_scene_id]
